"""All SQLite, ranking, and inference work belongs to one worker thread."""

from threading import Event

from .cancellation import ScanCancelled
from .db import IMAGE_SELECT, folders, open_database, to_image, vector_from
from .embedding import Embedder
from .groups import generate_groups
from .indexer import initial_progress, scan
from .vectors import Vectors


class Service:
    def __init__(self, config, report, embedder_factory=Embedder):
        self.config, self.report = config, report
        self.embedder = embedder_factory(config)
        self.db = None
        self.vectors = None
        self.progress = initial_progress()
        self.stopping = Event()

    def stop(self):
        self.stopping.set()

    def check_running(self):
        if self.stopping.is_set():
            raise ScanCancelled("Shutdown requested")

    def publish(self, **updates):
        self.check_running()
        self.progress = self.progress | updates
        self.report(self.progress | {"errors": list(self.progress["errors"])})

    def rescan(self):
        self.progress = initial_progress()
        try:
            self.publish(message="Loading EmbeddingGemma 2")
            if self.db is None:
                self.db = open_database(self.config)
                self.vectors = Vectors(self.db)
            self.embedder.load()
            self.check_running()

            def update(progress):
                self.progress = progress
                self.publish()

            scan(self.config, self.db, self.embedder, update, self.check_running)
            self.publish(stage="ranking", message="Building vector index")
            self.vectors.rebuild(
                lambda count: self.publish(
                    message=f"Building vector index · {count:,} images"
                )
            )
            self.publish(stage="grouping", message="Training discovery groups")
            generate_groups(
                self.db,
                lambda done, total: self.publish(
                    message=f"Grouping images · {done:,} / {total:,}"
                ),
                self.check_running,
            )
            self.publish(
                busy=False,
                stage="ready",
                etaSeconds=None,
                message=f"Ready · {self.progress['failed']} errors; see scan details"
                if self.progress["failed"]
                else "Library ready",
            )
        except ScanCancelled:
            self.progress = self.progress | dict(
                busy=False, stage="stopped", message="Indexing stopped", etaSeconds=None
            )
            self.report(self.progress)
        except Exception as exception:
            if self.stopping.is_set():
                return
            self.publish(busy=False, stage="error", message=str(exception))

    def close(self):
        if self.db is not None:
            self.db.close()

    def require_ready(self):
        self.check_running()
        if self.progress["stage"] != "ready":
            raise ValueError("Library is not ready; resolve the scan error and rescan")

    def get_image(self, image_id):
        self.require_ready()
        row = self.db.execute(IMAGE_SELECT + " AND i.id=?", (image_id,)).fetchone()
        if not row:
            raise ValueError("Image not found")
        return to_image(row)

    def get_folders(self):
        self.require_ready()
        return folders(self.db)

    def get_images(self, page, page_size, folder_id=None, group_id=None):
        self.require_ready()
        condition, args = "", []
        if folder_id:
            if not self.db.execute(
                "SELECT id FROM folders WHERE id=?", (folder_id,)
            ).fetchone():
                raise ValueError("Folder not found")
            condition += " AND i.folder_id=?"
            args.append(folder_id)
        if group_id is not None:
            condition += " AND i.group_id=?"
            args.append(group_id)
        total = self.db.execute(
            "SELECT COUNT(*) FROM images i WHERE i.status='ready'" + condition, args
        ).fetchone()[0]
        rows = self.db.execute(
            IMAGE_SELECT
            + condition
            + " ORDER BY i.path COLLATE NOCASE LIMIT ? OFFSET ?",
            (*args, page_size, (page - 1) * page_size),
        )
        return dict(
            items=[to_image(row) for row in rows],
            total=total,
            page=page,
            pageSize=page_size,
        )

    def get_groups(self):
        import json

        self.require_ready()
        groups = []
        for row in self.db.execute("SELECT * FROM groups ORDER BY count DESC,id"):
            representatives = []
            for key in json.loads(row["representatives"]):
                image = self.db.execute(
                    IMAGE_SELECT + " AND i.key=?", (key,)
                ).fetchone()
                if image:
                    representatives.append(to_image(image))
            groups.append(
                dict(id=row["id"], count=row["count"], representatives=representatives)
            )
        return groups

    def image_file(self, image_id, kind):
        self.require_ready()
        image = self.get_image(image_id)
        if kind == "preview":
            from pathlib import Path

            row = self.db.execute(
                "SELECT a.cache FROM images i JOIN assets a ON a.id=i.asset_id WHERE i.id=?",
                (image_id,),
            ).fetchone()
            file = Path(row["cache"]).resolve()
            if not file.is_relative_to(self.config.storage):
                raise ValueError("Preview is outside application storage")
        else:
            file = (self.config.library / image["path"]).resolve()
            if not file.is_relative_to(self.config.library):
                raise ValueError("File is outside the image library")
        if not file.is_file():
            raise ValueError("File not found; rescan the library")
        return file

    def search(self, request):
        self.require_ready()
        query = (request.get("query") or "").strip()
        reference = None
        if image_id := request.get("referenceImageId"):
            reference = self.db.execute(
                "SELECT i.key,a.vector,a.cache,a.media_type,a.video_manifest FROM images i JOIN assets a ON a.id=i.asset_id WHERE i.status='ready' AND i.id=?",
                (image_id,),
            ).fetchone()
            if not reference:
                raise ValueError("Reference image not found")
        if query:
            if reference and reference["media_type"] == "video":
                vector = self.embedder.embed(
                    query=query, video=reference["video_manifest"]
                )
            else:
                vector = self.embedder.embed(
                    query=query, image=reference["cache"] if reference else None
                )
        elif reference:
            vector = vector_from(reference["vector"])
        else:
            raise ValueError("Enter a search or select a reference image")
        self.check_running()
        matches = self.vectors.search(
            vector, request.get("folderId"), reference["key"] if reference else None
        )
        return dict(
            items=[
                to_image(
                    self.db.execute(IMAGE_SELECT + " AND i.key=?", (key,)).fetchone()
                )
                | {"score": score}
                for key, score in matches
            ],
            total=len(matches),
        )
