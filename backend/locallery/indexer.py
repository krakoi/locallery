"""Ordinary incremental scans; source files are never modified."""

import hashlib
import os
import time
from pathlib import PurePosixPath

from .images import SUPPORTED, make_preview
from .videos import SUPPORTED_VIDEOS, cache_exists, prepare_video, video_fingerprint


def opaque(value):
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def initial_progress():
    return dict(
        busy=True,
        stage="starting",
        discovered=0,
        processed=0,
        total=0,
        unchanged=0,
        indexed=0,
        skipped=0,
        failed=0,
        rate=0,
        etaSeconds=None,
        message="Opening library",
        errors=[],
        startedAt=int(time.time() * 1000),
    )


def scan(config, db, embedder, report):
    progress = initial_progress()
    progress.update(stage="discovering", message="Discovering images and videos")
    complete = True
    files, seen, seen_folders = [], set(), {"root"}
    fingerprint = embedder.fingerprint
    video_fp = video_fingerprint(fingerprint, config.video)
    db.execute(
        "UPDATE images SET status='pending' WHERE asset_id IN (SELECT id FROM assets WHERE fingerprint<>CASE WHEN media_type='video' THEN ? ELSE ? END)",
        (video_fp, fingerprint),
    )

    def emit():
        report(progress | {"errors": list(progress["errors"])})

    def error(path, exception):
        progress["failed"] += 1
        if len(progress["errors"]) < 30:
            progress["errors"].append({"path": str(path), "message": str(exception)})

    def walk(relative="", parent=None):
        nonlocal complete
        folder_id = opaque("folder:" + relative) if relative else "root"
        seen_folders.add(folder_id)
        db.execute(
            "INSERT OR REPLACE INTO folders VALUES (?,?,?,?)",
            (
                folder_id,
                relative,
                parent,
                PurePosixPath(relative).name if relative else "Library",
            ),
        )
        try:
            with os.scandir(config.library / relative) as entries:
                for entry in entries:
                    path = f"{relative}/{entry.name}" if relative else entry.name
                    if entry.name == ".locallery" or entry.is_symlink():
                        progress["skipped"] += 1
                    elif entry.is_dir(follow_symlinks=False):
                        walk(path, folder_id)
                    elif entry.is_file(follow_symlinks=False):
                        if (
                            PurePosixPath(entry.name).suffix.lower()
                            in SUPPORTED | SUPPORTED_VIDEOS
                        ):
                            files.append(path)
                            seen.add(path)
                            progress["discovered"] += 1
                        else:
                            progress["skipped"] += 1
                    if (progress["discovered"] + progress["skipped"]) % 100 == 0:
                        emit()
        except OSError as exception:
            complete = False
            error(relative or config.library, exception)

    emit()
    walk()
    progress.update(
        stage="processing",
        total=len(files),
        message="Checking files and generating media embeddings",
    )
    emit()
    started = time.monotonic()
    for path in files:
        size, mtime, digest = 0, "", None
        parent = str(PurePosixPath(path).parent)
        folder_id = "root" if parent == "." else opaque("folder:" + parent)
        try:
            source = config.library / path
            is_video = source.suffix.lower() in SUPPORTED_VIDEOS
            current_fp = video_fp if is_video else fingerprint
            stat = source.lstat()
            if source.is_symlink():
                raise ValueError("Source changed to a symlink; rescan")
            size, mtime = stat.st_size, str(stat.st_mtime_ns)
            old = db.execute(
                "SELECT i.*,a.fingerprint,a.cache,a.video_manifest FROM images i LEFT JOIN assets a ON a.id=i.asset_id WHERE i.path=?",
                (path,),
            ).fetchone()
            if (
                old
                and old["status"] == "ready"
                and old["size"] == size
                and old["mtime"] == mtime
                and old["fingerprint"] == current_fp
                and os.path.isfile(old["cache"])
                and (not is_video or cache_exists(old["video_manifest"]))
            ):
                progress["unchanged"] += 1
            else:
                with source.open("rb") as file:
                    digest = hashlib.file_digest(file, "sha256").hexdigest()
                asset_id = digest + "-" + current_fp
                cached = db.execute(
                    "SELECT cache,video_manifest FROM assets WHERE id=?", (asset_id,)
                ).fetchone()
                if (
                    cached
                    and os.path.isfile(cached["cache"])
                    and (not is_video or cache_exists(cached["video_manifest"]))
                ):
                    progress["unchanged"] += 1
                else:
                    duration, manifest = None, None
                    if is_video:

                        def report_frame(message):
                            progress["message"] = message
                            emit()

                        cache, width, height, duration, manifest = prepare_video(
                            source, asset_id, config.storage, config.video, report_frame
                        )
                        progress["message"] = f"Embedding sampled video · {source.name}"
                        emit()
                        vector = embedder.embed(video=manifest)
                    else:
                        cache, width, height = make_preview(
                            source, asset_id, config.storage
                        )
                        vector = embedder.embed(image=cache)
                    db.execute(
                        "INSERT OR REPLACE INTO assets(id,hash,fingerprint,cache,width,height,vector,media_type,duration,video_manifest) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (
                            asset_id,
                            digest,
                            current_fp,
                            cache,
                            width,
                            height,
                            vector.astype("<f4").tobytes(),
                            "video" if is_video else "image",
                            duration,
                            manifest,
                        ),
                    )
                    progress["indexed"] += 1
                db.execute(
                    """INSERT INTO images(id,path,folder_id,size,mtime,hash,status,error,asset_id) VALUES (?,?,?,?,?,?,'ready',NULL,?) ON CONFLICT(path) DO UPDATE SET folder_id=excluded.folder_id,size=excluded.size,mtime=excluded.mtime,hash=excluded.hash,status=excluded.status,error=NULL,asset_id=excluded.asset_id""",
                    (
                        opaque("image:" + path),
                        path,
                        folder_id,
                        size,
                        mtime,
                        digest,
                        asset_id,
                    ),
                )
        except Exception as exception:
            error(path, exception)
            db.execute(
                """INSERT INTO images(id,path,folder_id,size,mtime,hash,status,error,asset_id) VALUES (?,?,?,?,?,?,'error',?,NULL) ON CONFLICT(path) DO UPDATE SET size=excluded.size,mtime=excluded.mtime,hash=excluded.hash,status='error',error=excluded.error,asset_id=NULL""",
                (
                    opaque("image:" + path),
                    path,
                    folder_id,
                    size,
                    mtime,
                    digest,
                    str(exception),
                ),
            )
        progress["processed"] += 1
        elapsed = time.monotonic() - started
        progress["rate"] = progress["processed"] / elapsed if elapsed else 0
        progress["etaSeconds"] = (
            (progress["total"] - progress["processed"]) / progress["rate"]
            if progress["rate"]
            else None
        )
        emit()
    if complete:
        db.execute("BEGIN")
        try:
            db.executemany(
                "DELETE FROM images WHERE path=?",
                [
                    (row["path"],)
                    for row in db.execute("SELECT path FROM images").fetchall()
                    if row["path"] not in seen
                ],
            )
            db.executemany(
                "DELETE FROM folders WHERE id=?",
                [
                    (row["id"],)
                    for row in db.execute("SELECT id FROM folders").fetchall()
                    if row["id"] not in seen_folders
                ],
            )
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise
    progress["message"] = (
        "Scan finished" if complete else "Scan incomplete; previous records preserved"
    )
    emit()
    return complete
