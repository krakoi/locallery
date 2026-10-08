"""SQLite schema compatible with the original gallery."""

import sqlite3

import numpy as np

from .config import Config

IMAGE_SELECT = "SELECT i.id,i.path,i.folder_id AS folderId,a.width,a.height,a.media_type AS mediaType,a.duration FROM images i JOIN assets a ON a.id=i.asset_id WHERE i.status='ready'"


def open_database(config: Config):
    config.storage.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(
        config.storage / "library.sqlite", timeout=5, isolation_level=None
    )
    db.row_factory = sqlite3.Row
    db.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS folders (id TEXT PRIMARY KEY,path TEXT UNIQUE NOT NULL,parent_id TEXT,name TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS assets (id TEXT PRIMARY KEY,hash TEXT NOT NULL,fingerprint TEXT NOT NULL,cache TEXT NOT NULL,width INTEGER NOT NULL,height INTEGER NOT NULL,vector BLOB NOT NULL);
    CREATE TABLE IF NOT EXISTS images (key INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE NOT NULL,path TEXT UNIQUE NOT NULL,folder_id TEXT NOT NULL,size INTEGER NOT NULL,mtime TEXT NOT NULL,hash TEXT,status TEXT NOT NULL,error TEXT,asset_id TEXT,group_id INTEGER);
    CREATE INDEX IF NOT EXISTS image_folder ON images(folder_id);
    CREATE INDEX IF NOT EXISTS image_asset ON images(asset_id);
    CREATE INDEX IF NOT EXISTS image_group ON images(group_id);
    CREATE TABLE IF NOT EXISTS groups (id INTEGER PRIMARY KEY,count INTEGER NOT NULL,representatives TEXT NOT NULL);
    """)
    columns = {row["name"] for row in db.execute("PRAGMA table_info(assets)")}
    for name, declaration in (
        ("media_type", "TEXT NOT NULL DEFAULT 'image'"),
        ("duration", "REAL"),
        ("video_manifest", "TEXT"),
    ):
        if name not in columns:
            db.execute(f"ALTER TABLE assets ADD COLUMN {name} {declaration}")
    old = db.execute("SELECT value FROM settings WHERE key='root'").fetchone()
    if old and old["value"] != str(config.library):
        db.executescript("DELETE FROM images; DELETE FROM folders; DELETE FROM groups;")
    db.execute(
        "INSERT OR REPLACE INTO settings VALUES ('root',?)", (str(config.library),)
    )
    db.execute("INSERT OR IGNORE INTO folders VALUES ('root','',NULL,'Library')")
    return db


def to_image(row):
    result = dict(row)
    result["name"] = result["path"].rsplit("/", 1)[-1]
    return result


def folders(db):
    return [
        dict(row)
        for row in db.execute(
            "SELECT f.id,f.path,f.name,f.parent_id AS parentId,COUNT(i.key) AS count FROM folders f LEFT JOIN images i ON i.folder_id=f.id AND i.status='ready' GROUP BY f.id ORDER BY f.path COLLATE NOCASE"
        )
    ]


def scope_path(db, folder_id=None):
    if not folder_id or folder_id == "root":
        return ""
    row = db.execute("SELECT path FROM folders WHERE id=?", (folder_id,)).fetchone()
    if not row:
        raise ValueError("Folder not found")
    return row["path"]


def scope_clause(path):
    return (
        (" AND substr(i.path,1,length(?))=?", (path + "/", path + "/"))
        if path
        else ("", ())
    )


def vector_from(blob):
    return np.frombuffer(blob, dtype="<f4")


def vector_batches(db, path="", size=1024):
    clause, args = scope_clause(path)
    last = 0
    while rows := db.execute(
        f"SELECT i.key,a.vector FROM images i JOIN assets a ON a.id=i.asset_id WHERE i.status='ready' AND i.key>?{clause} ORDER BY i.key LIMIT ?",
        (last, *args, size),
    ).fetchall():
        yield rows
        last = rows[-1]["key"]
