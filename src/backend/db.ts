import { Database } from 'bun:sqlite';
import { join } from 'node:path';
import { mkdirSync } from 'node:fs';
import type { Config } from './config';
import type { ImageItem, Folder } from '../shared/types';

export function openDatabase(config: Config) {
  mkdirSync(config.storage.path, { recursive: true });
  const db = new Database(join(config.storage.path, 'library.sqlite'), {
    create: true,
  });
  db.exec(`PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;
 CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY,value TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS folders (id TEXT PRIMARY KEY,path TEXT UNIQUE NOT NULL,parent_id TEXT,name TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS assets (id TEXT PRIMARY KEY,hash TEXT NOT NULL,fingerprint TEXT NOT NULL,cache TEXT NOT NULL,width INTEGER NOT NULL,height INTEGER NOT NULL,vector BLOB NOT NULL);
 CREATE TABLE IF NOT EXISTS images (key INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE NOT NULL,path TEXT UNIQUE NOT NULL,folder_id TEXT NOT NULL,size INTEGER NOT NULL,mtime TEXT NOT NULL,hash TEXT,status TEXT NOT NULL,error TEXT,asset_id TEXT,group_id INTEGER);
 CREATE INDEX IF NOT EXISTS image_folder ON images(folder_id);
 CREATE INDEX IF NOT EXISTS image_asset ON images(asset_id);
 CREATE INDEX IF NOT EXISTS image_group ON images(group_id);
 CREATE TABLE IF NOT EXISTS groups (id INTEGER PRIMARY KEY,count INTEGER NOT NULL,representatives TEXT NOT NULL);
 `);
  const previous = db
    .query('SELECT value FROM settings WHERE key=?')
    .get('root') as { value: string } | null;
  if (previous && previous.value !== config.library.path) {
    db.exec('DELETE FROM images; DELETE FROM folders; DELETE FROM groups;');
  }
  db.query('INSERT OR REPLACE INTO settings VALUES (?,?)').run(
    'root',
    config.library.path,
  );
  db.query('INSERT OR IGNORE INTO folders VALUES (?,?,?,?)').run(
    'root',
    '',
    null,
    'Library',
  );
  return db;
}

export const imageSelect = `SELECT i.id,i.path,i.folder_id AS folderId,a.width,a.height FROM images i JOIN assets a ON a.id=i.asset_id WHERE i.status='ready'`;

export function toImage(value: unknown): ImageItem {
  const row = value as Omit<ImageItem, 'name'>;
  return { ...row, name: row.path.split('/').at(-1)! };
}

export function getImage(db: Database, id: string): ImageItem | null {
  const r = db.query(`${imageSelect} AND i.id=?`).get(id);
  return r ? toImage(r) : null;
}

export function scopePath(db: Database, id?: string): string {
  if (!id || id === 'root') {
    return '';
  }
  const f = db.query('SELECT path FROM folders WHERE id=?').get(id) as {
    path: string;
  } | null;
  if (!f) {
    throw new Error('Folder not found');
  }
  return f.path;
}

export function descendantClause(path: string) {
  return {
    sql: path ? ' AND substr(i.path,1,length(?))=?' : '',
    args: path ? [path + '/', path + '/'] : [],
  };
}

export function folders(db: Database): Folder[] {
  return db
    .query(
      `SELECT f.id,f.path,f.name,f.parent_id AS parentId,COUNT(i.key) AS count FROM folders f LEFT JOIN images i ON i.folder_id=f.id AND i.status='ready' GROUP BY f.id ORDER BY f.path COLLATE NOCASE`,
    )
    .all() as Folder[];
}

export function vectorFrom(blob: Uint8Array): Float32Array {
  const copy = Uint8Array.from(blob);
  return new Float32Array(copy.buffer);
}

export function vectorBlob(vector: Float32Array) {
  return new Uint8Array(vector.buffer, vector.byteOffset, vector.byteLength);
}
