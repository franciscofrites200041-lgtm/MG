-- Index de documentos del estudio. Un solo archivo SQLite, FTS5 embebido.
-- Tokenizer unicode61 con remove_diacritics=2 para que "juridico" matchee "jurídico".

CREATE TABLE IF NOT EXISTS files (
    path            TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    ext             TEXT NOT NULL,
    mtime           REAL NOT NULL,
    size            INTEGER NOT NULL,
    n_chunks        INTEGER NOT NULL,
    indexed_at      REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'ok',
    -- ponytail: pipeline lazy. preview = primeros ~2000 chars extraidos barato al
    -- indexar. fully_extracted=0 = solo preview en chunks (page=0). 1 = chunks
    -- reales por pagina. Se promueve on-demand al primer hit de query, o por el
    -- worker de background en la PC.
    preview         TEXT NOT NULL DEFAULT '',
    fully_extracted INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_files_mtime ON files(mtime);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
-- idx_files_fully_extracted lo crea la migracion en Python (para no romper DBs
-- viejas donde la columna todavia no existe cuando corre este script).

CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
    path       UNINDEXED,
    filename,
    page       UNINDEXED,
    text,
    tokenize = 'unicode61 remove_diacritics 2'
);

-- Re-ranker: un vector por chunk, mapeado 1:1 por rowid contra chunks.
-- Se puebla desde la PC (GPU) al indexar. En runtime el NAS solo embed la
-- query y hace cosine contra estos blobs (numpy in-memory).
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_rowid INTEGER PRIMARY KEY,
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vec         BLOB NOT NULL
);
