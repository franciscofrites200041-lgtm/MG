-- Index de documentos del estudio. Un solo archivo SQLite, FTS5 embebido.
-- Tokenizer unicode61 con remove_diacritics=2 para que "juridico" matchee "jurídico".

CREATE TABLE IF NOT EXISTS files (
    path        TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    ext         TEXT NOT NULL,
    mtime       REAL NOT NULL,
    size        INTEGER NOT NULL,
    n_chunks    INTEGER NOT NULL,
    indexed_at  REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'ok'
);

CREATE INDEX IF NOT EXISTS idx_files_mtime ON files(mtime);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);

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
