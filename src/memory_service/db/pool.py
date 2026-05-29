from __future__ import annotations
import pathlib
import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

_SCHEMA = pathlib.Path(__file__).with_name("schema.sql")


def _ensure_extension(database_url: str) -> None:
    with psycopg.connect(database_url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()


def _configure(conn) -> None:
    register_vector(conn)


def create_pool(database_url: str) -> ConnectionPool:
    _ensure_extension(database_url)  # extension must exist before register_vector runs
    pool = ConnectionPool(conninfo=database_url, min_size=1, max_size=10,
                          configure=_configure, open=True)
    return pool


def migrate(pool: ConnectionPool) -> None:
    ddl = _SCHEMA.read_text()
    with pool.connection() as conn:
        conn.execute(ddl)
        conn.commit()
