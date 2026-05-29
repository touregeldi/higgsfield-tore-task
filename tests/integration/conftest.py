import os
import uuid
import pytest
from memory_service.db.pool import create_pool, migrate

DB_URL = os.getenv("DATABASE_URL", "postgresql://memory:memory@localhost:5432/memory")


@pytest.fixture(scope="session")
def pool():
    try:
        p = create_pool(DB_URL)
        migrate(p)
    except Exception as exc:
        pytest.skip(f"Postgres not available: {exc}")
    yield p
    p.close()


@pytest.fixture(autouse=True)
def clean(pool):
    with pool.connection() as conn:
        conn.execute("DELETE FROM memories")
        conn.execute("DELETE FROM turns")
        conn.commit()
    yield


@pytest.fixture
def ids():
    return lambda: uuid.uuid4().hex
