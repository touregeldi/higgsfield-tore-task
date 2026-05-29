import pytest
pytestmark = pytest.mark.integration


def test_tables_exist_after_migrate(pool):
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        ).fetchall()
    names = {r[0] for r in rows}
    assert {"turns", "memories"} <= names
