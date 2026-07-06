from .base import BarStore
from .parquet_store import ParquetStore

__all__ = ["BarStore", "ParquetStore", "PostgresStore"]


def __getattr__(name):
    # Lazy import so Parquet-only usage never needs SQLAlchemy/psycopg loaded.
    if name == "PostgresStore":
        from .postgres_store import PostgresStore

        return PostgresStore
    raise AttributeError(name)
