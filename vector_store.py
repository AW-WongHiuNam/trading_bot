"""DELETED: this module was removed in favor of `vector_store_sqlite`.

The project now uses `vector_store_sqlite.VectorStore`. This file used to be a
compatibility shim; it has been intentionally disabled to avoid accidental
dependence on a legacy import path. Import `VectorStore` from
`vector_store_sqlite` instead.
"""

raise ImportError(
    "vector_store has been removed — import VectorStore from vector_store_sqlite instead"
)
