"""Qdrant embedded quickstart (file-backed).

Creates a local Qdrant collection under ./qdrant_test_db, inserts a few points,
searches, and prints results. Useful for verifying the Qdrant Python client is
working without any running server.
"""
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


def main() -> None:
    print(f"python: {sys.version}")
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            qc_ver = version("qdrant-client")
        except PackageNotFoundError:
            qc_ver = "unknown"
    except Exception:
        qc_ver = "unknown"
    print(f"qdrant-client: {qc_ver}")

    client = QdrantClient(path="qdrant_test_db")
    collection = "quickstart"
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )

    points = [
        PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4], payload={"text": "alpha"}),
        PointStruct(id=2, vector=[0.9, 0.1, 0.2, 0.0], payload={"text": "bravo"}),
        PointStruct(id=3, vector=[0.0, 0.1, 0.9, 0.2], payload={"text": "charlie"}),
    ]
    client.upsert(collection_name=collection, points=points, wait=True)
    print("inserted", len(points), "points")

    query_vec = [0.05, 0.15, 0.85, 0.2]
    res = client.query_points(collection_name=collection, query=query_vec, limit=3, with_payload=True)
    print("search results:")
    for point in res.points:
        print(f"id={point.id} score={point.score:.4f} payload={point.payload}")

    print("done")


if __name__ == "__main__":
    main()
