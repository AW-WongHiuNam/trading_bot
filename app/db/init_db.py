import os
from app.db.session import Base, engine


def init_db() -> None:
    os.makedirs("./app/data", exist_ok=True)
    Base.metadata.create_all(bind=engine)
