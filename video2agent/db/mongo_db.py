import os
from typing import Optional

from loguru import logger
from pymongo import MongoClient
from pymongo.server_api import ServerApi


class MongoDB:
    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None):
        self.client = MongoClient(
            uri or os.getenv("MONGO_DB"), server_api=ServerApi("1")
        )
        self.db = self.client[db_name or os.getenv("MONGO_DB_NAME")]
        try:
            self.client.admin.command("ping")
            logger.info(
                "Pinged your deployment. You successfully connected to MongoDB!"
            )
        except Exception as e:
            raise e

    def get_collection(self, collection_name: str):
        return self.db[collection_name]

    def upsert(self, collection: str, filter: dict, data: dict):
        col = self.get_collection(collection)
        col.update_one(filter, {"$set": data}, upsert=True)

    def delete(self, collection: str, filter: dict):
        col = self.get_collection(collection)
        col.delete_many(filter)

    def query(self, collection: str, filter: dict) -> list[dict]:
        col = self.get_collection(collection)
        return list(col.find(filter))

    def query_one(self, collection: str, filter: dict) -> dict | None:
        col = self.get_collection(collection)
        return col.find_one(filter)


if __name__ == "__main__":
    mongo_db = MongoDB()
    # Clean all collections in the database
    mongo_db.delete(collection="videos", filter={})
    logger.info("MongoDB cleaned successfully")
