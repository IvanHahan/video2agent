import datetime
import os
from typing import Dict, List, Optional, Union

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
    model,
)

load_dotenv()


# Constants
DEFAULT_VECTOR_DIM = 1536
DEFAULT_VISUAL_DIM = 768  # ViT-L-14 default
DEFAULT_TOP_K = 5
DEFAULT_OUTPUT_FIELDS = ["image_id", "meta", "text"]
MAX_VARCHAR_LENGTH = 2048
MAX_ID_LENGTH = 512

EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Search mode constants
SEARCH_MODE_DENSE = "dense"
SEARCH_MODE_SPARSE = "sparse"
VALID_SEARCH_MODES = [SEARCH_MODE_DENSE, SEARCH_MODE_SPARSE]


class VectorDB:
    """A vector database wrapper for Milvus with user and session management."""

    def __init__(
        self,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        vector_dim: int = DEFAULT_VECTOR_DIM,
        visual_dim: int = DEFAULT_VISUAL_DIM,
        db_name: Optional[str] = None,
    ):
        self.uri = uri or os.environ.get("MILVUS_URL")
        self.token = token or os.environ.get("MILVUS_TOKEN")
        self.db_name = db_name or os.environ.get("MILVUS_DB")
        self.user_id = user_id
        self.session_id = session_id
        self.vector_dim = vector_dim
        self.visual_dim = visual_dim

        if not self.uri:
            raise ValueError(
                "Milvus URI must be provided via parameter or MILVUS_URL environment variable"
            )

        try:
            self._create_milvus_client()
            self.embed_model = model.dense.OpenAIEmbeddingFunction(
                model_name=EMBEDDING_MODEL,
            )
            self._setup_db()
        except Exception as e:
            logger.error("Failed to initialize VectorDB: %s", str(e))
            raise

    def _create_milvus_client(self):
        try:
            self.client = MilvusClient(
                uri=self.uri, token=self.token, db_name=self.db_name
            )
            logger.info(f"Connected to Milvus database: {self.db_name}")
        except Exception as db_error:
            # If database doesn't exist, create it
            try:
                # Connect without specifying database to create it
                temp_client = MilvusClient(uri=self.uri, token=self.token)
                temp_client.create_database(self.db_name)
                logger.info(f"Created database: {self.db_name}")
                self.client = MilvusClient(
                    uri=self.uri, token=self.token, db_name=self.db_name
                )
            except Exception as create_error:
                logger.warning(
                    f"Failed to create database {self.db_name}: {str(create_error)}"
                )
                # Fallback to connection without database name
                self.client = MilvusClient(uri=self.uri, token=self.token)
            else:
                # Re-raise original error if no database name specified
                raise db_error

    def search(
        self,
        collection: str,
        keywords: Optional[str] = None,  # sparse_query
        text: Optional[str | List[float]] = None,  # dense_query
        output_fields: Optional[List[str]] = None,
        top_k: int = DEFAULT_TOP_K,
        filter: Optional[str] = None,
        threshold: float = None,
    ) -> List[Dict]:
        """
        Dynamically search using multiple search modes.

        Args:
            collection: Collection name
            image_input: Image for visual search (file path or numpy array)
            keywords: Keywords query for sparse search (BM25)
            text: text query for dense search (OpenAI embeddings)
            output_fields: Fields to return
            top_k: Number of results
            filter: Filter expression
            threshold: Threshold for confidence scores of examples

        Returns:
            List of documents with scores
        """

        try:
            filter_expr = self._build_filter_expression(filter)

            # Always use hybrid search approach for consistency
            results = self._hybrid_search(
                collection,
                keywords,
                text,
                output_fields,
                top_k,
                filter_expr,
            )
            if threshold is not None:
                results = [r for r in results if r["distance"] <= threshold]
            return results
        except Exception as e:
            logger.error("Search failed for collection %s: %s", collection, str(e))
            return []

    def _hybrid_search(
        self,
        collection: str,
        keywords: Optional[str],  # sparse_query
        description: Optional[str],  # dense_query
        output_fields: Optional[List[str]],
        top_k: int,
        filter_expr: Optional[str],
    ) -> List[Dict]:
        """Perform hybrid search using dense/sparse/visual results using official Milvus hybrid search API."""
        try:
            search_requests = []
            # Build search requests for each mode
            if description:
                dense_embedding = (
                    self.embed_model.encode_documents([description])[0]
                    if isinstance(description, str)
                    else description
                )
                dense_search_params = {"metric_type": "IP"}
                search_requests.append(
                    AnnSearchRequest(
                        [dense_embedding],
                        "dense_embedding",
                        dense_search_params,
                        limit=top_k,
                        expr=filter_expr,
                    )
                )
            if keywords:
                # Sparse search using BM25
                sparse_search_params = {"metric_type": "BM25"}
                search_requests.append(
                    AnnSearchRequest(
                        [keywords],
                        "sparse_embedding",
                        sparse_search_params,
                        limit=top_k,
                        expr=filter_expr,
                    )
                )

            if not search_requests:
                logger.warning("No valid search requests generated")
                return []

            # Perform hybrid search
            results = self.client.hybrid_search(
                collection,
                search_requests,
                ranker=RRFRanker(),
                limit=top_k,
                output_fields=output_fields or DEFAULT_OUTPUT_FIELDS,
            )

            return results[0] if results else []

        except Exception as e:
            logger.error("Dynamic hybrid search failed: %s", str(e))
            return []

    def _build_filter_expression(
        self,
        custom_filter: Optional[str],
    ) -> Optional[str]:
        """Build a filter expression for search queries."""
        filter_parts = []
        if custom_filter is not None:
            filter_parts.append(custom_filter)
        return " AND ".join(filter_parts) if filter_parts else None

    def get(
        self,
        collection: str,
        ids: List[str],
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Retrieve documents by their IDs.

        Args:
            collection: Collection name
            ids: List of document IDs
            output_fields: Fields to return

        Returns:
            List of documents
        """
        try:
            results = self.client.get(
                collection_name=collection,
                ids=ids,
                output_fields=output_fields or DEFAULT_OUTPUT_FIELDS,
            )
            return results
        except Exception as e:
            logger.error("Get by IDs failed for collection %s: %s", collection, str(e))
            return []

    def upsert(
        self,
        collection: str,
        documents: List[Dict],
        image_inputs: List[Union[str, np.ndarray]],
        keywords: Optional[List[str]] = None,
        texts: Optional[List[str | float]] = None,
        insert_meta=True,
    ):
        """
        Insert or update documents with text and visual embeddings.
        Documents are identified by 'image_id'. If an existing document with the
        same image_id is found, it will be updated; otherwise a new one is created.

        Args:
            collection: Collection name
            documents: List of document dicts (must contain 'image_id' and 'meta')
            keywords: List of keyword strings for each document (for sparse embedding)
            descriptions: List of description strings or embeddings for each document (for dense embedding)
            image_inputs: List of images (file paths or numpy arrays)
            insert_meta: Whether to add timestamp metadata

        Returns:
            List of document IDs
        """
        try:
            if len(documents) != len(image_inputs):
                raise ValueError("Number of documents must match number of images")

            # Prepare documents with metadata
            if insert_meta:
                documents = self._prepare_documents(documents)

            # Add keywords and descriptions to documents
            for i, doc in enumerate(documents):
                # Note that if sample already existed in the database, results will be overwritten by the newly passed ones
                if keywords:
                    doc["keywords"] = keywords[i]
                if texts:
                    doc["text"] = texts[i]

            # Generate dense embeddings from texts
            if texts:
                dense_vectors = (
                    self.embed_model.encode_documents(texts)
                    if isinstance(texts[0], str)
                    else texts
                )
                for i, doc in enumerate(documents):
                    doc["dense_embedding"] = dense_vectors[i]

            response = self.client.upsert(collection_name=collection, data=documents)
            logger.info(
                f"Upserted {response['upsert_count']} documents to collection {collection}"
            )
            return response["primary_keys"]
        except Exception as e:
            logger.error(f"Upsert failed for collection {collection}: {str(e)}")
            raise

    def _prepare_documents(self, documents: List[Dict]) -> List[Dict]:
        """Prepare documents by adding metadata and generating IDs."""
        prepared_docs = []
        for doc in documents:
            if "timestamp" not in doc:
                doc["timestamp"] = datetime.datetime.now(
                    datetime.timezone.utc
                ).timestamp()
            prepared_docs.append(doc)
        return prepared_docs

    def delete(self, collection: str, filter_expr: str) -> int:
        """
        Delete documents from a collection based on filter expression.

        Args:
            collection: Name of the collection
            filter_expr: Filter expression to identify documents to delete

        Returns:
            Number of deleted documents
        """
        try:
            result = self.client.delete(collection_name=collection, filter=filter_expr)
            logger.info(f"Deleted documents from collection {collection}")
            return result
        except Exception as e:
            logger.error(f"Delete failed for collection {collection}: {str(e)}")
            raise

    def get_collection_info(self, collection: str) -> Dict:
        """Get information about a collection."""
        try:
            if not self.client.has_collection(collection):
                return {"exists": False}

            stats = self.client.get_collection_stats(collection)
            return {
                "exists": True,
                "stats": stats,
            }
        except Exception as e:
            logger.error("Failed to get collection info for %s: %s", collection, str(e))
            return {"exists": False, "error": str(e)}

    def _is_collection_empty(self, collection: str) -> bool:
        """Check if a collection is empty (has no documents)."""
        try:
            if not self.client.has_collection(collection):
                return True

            stats = self.client.get_collection_stats(collection)
            # Check if row_count exists and is 0
            row_count = stats.get("row_count", 0)
            return row_count == 0
        except Exception as e:
            logger.warning(
                "Failed to check if collection %s is empty: %s", collection, str(e)
            )
            return True  # Assume empty if we can't check

    def _setup_db(self) -> None:
        """Set up the necessary collections in the vector database."""
        try:
            self._setup_bullets_collection()
            self._setup_transcript_collection()
            self._setup_videos_collection()
            logger.info("Vector database setup completed successfully")

        except Exception as e:
            logger.error("Database setup failed: %s", str(e))
            raise

    def _create_collection_schema(self, auto_id: bool = False) -> object:
        """Create a base schema for collections."""
        return MilvusClient.create_schema(
            auto_id=auto_id,
            enable_dynamic_field=True,
        )

    def _setup_videos_collection(self) -> None:
        """Set up the videos collection."""
        collection_name = "videos"
        if self.client.has_collection(collection_name):
            return
            # self.client.drop_collection(collection_name)

        schema = self._create_collection_schema(auto_id=False)  #

        # ID field Removed completely due to compatability issues, made image_id a primary one
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            max_length=MAX_ID_LENGTH,
            is_primary=True,
        )
        schema.add_field(
            field_name="title",
            datatype=DataType.VARCHAR,
            max_length=MAX_VARCHAR_LENGTH,
        )
        schema.add_field(
            field_name="description",
            datatype=DataType.VARCHAR,
            max_length=MAX_VARCHAR_LENGTH,
        )
        schema.add_field(
            field_name="sparse_embedding",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_field(
            field_name="dense_embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.vector_dim,
        )

        index_params = self.client.prepare_index_params()
        # index_params.add_index(field_name="id", index_type="AUTOINDEX")
        index_params.add_index(field_name="id", index_type="AUTOINDEX")
        index_params.add_index(field_name="video_id", index_type="AUTOINDEX")
        index_params.add_index(
            field_name="sparse_embedding",
            index_type="AUTOINDEX",
            metric_type="BM25",
        )
        index_params.add_index(
            field_name="dense_embedding",
            index_type="AUTOINDEX",
            metric_type="IP",
        )

        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["keywords"],
            output_field_names=["sparse_embedding"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        self.client.load_collection(collection_name)

    def _setup_transcript_collection(self) -> None:
        """Set up the transcripts collection."""
        collection_name = "transcripts"
        if self.client.has_collection(collection_name):
            # return
            self.client.drop_collection(collection_name)

        schema = self._create_collection_schema(auto_id=False)  #

        # ID field Removed completely due to compatability issues, made image_id a primary one
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            max_length=MAX_ID_LENGTH,
            is_primary=True,
        )
        schema.add_field(
            field_name="video_id",
            datatype=DataType.VARCHAR,
            max_length=MAX_ID_LENGTH,
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=MAX_VARCHAR_LENGTH,
        )
        schema.add_field(
            field_name="start",
            datatype=DataType.FLOAT,
        )
        schema.add_field(
            field_name="duration",
            datatype=DataType.FLOAT,
        )
        schema.add_field(
            field_name="sparse_embedding",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_field(
            field_name="dense_embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.vector_dim,
        )

        index_params = self.client.prepare_index_params()
        # index_params.add_index(field_name="id", index_type="AUTOINDEX")
        index_params.add_index(field_name="id", index_type="AUTOINDEX")
        index_params.add_index(field_name="video_id", index_type="AUTOINDEX")
        index_params.add_index(
            field_name="sparse_embedding",
            index_type="AUTOINDEX",
            metric_type="BM25",
        )
        index_params.add_index(
            field_name="dense_embedding",
            index_type="AUTOINDEX",
            metric_type="IP",
        )

        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["keywords"],
            output_field_names=["sparse_embedding"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        self.client.load_collection(collection_name)

    def _setup_bullets_collection(self) -> None:
        """Set up the bullets collection."""
        collection_name = "bullets"
        if self.client.has_collection(collection_name):
            return
            # self.client.drop_collection(collection_name)

        schema = self._create_collection_schema(auto_id=False)  #

        # ID field Removed completely due to compatability issues, made image_id a primary one
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            max_length=MAX_ID_LENGTH,
            is_primary=True,
        )
        schema.add_field(
            field_name="keywords",
            datatype=DataType.VARCHAR,
            max_length=MAX_VARCHAR_LENGTH,
            enable_analyzer=True,
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=MAX_VARCHAR_LENGTH,
        )
        schema.add_field(
            field_name="timestamp",
            datatype=DataType.FLOAT,
        )
        schema.add_field(
            field_name="sparse_embedding",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_field(
            field_name="dense_embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.vector_dim,
        )

        index_params = self.client.prepare_index_params()
        # index_params.add_index(field_name="id", index_type="AUTOINDEX")
        index_params.add_index(field_name="id", index_type="AUTOINDEX")
        index_params.add_index(field_name="video_id", index_type="AUTOINDEX")
        index_params.add_index(
            field_name="sparse_embedding",
            index_type="AUTOINDEX",
            metric_type="BM25",
        )
        index_params.add_index(
            field_name="dense_embedding",
            index_type="AUTOINDEX",
            metric_type="IP",
        )

        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["keywords"],
            output_field_names=["sparse_embedding"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        self.client.load_collection(collection_name)


if __name__ == "__main__":
    # Example usage
    from youtube import (
        get_video_transcript,
        get_youtube_video_info,
        merge_transcript_snippets,
    )

    db = VectorDB()
    video_id = "48ZK2JcoHyU"
    transcript = get_video_transcript(video_id, ["uk"], preserve_formatting=True)
    transcript = merge_transcript_snippets(transcript, max_tokens=500)
    info = get_youtube_video_info(video_id)

    db.upsert(
        collection="transcripts",
        documents=[
            {
                "id": f"{video_id}_{i}",
                "video_id": video_id,
                "text": t.text,
                "keywords": t.text,
                "start": t.start,
                "duration": t.duration,
            }
            for i, t in enumerate(transcript)
        ],
        texts=[t.text for t in transcript],
    )

    db.upsert(
        collection="videos",
        documents=[
            {
                "id": video_id,
                "title": info.title,
                "description": info.description,
            }
        ],
    )
