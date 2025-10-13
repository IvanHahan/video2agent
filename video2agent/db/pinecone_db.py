import itertools
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


# Constants
DEFAULT_VECTOR_DIM = 1536
DEFAULT_TOP_K = 5
DEFAULT_OUTPUT_FIELDS = ["image_id", "meta", "text"]

EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")


def chunks(iterable, batch_size=200):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


class VectorDB:
    """A vector database wrapper for Pinecone with user and session management."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        vector_dim: int = DEFAULT_VECTOR_DIM,
        namespace: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self.user_id = user_id
        self.session_id = session_id
        self.vector_dim = vector_dim
        self.namespace = namespace or "default"

        if not self.api_key:
            raise ValueError(
                "Pinecone API key must be provided via parameter or PINECONE_API_KEY environment variable"
            )

        try:
            self.pc = Pinecone(api_key=self.api_key)
            self.openai_client = OpenAI()
            self._setup_db()
            logger.info("Connected to Pinecone")
        except Exception as e:
            logger.error(f"Failed to initialize VectorDB: {str(e)}")
            raise

    def _get_embeddings(self, texts: List[str]):
        """Generate embedding using OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                input=texts, model=EMBEDDING_MODEL
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise

    def search(
        self,
        collection: str,
        text: Optional[str | List[float]] = None,
        top_k: int = DEFAULT_TOP_K,
        filter: Optional[Dict] = None,
        threshold: float = None,
    ) -> List[Dict]:
        """
        Search using text embeddings.

        Args:
            collection: Index name (collection name)
            keywords: Keywords query (will be combined with text for embedding)
            text: Text query or pre-computed embedding for dense search
            output_fields: Fields to return (not used in Pinecone, all metadata returned)
            top_k: Number of results
            filter: Filter dictionary for metadata filtering
            threshold: Threshold for confidence scores

        Returns:
            List of documents with scores
        """
        try:
            index = self.pc.Index(collection)

            # Combine keywords and text for search query
            query_text = text

            if not query_text:
                logger.warning("No search query provided")
                return []

            # Generate embedding if text provided
            if isinstance(query_text, str):
                query_vector = self._get_embeddings([query_text])[0]
            else:
                query_vector = query_text

            # Perform search
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter,
                include_metadata=True,
                namespace=self.namespace,
            )

            # Convert Pinecone results to standard format
            documents = []
            for match in results.matches:
                doc = {
                    "id": match.id,
                    "distance": match.score,  # Pinecone returns similarity score
                    **match.metadata,
                }

                # Apply threshold filter if specified
                if threshold is None or match.score >= threshold:
                    documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Search failed for index {collection}: {str(e)}")
            return []

    def get(
        self,
        collection: str,
        ids: List[str],
    ) -> List[Dict]:
        """
        Retrieve documents by their IDs.

        Args:
            collection: Index name
            ids: List of document IDs
            output_fields: Fields to return (not used in Pinecone)

        Returns:
            List of documents
        """
        try:
            index = self.pc.Index(collection)
            results = index.fetch(ids=ids, namespace=self.namespace)

            documents = []
            for doc_id, vector_data in results.vectors.items():
                doc = {"_id": doc_id, **vector_data.metadata}
                documents.append(doc)

            return documents
        except Exception as e:
            logger.error(f"Get by IDs failed for index {collection}: {str(e)}")
            return []

    def upsert(
        self,
        collection: str,
        documents: List[Dict],
        texts: Optional[List[str | List[float]]] = None,
    ):
        """
        Insert or update documents with embeddings.

        Args:
            collection: Index name
            documents: List of document dicts (must contain 'id')
            keywords: List of keyword strings for each document
            texts: List of text strings or embeddings for each document
            insert_meta: Whether to add timestamp metadata (not implemented)

        Returns:
            List of document IDs
        """
        try:
            index = self.pc.Index(collection)

            # Prepare vectors for upsert
            vectors = self._get_embeddings(texts)
            records = []
            for i, doc in enumerate(documents):
                doc_id = doc.get("id")
                if not doc_id:
                    raise ValueError(f"Document at index {i} missing 'id' field")

                # Prepare metadata (exclude '_id' as it's the vector ID)
                metadata = {k: v for k, v in doc.items() if k != "_id"}

                records.append(
                    {"id": doc_id, "values": vectors[i], "metadata": metadata}
                )

            # Batch upsert
            for ids_vectors_chunk in chunks(records, batch_size=200):
                index.upsert(vectors=ids_vectors_chunk, namespace=self.namespace)
            logger.info(f"Upserted {len(records)} documents to index {collection}")

            return [r["id"] for r in records]

        except Exception as e:
            logger.error(f"Upsert failed for index {collection}: {str(e)}")
            raise

    def delete(
        self, collection: str, filter_expr: Dict = None, ids: List[str] = None
    ) -> int:
        """
        Delete documents from an index.

        Args:
            collection: Index name
            filter_expr: Metadata filter dictionary
            ids: List of IDs to delete

        Returns:
            Number of deleted documents (Pinecone doesn't return this)
        """
        try:
            index = self.pc.Index(collection)

            if ids:
                index.delete(ids=ids, namespace=self.namespace)
                logger.info(f"Deleted {len(ids)} documents from index {collection}")
                return len(ids)
            elif filter_expr:
                index.delete(filter=filter_expr, namespace=self.namespace)
                logger.info(
                    f"Deleted documents matching filter from index {collection}"
                )
                return -1  # Pinecone doesn't return count for filter-based deletes
            else:
                logger.warning("No IDs or filter provided for delete operation")
                return 0

        except Exception as e:
            logger.error(f"Delete failed for index {collection}: {str(e)}")
            raise

    def get_collection_info(self, collection: str) -> Dict:
        """Get information about an index."""
        try:
            index = self.pc.Index(collection)
            stats = index.describe_index_stats()

            return {
                "exists": True,
                "stats": {
                    "total_vector_count": stats.total_vector_count,
                    "dimension": stats.dimension,
                    "namespaces": stats.namespaces,
                },
            }
        except Exception as e:
            logger.error(f"Failed to get index info for {collection}: {str(e)}")
            return {"exists": False, "error": str(e)}

    def _is_collection_empty(self, collection: str) -> bool:
        """Check if an index is empty (has no documents)."""
        try:
            info = self.get_collection_info(collection)
            if not info.get("exists"):
                return True

            total_count = info.get("stats", {}).get("total_vector_count", 0)
            return total_count == 0
        except Exception as e:
            logger.warning(f"Failed to check if index {collection} is empty: {str(e)}")
            return True

    def _setup_db(self) -> None:
        """Set up the necessary indexes in Pinecone."""
        try:
            self._setup_index("transcripts")
            logger.info("Pinecone setup completed successfully")
        except Exception as e:
            logger.error(f"Database setup failed: {str(e)}")
            raise

    def _setup_index(self, index_name: str) -> None:
        """Create a Pinecone index if it doesn't exist."""
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]

            if index_name not in existing_indexes:
                logger.info(f"Creating index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=self.vector_dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
                )
                logger.info(f"Created index: {index_name}")
            else:
                logger.info(f"Index {index_name} already exists")

        except Exception as e:
            logger.error(f"Failed to setup index {index_name}: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    from video2agent.youtube import (
        get_video_transcript,
        get_youtube_video_info,
        merge_transcript_snippets,
    )

    db = VectorDB()
    video_id = "48ZK2JcoHyU"
    transcript = get_video_transcript(video_id, ["uk"], preserve_formatting=True)
    transcript = merge_transcript_snippets(transcript, max_tokens=500)
    info = get_youtube_video_info(video_id)

    # Upsert transcripts
    db.upsert(
        collection="transcripts",
        documents=[
            {
                "id": f"{video_id}_{i}",
                "video_id": video_id,
                "text": t.text,
                "start": t.start,
                "duration": t.duration,
            }
            for i, t in enumerate(transcript)
        ],
        texts=[t.text for t in transcript],
    )
    print("Upsert completed")
