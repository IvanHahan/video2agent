from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .llm import OpenAIModel
from .milvus_db import VectorDB
from .prompts import SYSTEM_MESSAGE
from .youtube import (
    get_video_transcript,
    get_youtube_video_info,
    merge_transcript_snippets,
)


class YoutubeVideoAgent:
    def __init__(
        self,
        video_id: str,
        llm: ChatOpenAI,
        db: VectorDB,
        languages: list[str] = None,
        max_history_messages: int = 5,
    ):
        self.video_id = video_id
        self.llm = llm
        self.db = db
        self.max_history_messages = max_history_messages
        self.chat_history = []  # List of tuples: (role, content)

        # Process the video and create vector index during initialization
        if languages is None:
            languages = ["en"]
        self._process_youtube_video(languages)
        self.video_info = self.db.get(
            collection="videos",
            ids=[self.video_id],
            output_fields=["title", "text"],
        )

    def __del__(self):
        """Destructor to clean up video data when the agent instance is destroyed."""
        try:
            # Delete all transcript snippets for this video
            self.db.delete(
                collection="transcripts", filter_expr=f"video_id == '{self.video_id}'"
            )

            # Delete the video metadata
            self.db.delete(
                collection="videos",
                ids=[self.video_id],
            )
        except Exception as e:
            # Silently handle errors during cleanup to avoid issues during shutdown
            pass

    def _process_youtube_video(self, languages: list[str]):
        """Process the video and store its transcript in the vector database."""
        # Check if video already exists in the database
        existing_video = self.db.get(
            collection="videos",
            ids=[self.video_id],
            output_fields=["id"],
        )
        if existing_video:
            return
        # Fetch video info and transcript
        video_info = get_youtube_video_info(self.video_id)
        transcript = get_video_transcript(self.video_id, languages=languages)

        # Merge transcript snippets to fit within token limits
        merged_transcript = merge_transcript_snippets(transcript, max_tokens=500)
        # Store in the database

        self.db.upsert(
            collection="transcripts",
            documents=[
                {
                    "id": f"{self.video_id}_{i}",
                    "video_id": self.video_id,
                    "text": t.text,
                    "keywords": t.text,
                    "start": t.start,
                    "duration": t.duration,
                }
                for i, t in enumerate(merged_transcript)
            ],
            texts=[t.text for t in merged_transcript],
        )

        self.db.upsert(
            collection="videos",
            documents=[
                {
                    "id": self.video_id,
                    "title": video_info.title,
                    "text": video_info.description,
                    "keywords": video_info.description,
                }
            ],
            texts=[video_info.description],
        )

    def run(self, user_question: str):
        # Retrieve relevant transcript snippets from the database
        relevant_snippets = self.db.search(
            collection="transcripts",
            text=user_question,
            filter=f"video_id == '{self.video_id}'",
            output_fields=["text"],
            top_k=5,
        )

        # Build the context for the current question
        transcript_text = "\n".join([snippet["text"] for snippet in relevant_snippets])

        # Build messages list: system + history + current question
        messages = [
            SystemMessage(
                content=SYSTEM_MESSAGE.format(
                    video_title=self.video_info[0]["title"] if self.video_info else "",
                    video_description=(
                        self.video_info[0]["text"] if self.video_info else ""
                    ),
                    transcript_text=transcript_text,
                )
            )
        ]

        # Add chat history messages
        for msg in self.chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # Add current question with context
        messages.append(HumanMessage(user_question))

        # Run the LLM with the message list
        response = self.llm.invoke(messages)

        # Track the conversation in history
        self._add_to_history("user", user_question)
        response_content = (
            response.content if hasattr(response, "content") else str(response)
        )
        self._add_to_history("assistant", response_content)

        return response

    def stream(self, user_question: str):
        """Stream responses from the agent."""
        # Retrieve relevant transcript snippets from the database
        relevant_snippets = self.db.search(
            collection="transcripts",
            text=user_question,
            filter=f"video_id == '{self.video_id}'",
            output_fields=["text"],
            top_k=5,
        )
        video_info = self.db.get(
            collection="videos",
            ids=[self.video_id],
            output_fields=["title", "text"],
        )

        # Build the context for the current question
        transcript_text = "\n".join([snippet["text"] for snippet in relevant_snippets])

        # Build messages list: system + history + current question
        messages = [
            SystemMessage(
                content=SYSTEM_MESSAGE.format(
                    video_title=video_info[0]["title"] if video_info else "",
                    video_description=video_info[0]["text"] if video_info else "",
                    transcript_text=transcript_text,
                )
            )
        ]

        # Add chat history messages
        for msg in self.chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # Add current question with context
        messages.append(HumanMessage(user_question))

        # Track the conversation in history
        self._add_to_history("user", user_question)

        # Stream the response
        full_response = ""
        for chunk in self.llm.stream(messages):
            chunk_content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_response += chunk_content
            yield chunk_content

        # Add full response to history after streaming is complete
        self._add_to_history("assistant", full_response)

    def _add_to_history(self, role: str, content: str):
        """Add a message to chat history and maintain max length."""
        self.chat_history.append({"role": role, "content": content})

        # Keep only the last N messages (counting both user and assistant messages)
        if len(self.chat_history) > self.max_history_messages:
            self.chat_history = self.chat_history[-self.max_history_messages :]

    def clear_history(self):
        """Clear the chat history."""
        self.chat_history = []

    def get_history(self) -> list:
        """Get the current chat history."""
        return self.chat_history.copy()

    @classmethod
    def build(
        cls,
        video_id: str,
        languages: list[str] = None,
        max_history_messages: int = 10,
    ):
        llm = OpenAIModel()
        db = VectorDB()
        return cls(
            video_id=video_id,
            llm=llm,
            db=db,
            languages=languages,
            max_history_messages=max_history_messages,
        )
