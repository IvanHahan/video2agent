from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

from .db import VectorDB
from .llm import OpenAIModel
from .prompts import SYSTEM_MESSAGE
from .youtube import (
    get_video_transcript,
    get_youtube_video_info,
    merge_transcript_snippets,
)


class VideoAgent:
    def __init__(self, llm: ChatOpenAI, db: VectorDB, max_history_messages: int = 5):
        self.llm = llm
        self.db = db
        self.max_history_messages = max_history_messages
        self.chat_history = []  # List of tuples: (role, content)

    def process_youtube_video(self, video_id: str, languages: list[str]):
        # Fetch video info and transcript
        video_info = get_youtube_video_info(video_id)
        transcript = get_video_transcript(video_id, languages=languages)

        # Merge transcript snippets to fit within token limits
        merged_transcript = merge_transcript_snippets(transcript, max_tokens=500)
        # Store in the database

        self.db.upsert(
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
                for i, t in enumerate(merged_transcript)
            ],
            texts=[t.text for t in merged_transcript],
        )

        self.db.upsert(
            collection="videos",
            documents=[
                {
                    "id": video_id,
                    "title": video_info.title,
                    "text": video_info.description,
                    "keywords": video_info.description,
                }
            ],
            texts=[video_info.description],
        )

    def run(self, user_question, video_id: str):
        # Retrieve relevant transcript snippets from the database
        relevant_snippets = self.db.search(
            collection="transcripts",
            text=user_question,
            filter=f"video_id == '{video_id}'",
            output_fields=["text"],
            top_k=5,
        )
        video_info = self.db.get(
            collection="videos",
            ids=[video_id],
            output_fields=["title", "text"],
        )

        # Build the context for the current question
        context = f"""Based on the following video information and transcript snippets, please answer the user's question.

Video Title: {video_info[0]["title"] if video_info else ""}
Video Description: {video_info[0]["text"] if video_info else ""}

Relevant Transcript Snippets:
{"\n".join([snippet["text"] for snippet in relevant_snippets])}

User Question: {user_question}

Please provide a comprehensive answer based on the video content."""

        # Build messages list: system + history + current question
        messages = [SystemMessage(content=SYSTEM_MESSAGE)]
        
        # Add chat history messages
        for msg in self.chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # Add current question with context
        messages.append(HumanMessage(content=context))

        # Run the LLM with the message list
        response = self.llm.invoke(messages)

        # Track the conversation in history
        self._add_to_history("user", user_question)
        response_content = response.content if hasattr(response, "content") else str(response)
        self._add_to_history("assistant", response_content)

        return response

    def _add_to_history(self, role: str, content: str):
        """Add a message to chat history and maintain max length."""
        self.chat_history.append({"role": role, "content": content})
        
        # Keep only the last N messages (counting both user and assistant messages)
        if len(self.chat_history) > self.max_history_messages:
            self.chat_history = self.chat_history[-self.max_history_messages:]
    
    def clear_history(self):
        """Clear the chat history."""
        self.chat_history = []
    
    def get_history(self) -> list:
        """Get the current chat history."""
        return self.chat_history.copy()

    @classmethod
    def build(cls, max_history_messages: int = 10):
        llm = OpenAIModel()
        db = VectorDB()
        return cls(llm, db, max_history_messages=max_history_messages)
