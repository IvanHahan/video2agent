from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from tqdm import tqdm

from .data_model import FrameDescription, TranscriptSnippet
from .db import PineconeDB
from .llm import OpenAIModel
from .prompts import DESCRIBE_FRAME_PROMPT, SYSTEM_MESSAGE
from .vlm import MODEL_URLS, VisionModel
from .youtube import (
    download_video,
    extract_frame_at_timecode,
    get_video_transcript,
    get_youtube_video_info,
    merge_transcript_snippets,
)


class YoutubeVideoAgent:
    def __init__(
        self,
        video_id: str,
        llm: OpenAIModel,
        vlm: VisionModel,
        db: PineconeDB,
        languages: list[str] = None,
        max_history_messages: int = 5,
    ):
        self.video_id = video_id
        self.llm = llm
        self.vlm = vlm
        self.db = db
        self.max_history_messages = max_history_messages
        self.chat_history = []  # List of tuples: (role, content)

        # Process the video and create vector index during initialization
        if languages is None:
            languages = ["en"]
        self._process_youtube_video(languages)

    # def __del__(self):
    #     """Destructor to clean up video data when the agent instance is destroyed."""
    #     try:
    #         # Delete all transcript snippets for this video
    #         self.db.delete(collection="transcripts", filter={"video_id": self.video_id})
    #     except Exception as e:
    #         # Silently handle errors during cleanup to avoid issues during shutdown
    #         pass

    def _process_single_frame(
        self, video_path: str, transcript: TranscriptSnippet, index: int
    ) -> tuple[int, FrameDescription]:
        """Process a single frame and return its index and description."""
        frame_path = extract_frame_at_timecode(video_path, transcript.start)
        response = self.llm.invoke(
            DESCRIBE_FRAME_PROMPT.format(transcript_text=transcript.text),
            image=frame_path,
            text_format=FrameDescription,
        )
        return index, response

    def _understand_video(
        self,
        video_path: str,
        transcripts: List[TranscriptSnippet],
        max_workers: int = 4,
    ) -> List[str]:
        descriptions = [None] * len(transcripts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._process_single_frame, video_path, t, i): i
                for i, t in enumerate(transcripts)
            }

            # Process completed tasks with progress bar
            with tqdm(
                total=len(transcripts), desc="Understanding video frames..."
            ) as pbar:
                for future in as_completed(future_to_index):
                    index, description = future.result()
                    descriptions[index] = description
                    pbar.update(1)

        return descriptions

    def _process_youtube_video(self, languages: list[str]):
        """Process the video and store its transcript in the vector database."""
        # Check if video already exists in the database

        # Fetch video info and transcript
        self.video_info = get_youtube_video_info(self.video_id)
        # Check if video already exists in the database
        transcript = get_video_transcript(self.video_id, languages=languages)

        # Merge transcript snippets to fit within token limits
        merged_transcript = merge_transcript_snippets(transcript, max_tokens=500)
        video_path = download_video(self.video_id)
        descriptions = self._understand_video(video_path, merged_transcript)

        self.db.upsert(
            collection="transcripts",
            documents=[
                {
                    "id": f"{self.video_id}_{i}",
                    "video_id": self.video_id,
                    "text": t.text,
                    "start": t.start,
                    "duration": t.duration,
                    "details": d.key_info,
                }
                for i, (t, d) in enumerate(zip(merged_transcript, descriptions))
            ],
            texts=[t.text for t in merged_transcript],
        )

    def run(self, user_question: str):
        # Retrieve relevant transcript snippets from the database
        relevant_snippets = self.db.search(
            collection="transcripts",
            text=user_question,
            filter={"video_id": self.video_id},
            top_k=5,
        )

        # Build the context for the current question
        snippets_descs = "\n\n".join(
            [
                f"Transcript {i}: {snippet['text']}\nFragment Description: {snippet['key_info']}"
                for i, snippet in enumerate(relevant_snippets)
            ]
        )

        # Build messages list: system + history + current question
        messages = [
            SystemMessage(
                content=SYSTEM_MESSAGE.format(
                    video_title=self.video_info[0]["title"] if self.video_info else "",
                    video_description=(
                        self.video_info[0]["text"] if self.video_info else ""
                    ),
                    snippets=snippets_descs,
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
            filter={"video_id": self.video_id},
            top_k=5,
        )

        # Build the context for the current question
        transcript_text = "\n".join([snippet["text"] for snippet in relevant_snippets])

        # Build messages list: system + history + current question
        messages = [
            SystemMessage(
                content=SYSTEM_MESSAGE.format(
                    video_title=self.video_info.title,
                    video_description=self.video_info.description,
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
        db = PineconeDB()
        vlm = VisionModel(base_url=MODEL_URLS["qwenvl3_30b"])
        return cls(
            video_id=video_id,
            llm=llm,
            vlm=vlm,
            db=db,
            languages=languages,
            max_history_messages=max_history_messages,
        )
