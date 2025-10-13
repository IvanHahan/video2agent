from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .db import VectorDB
from .prompts import SYSTEM_MESSAGE
from .youtube import (
    get_video_transcript,
    get_youtube_video_info,
    merge_transcript_snippets,
)


def create_chat_prompt_template():
    template = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_MESSAGE),
            (
                "human",
                """Based on the following video information and transcript snippets, please answer the user's question.

Video Title: {title}
Video Description: {description}

Relevant Transcript Snippets:
{transcript_snippets}

User Question: {question}

Please provide a comprehensive answer based on the video content.""",
            ),
        ]
    )
    return template


class VideoAgent:
    def __init__(self, llm: ChatOpenAI, db: VectorDB):
        self.llm = llm
        self.db = db

    def process_youtube_video(self, video_id: str, languages: list[str]):
        # Fetch video info and transcript
        video_info = get_youtube_video_info(video_id)
        transcript = get_video_transcript(video_id, languages=languages)

        # Merge transcript snippets to fit within token limits
        merged_transcript = merge_transcript_snippets(transcript, max_tokens=500)
        # Store in the database
        self.db.upsert("transcripts", merged_transcript, video_id)

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
                for i, t in enumerate(transcript)
            ],
            texts=[t.text for t in transcript],
        )

        self.db.upsert(
            collection="videos",
            documents=[
                {
                    "id": video_id,
                    "title": video_info.title,
                    "description": video_info.description,
                }
            ],
        )

    def run(self, user_question, video_id: str):
        # Retrieve relevant transcript snippets from the database
        relevant_snippets = self.db.search(
            collection="transcripts",
            query=user_question,
            filter=f"video_id == {video_id}",
            k=5,
        )
        video_info = self.db.get(
            collection="videos",
            ids=[video_id],
        )

        # Create the chat template
        template = create_chat_prompt_template()

        # Format the prompt with the retrieved data
        formatted_prompt = template.format_messages(
            title=video_info[0]["title"] if video_info else "",
            description=video_info[0]["description"] if video_info else "",
            transcript_snippets="\n".join(
                [snippet["text"] for snippet in relevant_snippets]
            ),
            question=user_question,
        )

        # Run the LLM with the formatted prompt
        response = self.llm.invoke(formatted_prompt)

        return response.content
