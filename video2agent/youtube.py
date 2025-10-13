import tiktoken
from pytubefix import YouTube
from youtube_transcript_api import YouTubeTranscriptApi

from .data_model import TranscriptSnippet


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to MM:SS or HH:MM:SS format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def merge_transcript_snippets(
    snippets: list[TranscriptSnippet],
    max_tokens: int = 500,
    encoding_name: str = "cl100k_base",
) -> list[TranscriptSnippet]:
    """
    Merge video transcript snippets into larger chunks with a maximum token limit.
    Each sub-snippet within the merged chunk is marked with its own timestamp.

    Args:
        snippets: List of transcript snippets, each with 'text', 'start', and 'duration' keys
        max_tokens: Maximum number of tokens per merged snippet (default: 500)
        encoding_name: Tokenizer encoding to use (default: "cl100k_base" for GPT-4)

    Returns:
        List of dictionaries with 'text', 'start', and 'duration' keys where:
        - text: formatted string with each sub-snippet marked like "[00:00] text here [00:15] more text"
        - start: start time of the first snippet in the chunk
        - duration: total duration from start to end of the chunk
    """
    if not snippets:
        return []

    encoding = tiktoken.get_encoding(encoding_name)
    merged_snippets = []

    current_parts = []
    current_tokens = 0
    chunk_start = None
    chunk_end = None

    for snippet in snippets:
        text = snippet.text
        start_time = format_timestamp(snippet.start)

        # Format this sub-snippet with its timestamp
        formatted_part = f"[{start_time}] {text}"
        part_tokens = len(encoding.encode(formatted_part))

        # If adding this snippet would exceed max_tokens, save current and start new
        if current_tokens + part_tokens > max_tokens and current_parts:
            # Save the current merged snippet
            merged_snippets.append(
                TranscriptSnippet(
                    text=" ".join(current_parts),
                    start=chunk_start,
                    duration=chunk_end - chunk_start,
                )
            )

            # Start new snippet with this part
            current_parts = [formatted_part]
            current_tokens = part_tokens
            chunk_start = snippet.start
            chunk_end = snippet.start + snippet.duration
        else:
            # Add to current snippet
            current_parts.append(formatted_part)
            current_tokens += part_tokens

            # Track the start and end of the chunk
            if chunk_start is None:
                chunk_start = snippet.start
            chunk_end = snippet.start + snippet.duration

    # Add the last snippet if it has content
    if current_parts:
        merged_snippets.append(
            TranscriptSnippet(
                text=" ".join(current_parts),
                start=chunk_start,
                duration=chunk_end - chunk_start,
            )
        )

    return merged_snippets


def get_video_transcript(
    video_id: str, languages: list[str], preserve_formatting: bool = False
) -> list[TranscriptSnippet]:
    ytt_api = YouTubeTranscriptApi()
    return [
        TranscriptSnippet(
            text=snippet.text, start=snippet.start, duration=snippet.duration
        )
        for snippet in ytt_api.fetch(
            video_id, languages=languages, preserve_formatting=preserve_formatting
        )
    ]


def get_youtube_video_info(video_id: str):
    url = f"https://www.youtube.com/watch?v={video_id}"
    yt = YouTube(url)
    return yt


if __name__ == "__main__":
    video_id = "48ZK2JcoHyU"
    transcript = get_video_transcript(video_id, languages=["uk"])
    info = get_youtube_video_info(video_id)
    merged_transcript = merge_transcript_snippets(transcript, max_tokens=500)
    print(info)
