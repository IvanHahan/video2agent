import os
from pathlib import Path

import cv2
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


def download_video(video: str, output_dir: str = "frames") -> str:
    """
    Download a YouTube video to a temporary location.

    Args:
        video_id: YouTube video ID
        output_dir: Directory to save the video (default: "frames")

    Returns:
        Path to the downloaded video file
    """
    # Create output directory if it doesn't exist

    # Get video stream URL
    yt = get_youtube_video_info(video) if isinstance(video, str) else video

    output_dir = os.path.join(output_dir, yt.video_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Try to get 1080p stream first, then fall back to lower resolutions
    stream = (
        yt.streams.filter(
            adaptive=True, mime_type="video/mp4", resolution="1080p"
        ).first()
        or yt.streams.filter(
            adaptive=True, mime_type="video/mp4", resolution="720p"
        ).first()
        or yt.streams.filter(adaptive=True, mime_type="video/mp4")
        .order_by("resolution")
        .desc()
        .first()
    )

    if not stream:
        raise ValueError(f"No suitable video stream found for video ID: {video_id}")

    # Download video to temporary location
    temp_video_path = stream.download(output_path=output_dir, filename="temp_video.mp4")

    # Check if video was successfully downloaded
    if not os.path.exists(temp_video_path):
        raise RuntimeError(f"Failed to download video for video ID: {yt.video_id}")

    return temp_video_path


def extract_frames_at_intervals(
    video_path: str, output_dir: str = "frames", interval: int = 30
) -> list[str]:
    """
    Extract frames from a video file at specified intervals.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames (default: "frames")
        interval: Interval in seconds between frame extractions (default: 30)

    Returns:
        List of file paths to extracted frames
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Open video with OpenCV
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)  # Convert seconds to frame count

    frame_paths = []
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame at specified intervals
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            frame_filename = (
                f"frame_{saved_count:04d}_{format_timestamp(timestamp)}.jpg"
            )
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1

        frame_count += 1

    cap.release()
    return frame_paths


def extract_frame_at_timecode(
    video_path: str, timecode: float, filename: str = None
) -> str:
    """
    Extract a single frame from a video at a specific timecode.

    Args:
        video_path: Path to the video file
        timecode: Time in seconds where to extract the frame
        output_dir: Directory to save extracted frame (default: "frames")
        filename: Optional custom filename (default: auto-generated with timestamp)

    Returns:
        Path to the extracted frame
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(video_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Open video with OpenCV
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    if timecode < 0 or timecode > duration:
        cap.release()
        raise ValueError(
            f"Timecode {timecode}s is out of range. Video duration is {duration:.2f}s"
        )

    # Set video position to the desired timecode
    frame_number = int(timecode * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to extract frame at timecode {timecode}s")

    # Generate filename if not provided
    if filename is None:
        filename = f"frame_{format_timestamp(timecode)}.jpg"

    frame_path = os.path.join(output_dir, filename)
    cv2.imwrite(frame_path, frame)

    return frame_path


def extract_frame_from_youtube_stream(
    video_id: str, timecode: float, output_dir: str = "frames", filename: str = None
) -> str:
    """
    Extract a single frame from a YouTube video at a specific timecode without downloading the full video.
    Uses direct streaming to seek to the desired position.

    Args:
        video_id: YouTube video ID or full YouTube object
        timecode: Time in seconds where to extract the frame
        output_dir: Directory to save extracted frame (default: "frames")
        filename: Optional custom filename (default: auto-generated with timestamp)

    Returns:
        Path to the extracted frame
    """
    # Get YouTube video info
    yt = get_youtube_video_info(video_id) if isinstance(video_id, str) else video_id

    # Create output directory with video ID
    output_path = os.path.join(output_dir, yt.video_id)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Get the best available stream (prefer 720p for balance of quality and speed)
    stream = (
        yt.streams.filter(
            adaptive=True, mime_type="video/mp4", resolution="720p"
        ).first()
        or yt.streams.filter(
            adaptive=True, mime_type="video/mp4", resolution="1080p"
        ).first()
        or yt.streams.filter(adaptive=True, mime_type="video/mp4")
        .order_by("resolution")
        .desc()
        .first()
    )

    if not stream:
        raise ValueError(f"No suitable video stream found for video ID: {yt.video_id}")

    # Get the stream URL
    stream_url = stream.url

    # Open video stream with OpenCV
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video stream for video ID: {yt.video_id}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate duration (if total_frames is available, otherwise skip validation)
    if total_frames > 0:
        duration = total_frames / fps
        if timecode < 0 or timecode > duration:
            cap.release()
            raise ValueError(
                f"Timecode {timecode}s is out of range. Video duration is {duration:.2f}s"
            )

    # Set video position to the desired timecode
    frame_number = int(timecode * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Alternative: seek by milliseconds (sometimes more reliable for streams)
    cap.set(cv2.CAP_PROP_POS_MSEC, timecode * 1000)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(
            f"Failed to extract frame at timecode {timecode}s from stream"
        )

    # Generate filename if not provided
    if filename is None:
        filename = f"frame_{format_timestamp(timecode)}.jpg"

    frame_path = os.path.join(output_path, filename)
    cv2.imwrite(frame_path, frame)

    return frame_path


def get_video_frames(video_id: str, output_dir: str = "frames", interval: int = 30):
    """
    Extract frames from a YouTube video at specified intervals.

    Args:
        video_id: YouTube video ID
        output_dir: Directory to save extracted frames (default: "frames")
        interval: Interval in seconds between frame extractions (default: 30)

    Returns:
        List of file paths to extracted frames
    """
    temp_video_path = download_video(video_id, output_dir)

    try:
        frame_paths = extract_frames_at_intervals(temp_video_path, output_dir, interval)
        return frame_paths
    finally:
        # Clean up temporary video file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


def get_youtube_video_info(video_id: str):
    url = f"https://www.youtube.com/watch?v={video_id}"
    yt = YouTube(url)
    return yt


if __name__ == "__main__":
    video_id = "48ZK2JcoHyU"
    transcript = get_video_transcript(video_id, languages=["uk"])
    info = get_youtube_video_info(video_id)
    merged_transcript = merge_transcript_snippets(transcript, max_tokens=500)
    frame_paths = get_video_frames(
        video_id, interval=60
    )  # Extract a frame every 60 seconds
    print(info)
