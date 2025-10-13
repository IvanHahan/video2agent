import chainlit as cl

from video2agent.agent import VideoAgent

# Initialize the agent globally
agent = None


@cl.on_chat_start
async def start():
    """Initialize the chat session and prompt user for video ID."""
    global agent

    # Initialize the agent
    agent = VideoAgent.build()

    # Store agent in user session
    cl.user_session.set("agent", agent)
    cl.user_session.set("video_id", None)
    cl.user_session.set("video_processed", False)

    # Welcome message
    await cl.Message(
        content="👋 Welcome to Video2Agent! \n\n"
        "I can help you chat with any YouTube video. "
        "Just provide me with a YouTube video ID or URL, and I'll process it for you.\n\n"
        "**Example:** `48ZK2JcoHyU` or `https://www.youtube.com/watch?v=48ZK2JcoHyU`\n\n"
        "**Commands:**\n"
        "- `/video` - Switch to a new video\n"
        "- `/clear` - Clear chat history\n"
    ).send()

    # Ask for video ID
    res = await cl.AskUserMessage(
        content="Please enter the YouTube video ID or URL:", timeout=300
    ).send()

    if res:
        video_input = res["output"].strip()

        # Extract video ID from URL if necessary
        video_id = extract_video_id(video_input)

        if video_id:
            # Show loading message
            msg = cl.Message(
                content=f"🔄 Processing video: `{video_id}`...\n\nThis may take a moment."
            )
            await msg.send()

            try:
                # Process the video
                agent.process_youtube_video(video_id, languages=["en", "uk"])

                # Update session only after successful processing
                cl.user_session.set("video_id", video_id)
                cl.user_session.set("video_processed", True)

                # Update message
                msg.content = f"✅ Video `{video_id}` has been processed successfully!\n\nYou can now ask questions about the video."
                await msg.update()

            except Exception as e:
                msg.content = f"❌ Error processing video: {str(e)}\n\nPlease use `/video` command to try a different video."
                await msg.update()
                # Don't set video as processed if it failed
                cl.user_session.set("video_processed", False)
        else:
            await cl.Message(
                content="❌ Invalid video ID or URL. Please use `/video` command to try again."
            ).send()


def extract_video_id(input_str: str) -> str:
    """Extract video ID from YouTube URL or return the ID if already provided."""
    input_str = input_str.strip()

    # Check if it's already a video ID (11 characters)
    if len(input_str) == 11 and not ("/" in input_str or "." in input_str):
        return input_str

    # Extract from various YouTube URL formats
    patterns = [
        "watch?v=",
        "youtu.be/",
        "youtube.com/embed/",
        "youtube.com/v/",
    ]

    for pattern in patterns:
        if pattern in input_str:
            start_idx = input_str.find(pattern) + len(pattern)
            video_id = input_str[start_idx : start_idx + 11]
            # Remove any additional parameters
            if "&" in video_id:
                video_id = video_id[: video_id.find("&")]
            if "?" in video_id:
                video_id = video_id[: video_id.find("?")]
            return video_id

    return input_str


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and respond with answers from the video."""
    agent = cl.user_session.get("agent")
    video_id = cl.user_session.get("video_id")
    video_processed = cl.user_session.get("video_processed")

    # Check if video is processed
    if not video_processed or not video_id:
        await cl.Message(
            content="⚠️ Please provide a video ID first. Restart the chat to begin."
        ).send()
        return

    # Check for special commands
    if message.content.lower().startswith("/video"):
        # Allow user to switch to a new video
        res = await cl.AskUserMessage(
            content="Please enter the new YouTube video ID or URL:", timeout=300
        ).send()

        if res:
            video_input = res["output"].strip()
            new_video_id = extract_video_id(video_input)

            if new_video_id:
                msg = cl.Message(content=f"🔄 Processing video: `{new_video_id}`...")
                await msg.send()

                try:
                    # Clear history BEFORE processing new video
                    agent.clear_history()

                    # Process the new video (this will delete old transcript data)
                    agent.process_youtube_video(new_video_id, languages=["en", "uk"])

                    # Update session with new video ID
                    cl.user_session.set("video_id", new_video_id)
                    cl.user_session.set("video_processed", True)

                    msg.content = f"✅ Video `{new_video_id}` has been processed successfully!\n\nChat history has been cleared. You can now ask questions about the new video."
                    await msg.update()

                except Exception as e:
                    msg.content = (
                        f"❌ Error processing video: {str(e)}\n\nPlease try again."
                    )
                    await msg.update()
                    # Reset to previous video state if processing failed
                    cl.user_session.set("video_id", video_id)
            else:
                await cl.Message(content="❌ Invalid video ID or URL.").send()
        return

    if message.content.lower().startswith("/clear"):
        # Clear chat history
        agent.clear_history()
        await cl.Message(
            content="🗑️ Chat history has been cleared. Your next question will start a fresh conversation."
        ).send()
        return

    if message.content.lower().startswith("/history"):
        # Show current chat history
        history = agent.get_history()
        if not history:
            await cl.Message(content="📭 No chat history yet.").send()
        else:
            history_text = f"📜 **Chat History** ({len(history)} messages):\n\n"
            for i, msg in enumerate(history, 1):
                role = "**You**" if msg["role"] == "user" else "**Assistant**"
                content = (
                    msg["content"][:100] + "..."
                    if len(msg["content"]) > 100
                    else msg["content"]
                )
                history_text += f"{i}. {role}: {content}\n\n"
            await cl.Message(content=history_text).send()
        return

    # Show thinking indicator
    msg = cl.Message(content="🤔 Thinking...")
    await msg.send()

    try:
        # Get answer from agent
        answer = agent.run(message.content, video_id)

        # Extract content from the response
        if hasattr(answer, "content"):
            response_text = answer.content
        else:
            response_text = str(answer)

        # Ensure we have a valid response
        if not response_text or response_text.strip() == "":
            response_text = (
                "I couldn't generate a response. Please try rephrasing your question."
            )

        # Update message with the answer
        msg.content = response_text
        await msg.update()

    except Exception as e:
        error_message = f"❌ Error: {str(e)}"
        msg.content = error_message
        await msg.update()

        # Log the error for debugging
        import traceback

        print(f"Error in question answering: {traceback.format_exc()}")


@cl.set_chat_profiles
async def chat_profile():
    """Define chat profiles if needed."""
    return [
        cl.ChatProfile(
            name="Video Chat",
            markdown_description="Chat with YouTube videos",
            icon="https://api.iconify.design/logos:youtube-icon.svg",
        ),
    ]


if __name__ == "__main__":
    # This is just for reference, use `chainlit run app.py` to start
    pass
