import streamlit as st

from video2agent.youtube_agent import YoutubeVideoAgent


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


def initialize_session_state():
    """Initialize session state variables."""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "video_id" not in st.session_state:
        st.session_state.video_id = None
    if "video_processed" not in st.session_state:
        st.session_state.video_processed = False
    if "messages" not in st.session_state:
        st.session_state.messages = []


def process_video(video_input: str):
    """Process a new video."""
    video_id = extract_video_id(video_input)

    if not video_id:
        return False, "❌ Invalid video ID or URL."

    # try:
    with st.spinner(f"🔄 Processing video: `{video_id}`..."):
        # Create a new agent with the new video
        new_agent = YoutubeVideoAgent.build(video_id=video_id, languages=["en", "uk"])

        # Update session state
        st.session_state.agent = new_agent
        st.session_state.video_id = video_id
        st.session_state.video_processed = True
        st.session_state.messages = []  # Clear chat history

    return (
        True,
        f"✅ Video `{video_id}` has been processed successfully!\n\nChat history has been cleared. You can now ask questions about the video.",
    )

    # except Exception as e:
    #     return False, f"❌ Error processing video: {str(e)}\n\nPlease try again."


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Video2Agent", page_icon="🎥", layout="wide")

    initialize_session_state()

    # Sidebar for video input and controls
    with st.sidebar:
        st.title("🎥 Video2Agent")
        st.markdown("Chat with any YouTube video!")

        st.markdown("---")

        # Video input
        st.subheader("📹 Video Setup")
        video_input = st.text_input(
            "Enter YouTube Video ID or URL:",
            placeholder="48ZK2JcoHyU or https://www.youtube.com/watch?v=48ZK2JcoHyU",
        )

        if st.button("Process Video", type="primary"):
            if video_input:
                success, message = process_video(video_input)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.warning("⚠️ Please enter a video ID or URL.")

        # Show current video
        if st.session_state.video_processed and st.session_state.video_id:
            st.markdown("---")
            st.subheader("Current Video")
            st.info(f"📺 Video ID: `{st.session_state.video_id}`")

            # Display video thumbnail
            st.markdown(
                f"[![Video](https://img.youtube.com/vi/{st.session_state.video_id}/0.jpg)]"
                f"(https://www.youtube.com/watch?v={st.session_state.video_id})"
            )

        st.markdown("---")

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            if st.session_state.agent:
                st.session_state.agent.clear_history()
                st.session_state.messages = []
                st.success("Chat history cleared!")
                st.rerun()
            else:
                st.warning("No chat history to clear.")

        st.markdown("---")

        # Help section
        with st.expander("ℹ️ Help"):
            st.markdown(
                """
            **How to use:**
            1. Enter a YouTube video ID or full URL
            2. Click "Process Video" to analyze the video
            3. Ask questions about the video in the chat
            
            **Example video IDs:**
            - `48ZK2JcoHyU`
            - `https://www.youtube.com/watch?v=48ZK2JcoHyU`
            - `https://youtu.be/48ZK2JcoHyU`
            """
            )

    # Main chat interface
    st.title("💬 Chat with Your Video")

    # Display welcome message if no video is processed
    if not st.session_state.video_processed:
        st.info(
            "👋 Welcome to Video2Agent!\n\n"
            "Please enter a YouTube video ID or URL in the sidebar to get started."
        )
        return

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the video..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Stream response from agent
                for chunk in st.session_state.agent.stream(prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                # Ensure we have a valid response
                if not full_response or full_response.strip() == "":
                    full_response = "I couldn't generate a response. Please try rephrasing your question."
                    message_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"❌ Error: {str(e)}"
                message_placeholder.markdown(full_response)

                # Log the error for debugging
                import traceback

                print(f"Error in question answering: {traceback.format_exc()}")

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


if __name__ == "__main__":
    main()
