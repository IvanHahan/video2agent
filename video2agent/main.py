from video2agent.agent import VideoAgent


def main():
    video_id = "48ZK2JcoHyU"  # Example YouTube video ID
    agent = VideoAgent.build(video_id=video_id, languages=["uk"])
    user_question = "What is the main topic of the video?"
    for chunk in agent.stream(user_question):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    main()
