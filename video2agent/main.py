from video2agent.agent import VideoAgent


def main():
    agent = VideoAgent.build()
    video_id = "48ZK2JcoHyU"  # Example YouTube video ID
    user_question = "What is the main topic of the video?"
    # agent.process_youtube_video(video_id, languages=["uk"])
    answer = agent.run(user_question, video_id)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
