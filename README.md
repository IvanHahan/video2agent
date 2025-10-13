# Video2Agent 🎥🤖

Video2Agent is an AI-powered chatbot that allows you to have interactive conversations with YouTube videos. Simply provide a YouTube video ID or URL, and the agent will process the video's transcript to answer your questions about its content.

## Features

- 💬 **Interactive Chat Interface**: Chat with any YouTube video using a web-based UI powered by Chainlit
- 🔍 **Semantic Search**: Uses vector embeddings and Milvus database for intelligent content retrieval
- 🌍 **Multi-language Support**: Process videos in multiple languages including English and Ukrainian
- 📝 **Transcript Processing**: Automatically fetches and processes video transcripts
- 🧠 **Context-Aware Responses**: Maintains chat history for contextual conversations
- 🔄 **Video Switching**: Easily switch between different videos in the same session

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key
- Milvus (vector database)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/IvanHahan/video2agent.git
cd video2agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

4. Ensure Milvus is running (or configure connection to your Milvus instance)

## Usage

### Web Interface (Chainlit)

Run the interactive web interface:

```bash
chainlit run video2agent/app.py
```

Then open your browser and navigate to the provided URL (typically `http://localhost:8000`).

**Commands:**
- `/video {video_id}` - Switch to a new video
- `/clear` - Clear chat history

**Example:**
```
/video 48ZK2JcoHyU
```
or
```
/video https://www.youtube.com/watch?v=48ZK2JcoHyU
```

### Programmatic Usage

You can also use Video2Agent programmatically in your Python scripts:

```python
from video2agent.youtube_agent import YoutubeVideoAgent

# Initialize the agent with a video ID
video_id = "48ZK2JcoHyU"
agent = YoutubeVideoAgent.build(video_id=video_id, languages=["en"])

# Ask questions about the video
user_question = "What is the main topic of the video?"
for chunk in agent.stream(user_question):
    print(chunk, end="", flush=True)
```

## Project Structure

```
video2agent/
├── app.py              # Chainlit web interface
├── youtube_agent.py    # Main agent implementation
├── youtube.py          # YouTube video processing utilities
├── milvus_db.py        # Vector database operations
├── llm.py              # LLM configuration
├── chains.py           # LangChain integration
├── prompts.py          # System prompts and templates
├── data_model.py       # Data models
└── parsers.py          # Response parsers
```

## Dependencies

- **openai**: OpenAI API client
- **pymilvus**: Milvus vector database client
- **chainlit**: Interactive chat UI framework
- **pytubefix**: YouTube video information retrieval
- **youtube_transcript_api**: YouTube transcript fetching
- **tiktoken**: Token counting for LLM context management
- **langchain_openai**: LangChain OpenAI integration

## How It Works

1. **Video Processing**: When you provide a YouTube video ID, the agent:
   - Fetches the video's transcript using YouTube's API
   - Splits the transcript into manageable snippets
   - Creates vector embeddings of the content
   - Stores embeddings in Milvus vector database

2. **Question Answering**: When you ask a question:
   - Your question is embedded and used to search for relevant video segments
   - The most relevant transcript snippets are retrieved
   - An LLM (OpenAI) generates a response based on the context
   - The response is streamed back to you in real-time

3. **Context Management**: The agent maintains chat history to provide contextually aware responses across multiple interactions.

## Configuration

You can customize the agent behavior by modifying:
- **Languages**: Specify preferred transcript languages when building the agent
- **Max History Messages**: Control how many previous messages are included in context
- **LLM Model**: Configure the OpenAI model in `llm.py`
- **Vector DB Settings**: Adjust Milvus connection settings in `milvus_db.py`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- UI powered by [Chainlit](https://github.com/Chainlit/chainlit)
- Vector storage by [Milvus](https://milvus.io/)
- LLM by [OpenAI](https://openai.com/)
