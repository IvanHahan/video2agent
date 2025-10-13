# Video2Agent 🎥🤖

Video2Agent is an AI-powered chatbot that allows you to have interactive conversations with YouTube videos. Simply provide a YouTube video ID or URL, and the agent will process the video's transcript to answer your questions about its content.

## Features

- 💬 **Dual Interface**: Choose between Chainlit or Streamlit web interfaces
- 🔍 **Semantic Search**: Uses vector embeddings and Pinecone database for intelligent content retrieval
- 🌍 **Multi-language Support**: Process videos in multiple languages including English and Ukrainian
- 📝 **Transcript Processing**: Automatically fetches and processes video transcripts
- 🧠 **Context-Aware Responses**: Maintains chat history for contextual conversations
- 🔄 **Video Switching**: Easily switch between different videos in the same session
- 🎯 **Flexible Vector Storage**: Supports both Pinecone and Milvus vector databases

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key (or Milvus instance)

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
export PINECONE_API_KEY="your-pinecone-api-key"  # If using Pinecone
```

## Usage

### Option 1: Chainlit Interface

Run the interactive Chainlit web interface:

```bash
chainlit run chainlit_app.py
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

### Option 2: Streamlit Interface

Run the Streamlit web application:

```bash
streamlit run streamlit_app.py
```

Then open your browser to the provided URL (typically `http://localhost:8501`).

**Features:**
- Clean sidebar for video input
- Video thumbnail and metadata display
- Real-time streaming responses
- Persistent chat history
- Easy video switching

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
chainlit_app.py         # Chainlit web interface
streamlit_app.py        # Streamlit web interface
requirements.txt        # Python dependencies
video2agent/
├── __init__.py
├── youtube_agent.py    # Main agent implementation
├── youtube.py          # YouTube video processing utilities
├── llm.py              # LLM configuration
├── chains.py           # LangChain integration
├── prompts.py          # System prompts and templates
├── data_model.py       # Data models
├── parsers.py          # Response parsers
├── main.py             # Entry point and utilities
└── db/
    ├── __init__.py
    ├── milvus_db.py    # Milvus vector database integration
    └── pinecone_db.py  # Pinecone vector database integration
```

## Dependencies

- **openai**: OpenAI API client for LLM interactions
- **chainlit**: Interactive chat UI framework
- **streamlit**: Modern web app framework for data applications
- **pinecone**: Pinecone vector database client
- **pytubefix**: YouTube video information retrieval
- **youtube_transcript_api**: YouTube transcript fetching
- **tiktoken**: Token counting for LLM context management
- **langchain_openai**: LangChain OpenAI integration
- **loguru**: Advanced logging library

## How It Works

1. **Video Processing**: When you provide a YouTube video ID, the agent:
   - Fetches the video's transcript using YouTube's API
   - Splits the transcript into manageable snippets
   - Creates vector embeddings of the content
   - Stores embeddings in your chosen vector database (Pinecone or Milvus)

2. **Question Answering**: When you ask a question:
   - Your question is embedded and used to search for relevant video segments
   - The most relevant transcript snippets are retrieved from the vector database
   - An LLM (OpenAI) generates a response based on the retrieved context
   - The response is streamed back to you in real-time

3. **Context Management**: The agent maintains chat history to provide contextually aware responses across multiple interactions.

## Configuration

You can customize the agent behavior by modifying:
- **Languages**: Specify preferred transcript languages when building the agent
- **Max History Messages**: Control how many previous messages are included in context (default: 5)
- **LLM Model**: Configure the OpenAI model in `video2agent/llm.py`
- **Vector DB Settings**: 
  - Pinecone: Configure in `video2agent/db/pinecone_db.py`
  - Milvus: Configure in `video2agent/db/milvus_db.py`
- **UI Preference**: Choose between Chainlit or Streamlit based on your needs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- UI powered by [Chainlit](https://github.com/Chainlit/chainlit) and [Streamlit](https://streamlit.io/)
- Vector storage by [Pinecone](https://www.pinecone.io/) and [Milvus](https://milvus.io/)
- LLM by [OpenAI](https://openai.com/)
- YouTube data fetching via [pytubefix](https://github.com/JuanBindez/pytubefix) and [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api)
