# 🎥 Video Search Using NLP

A web application that enables users to search for specific spoken content 
within a video using natural language queries — no manual scrubbing through 
the timeline required.

## How It Works

1. User uploads a video file through the web interface
2. Audio is extracted from the video using MoviePy
3. OpenAI's Whisper model transcribes the audio into timestamped text segments
4. Each segment is converted into a semantic embedding using Sentence 
   Transformers (all-MiniLM-L6-v2)
5. When a user enters a search query, it's embedded the same way, and 
   cosine similarity is used to find the most relevant segments
6. Matching results are displayed with their relevance score, and the 
   video player jumps directly to the matched timestamp

## Tech Stack

- **Speech-to-Text:** OpenAI Whisper
- **NLP/Embeddings:** Sentence Transformers (Hugging Face)
- **Video/Audio Processing:** MoviePy
- **Similarity Search:** Cosine Similarity (PyTorch/sentence-transformers util)
- **Frontend/UI:** Streamlit
- **Language:** Python

## Features

- Upload any video and search its content using plain English queries
- Semantic search — understands meaning, not just exact keyword matches
- Returns top matching segments with confidence scores and timestamps
- Automatically jumps to the exact moment in the video


## Setup

\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`
