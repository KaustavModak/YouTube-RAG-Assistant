#  YouTube RAG Assistant

An AI-powered **YouTube Video Question Answering Assistant** built using **Retrieval-Augmented Generation (RAG)**.

The application allows users to provide a YouTube video URL and interact with the video's transcript through a conversational chat interface.

Instead of relying on the language model's general knowledge, the system retrieves relevant information from the video's transcript and uses it as context for generating answers.

---

##  Overview

The YouTube RAG Assistant follows a complete Retrieval-Augmented Generation pipeline:

```text
YouTube Video URL
        │
        ▼
Extract Video ID
        │
        ▼
Fetch Transcript
        │
        ▼
Text Chunking
        │
        ▼
Generate Embeddings
        │
        ▼
Persistent FAISS Vector Store
        │
        ▼
Similarity Retrieval
        │
        ▼
Cross-Encoder Reranking
        │
        ▼
Question Rewriting
        │
        ▼
Qwen LLM
        │
        ▼
Answer + Transcript References
```

The system also maintains conversation history, allowing users to ask follow-up questions naturally.

---

##  Features

###  YouTube Video Processing

- Accepts YouTube video URLs.
- Extracts the YouTube video ID.
- Retrieves the video's transcript using FreeTranscriptAPI.
- Processes the transcript for downstream retrieval.

###  Intelligent Text Chunking

The transcript is divided into smaller overlapping chunks using:

```text
RecursiveCharacterTextSplitter
```

Current configuration:

```python
chunk_size = 1000
chunk_overlap = 200
```

This allows the retrieval system to work with smaller and more relevant pieces of the transcript.

---

###  Semantic Embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to convert transcript chunks into numerical vector representations.

Embeddings are normalized before being stored.

```python
encode_kwargs={
    "normalize_embeddings": True
}
```

---

###  Persistent FAISS Vector Store

Transcript embeddings are stored using **FAISS**.

Each YouTube video gets its own persistent vector store:

```text
data/
└── faiss/
    └── <video_id>/
        ├── index.faiss
        └── index.pkl
```

If the same video is processed again, the existing vector store can be loaded instead of recreating the embeddings.

---

###  Semantic Retrieval

When the user asks a question:

1. The question is converted into a semantic representation.
2. FAISS searches for relevant transcript chunks.
3. The top candidate documents are retrieved.

Current retrieval configuration:

```python
search_type="similarity"
search_kwargs={
    "k": 10
}
```

The system initially retrieves 10 candidate chunks.

---

###  Cross-Encoder Reranking

The retrieved documents are further ranked using:

```text
BAAI/bge-reranker-base
```

The reranker evaluates:

```text
(query, document)
```

pairs and assigns relevance scores.

The best four documents are selected as the final context:

```python
top_k = 4
```

This improves retrieval quality when the most relevant information is not among the highest-ranked FAISS results.

---

###  Conversational RAG

The application supports follow-up questions.

For example:

```text
User:
What is RAG?

Assistant:
RAG combines retrieval with generation...

User:
Why is it useful?

Assistant:
It is useful because...
```

The second question can depend on the previous conversation.

The system therefore uses conversation history to rewrite the user's question into a standalone question before retrieval.

---

###  Question Rewriting

Before performing retrieval, the system sends:

```text
Conversation History
+
Current Question
```

to the language model.

The model rewrites the question into a standalone question.

For example:

```text
Conversation:

User:
What is retrieval augmented generation?

Assistant:
RAG retrieves relevant information...

User:
Why is it better?
```

The system can rewrite the second question as:

```text
Why is retrieval augmented generation better?
```

This standalone question is then used for retrieval and reranking.

---

###  Qwen Language Model

The application uses:

```text
Qwen/Qwen2.5-72B-Instruct
```

through Hugging Face Inference Providers.

The model is responsible for:

- Rewriting conversational questions.
- Generating the final answer.
- Following the RAG prompt instructions.

The generation prompt instructs the model to:

- Use the transcript as the primary source.
- Use conversation history for context.
- Avoid outside knowledge.
- Avoid hallucinating information.
- State when the answer cannot be found in the video.

---

###  Transcript References

The application displays the transcript sections used to generate the answer.

Example:

```text
Answer:
Retrieval-Augmented Generation combines information
retrieval with language model generation...

References:

> "The retrieval system searches the database..."

> "The retrieved documents are then passed..."
```

This gives users visibility into the transcript content that supported the generated answer.

---

###  Streamlit Interface

The application is built using Streamlit and provides:

- YouTube URL input
- Video processing
- Conversational chat interface
- Loading indicators
- Answer generation
- Transcript references
- Persistent session-based conversation history

---

#  Project Structure

```text
YOUTUBE RAG/
│
├── app.py
├── test.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
└── src/
    ├── __init__.py
    ├── ingestion.py
    ├── splitter.py
    ├── embeddings.py
    ├── vectorstore.py
    ├── llm.py
    ├── rag_chain.py
    ├── prompts.py
    ├── reranker.py
    └── youtube_utils.py
```

---

#  Module Responsibilities

## `app.py`

Main Streamlit application.

Responsible for:

- UI
- YouTube URL input
- Video processing
- Session state
- Conversation history
- Vector store loading/creation
- RAG chain initialization
- Answer display
- Transcript references

---

## `src/youtube_utils.py`

Responsible for extracting the video ID from different YouTube URL formats.

Example:

```text
https://www.youtube.com/watch?v=x63HCoDfAhQ
```

becomes:

```text
x63HCoDfAhQ
```

---

## `src/ingestion.py`

Responsible for retrieving the YouTube transcript through:

```text
FreeTranscriptAPI
```

The transcript is converted into a single text representation before chunking.

---

## `src/splitter.py`

Responsible for splitting the transcript into overlapping chunks.

Uses:

```text
RecursiveCharacterTextSplitter
```

Configuration:

```text
Chunk Size: 1000
Overlap: 200
```

---

## `src/embeddings.py`

Responsible for creating the embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model runs locally on CPU.

---

## `src/vectorstore.py`

Responsible for:

- Creating FAISS vector stores.
- Saving vector stores.
- Loading previously saved vector stores.

---

## `src/reranker.py`

Responsible for reranking retrieved documents using:

```text
BAAI/bge-reranker-base
```

The reranker improves the relevance of the final context supplied to the language model.

---

## `src/llm.py`

Responsible for initializing:

```text
Qwen/Qwen2.5-72B-Instruct
```

through:

```text
HuggingFaceEndpoint
ChatHuggingFace
```

---

## `src/prompts.py`

Contains the prompts used by the RAG pipeline.

There are two primary prompts:

### History Prompt

Converts conversational questions into standalone questions.

### Generation Prompt

Generates the final answer using:

```text
Conversation History
+
Retrieved Video Context
+
Standalone Question
```

---

## `src/rag_chain.py`

Contains the main RAG pipeline.

The pipeline performs:

```text
Conversation History
        ↓
Question Rewriting
        ↓
FAISS Retrieval
        ↓
Cross-Encoder Reranking
        ↓
Context Construction
        ↓
Answer Generation
        ↓
Transcript References
```

---

#  RAG Pipeline in Detail

## Step 1 — User provides a YouTube URL

Example:

```text
https://www.youtube.com/watch?v=x63HCoDfAhQ
```

The video ID is extracted.

---

## Step 2 — Transcript Retrieval

The application requests the transcript through FreeTranscriptAPI.

The transcript is then converted into text.

---

## Step 3 — Text Chunking

The transcript is divided into manageable chunks:

```text
Chunk 1
Chunk 2
Chunk 3
...
```

Chunks overlap to reduce the chance of losing contextual information between boundaries.

---

## Step 4 — Embedding Generation

Each chunk is converted into an embedding using:

```text
all-MiniLM-L6-v2
```

Conceptually:

```text
Transcript Chunk
       ↓
Embedding Model
       ↓
Vector
```

---

## Step 5 — FAISS Storage

The vectors are stored in FAISS.

This allows semantic similarity searches to be performed efficiently.

---

## Step 6 — Question Rewriting

For conversational queries, the previous conversation and current question are passed to the LLM.

The resulting standalone question is used for retrieval.

---

## Step 7 — Initial Retrieval

FAISS retrieves the top 10 semantically similar transcript chunks.

```text
User Question
      ↓
FAISS
      ↓
10 Candidate Documents
```

---

## Step 8 — Reranking

The BGE Cross-Encoder evaluates the relevance of the 10 candidates.

```text
10 Retrieved Documents
          ↓
Cross Encoder
          ↓
Relevance Scores
          ↓
Top 4 Documents
```

---

## Step 9 — Context Construction

The top four transcript chunks are combined into the context supplied to the language model.

```text
Document 1
+
Document 2
+
Document 3
+
Document 4
```

---

## Step 10 — Answer Generation

Qwen receives:

```text
Conversation History
+
Video Context
+
Question
```

and generates the final answer.

---

## Step 11 — References

The transcript chunks used as final context are returned along with the answer.

This allows the application to display the source transcript sections to the user.

---

#  Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Web interface |
| LangChain | RAG orchestration |
| Hugging Face | LLM and model integration |
| Qwen 2.5 72B | Text generation |
| Sentence Transformers | Embeddings and reranking |
| all-MiniLM-L6-v2 | Text embeddings |
| BGE Reranker Base | Document reranking |
| FAISS | Vector storage and similarity search |
| FreeTranscriptAPI | YouTube transcript retrieval |

---

#  Installation

## 1. Clone the repository

```bash
git clone https://github.com/KaustavModak/youtube-rag-assistant.git
```

Move into the project:

```bash
cd youtube-rag-assistant
```

---

## 2. Create a virtual environment

Using Python:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

#  Environment Variables

Create a `.env` file in the project root:

```env
FREETRANSCRIPT_API_KEY=your_free_transcript_api_key
HF_TOKEN=your_huggingface_token
```

### Hugging Face Token

The Hugging Face token is used to access the Qwen model through Hugging Face Inference Providers.

Your token should have permission to make inference requests.

**Never commit your `.env` file or API keys to GitHub.**

---

# ▶ Running the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

---

#  Deployment

The application can be deployed using **Streamlit Community Cloud**.

For deployment, add the required secrets through the Streamlit Cloud application settings.

Example:

```toml
FREETRANSCRIPT_API_KEY = "your_api_key"
HF_TOKEN = "hf_your_token"
```

Do not place API keys directly inside the source code.

---

#  Security

The following files and directories should not be committed:

```text
.env
.venv/
venv/
__pycache__/
data/faiss/
```

Recommended `.gitignore`:

```gitignore
.env
__pycache__/
*.pyc
*.pyo
.venv/
venv/
data/faiss/
.ipynb_checkpoints/
```

---

#  Example Interaction

### User

```text
What is retrieval augmented generation?
```

### Assistant

```text
Retrieval-Augmented Generation is a technique that
retrieves relevant information from a knowledge source
and provides it to a language model as context for
generating an answer.
```

### Follow-up

```text
Why is it useful?
```

The system uses the previous conversation to understand that:

```text
"it"
```

refers to:

```text
Retrieval-Augmented Generation
```

The question is rewritten before retrieval and the relevant transcript chunks are retrieved again.

---

#  Design Goals

The project focuses on building a practical RAG system with:

- Conversational question answering
- Persistent vector storage
- Semantic retrieval
- Cross-encoder reranking
- Grounded generation
- Transcript-based references
- Separation of ingestion, retrieval, generation, and UI components

---

#  Future Improvements

Potential future improvements include:

- Specialized video interaction modes
- Better summarization workflows
- Quiz generation
- Additional retrieval optimizations
- Improved UI/UX
- Support for additional transcript languages

---

#  Author

**Kaustav Modak**

AI/ML | NLP | Deep Learning | Data Science

GitHub:  
https://github.com/KaustavModak

LinkedIn:  
https://www.linkedin.com/in/kaustav-modak-214173276/