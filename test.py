from src.ingestion import get_transcript
from src.splitter import split_text
from src.embeddings import get_embeddings
from src.vectorstore import create_vector_store
from src.llm import get_llm
from src.rag_chain import build_rag_chain

from langchain_core.prompts import PromptTemplate

video_id = "Gfr50f6ZBvo"

# ingestion
transcript = get_transcript(video_id=video_id)

# splitting
documents = split_text(transcript)

# embeddings
embeddings = get_embeddings()

# vector store
vector_store = create_vector_store(
    documents=documents,
    embeddings=embeddings
)

# retriever
retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {'k':4}
)

# llm model
model = get_llm()

# prompt
prompt = PromptTemplate(
    template="""
    You are a question-answering assistant.

    Answer ONLY using the provided context.

    If the answer cannot be found in the context,
    say that you don't know based on the provided video.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """,
    input_variables=['context','question']
)

# rag chain
rag_chain = build_rag_chain(
    retriever=retriever,
    prompt=prompt,
    model=model
)

# query
query = "What is this video about"
answer = rag_chain.invoke(query)

print(answer)