from src.embeddings import get_embeddings
from src.vectorstore import load_vector_store
from src.reranker import get_reranker
from src.youtube_utils import extract_video_id

import os


# --------------------------------------------------
# 1. Video
# --------------------------------------------------

youtube_url = input("Enter YouTube URL: ")

video_id = extract_video_id(youtube_url)

faiss_path = os.path.join(
    "data",
    "faiss",
    video_id
)


# --------------------------------------------------
# 2. Load embeddings
# --------------------------------------------------

print("\nLoading embeddings...")

embeddings = get_embeddings()


# --------------------------------------------------
# 3. Load FAISS
# --------------------------------------------------

print("Loading FAISS vector store...")

vector_store = load_vector_store(
    path=faiss_path,
    embeddings=embeddings
)


# --------------------------------------------------
# 4. Create retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)


# --------------------------------------------------
# 5. Load reranker
# --------------------------------------------------

print("Loading reranker...")

reranker = get_reranker()


# --------------------------------------------------
# 6. Ask question
# --------------------------------------------------

query = input("\nEnter your question: ")


# --------------------------------------------------
# 7. FAISS retrieval
# --------------------------------------------------

documents = retriever.invoke(query)


print("\n" + "=" * 70)
print("FAISS RESULTS")
print("=" * 70)

for i, doc in enumerate(documents):

    print(f"\nRank {i + 1}")
    print("-" * 70)
    print(doc.page_content[:300])


# --------------------------------------------------
# 8. Reranking
# --------------------------------------------------

pairs = [
    [query, doc.page_content]
    for doc in documents
]

scores = reranker.predict(pairs)


ranked_documents = sorted(
    zip(documents, scores),
    key=lambda x: x[1],
    reverse=True
)


# --------------------------------------------------
# 9. Show reranked results
# --------------------------------------------------

print("\n" + "=" * 70)
print("RERANKED RESULTS")
print("=" * 70)

for i, (doc, score) in enumerate(ranked_documents):

    print(f"\nRank {i + 1}")
    print(f"Score: {score:.4f}")
    print("-" * 70)
    print(doc.page_content[:300])


# --------------------------------------------------
# 10. Final top 4
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL TOP 4 DOCUMENTS")
print("=" * 70)

for i, (doc, score) in enumerate(ranked_documents[:4]):

    print(f"\nRank {i + 1}")
    print(f"Score: {score:.4f}")
    print("-" * 70)
    print(doc.page_content[:500])