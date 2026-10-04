from langchain_community.vectorstores import FAISS
import os

def create_vector_store(documents, embeddings):

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    return vector_store

def save_vector_store(vector_store, path): # saving vectors locally
    os.makedirs(path,exist_ok=True)
    vector_store.save_local(path)

def load_vector_store(path, embeddings): # loading the vecotrs from local
    vector_store = FAISS.load_local(
        path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store