from src.ingestion import get_transcript
from src.splitter import split_text
from src.embeddings import get_embeddings
from src.vectorstore import create_vector_store
from src.llm import get_llm
from src.rag_chain import build_rag_chain
from src.youtube_utils import extract_video_id

from langchain_core.prompts import PromptTemplate

import streamlit as st

# application-level configuration
st.set_page_config(
    page_title="Youtube RAG Assitant",
    layout="wide"
)

# loading embedding model
# st.cache_resource → prevents expensive models from being loaded repeatedly.
@st.cache_resource
def load_embeddings():
    return get_embeddings()

# loading LLM model
@st.cache_resource
def load_llm():
    return get_llm()

embeddings = load_embeddings()
llm = load_llm()

# session state
# Remember user's session data
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit UI
st.title("YouTube RAG Assistant")
st.write(
    "Paste a YouTube video URL and ask questions about its content"
)

# YouTube URL input
youtube_url = st.text_input(
    label="YouTube video URL",
    placeholder="https://www.youtube.com/watch?v=Gfr50f6ZBvo"
)

process_button = st.button(
    label="Process Video",
    type="primary"
)

# process video
if process_button:
    if not youtube_url:
        st.warning("Please paste a YouTube video URL")
    else:
        try:

            # extracting video ID
            video_id = extract_video_id(youtube_url)
            st.info(f"Video ID detected {video_id}")

            # processing
            with st.status(
                label="Processing video...",
                expanded=True
            ):
                
                # 1) getting transcript
                st.write("Fetching transcript...")
                transcript = get_transcript(video_id=video_id)
                st.write(
                    f"Transcript loaded with {len(transcript)} characters"
                )

                # 2) splitting text
                st.write("Splitting text...")
                documents = split_text(transcript)
                st.write(f"Created {len(documents)} chunks")

                # 3) embeddings + FAISS
                st.write("Creating vector store...")
                vector_store = create_vector_store(
                    documents=documents,
                    embeddings=embeddings
                )
                
                # 4) creating retriever
                st.write("Creating retriever...")
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={'k':4}
                )

                # 5) creating prompt
                prompt = PromptTemplate(
                    template="""You are a question-answering assistant.

                    Answer the user's question using ONLY the
                    provided video transcript context.

                    Rules:

                    1. Do not use outside knowledge.
                    2. If the answer cannot be found in the
                    context, say:
                    "I don't know based on the provided video."
                    3. Do not invent information.
                    4. Keep the answer clear and concise.

                    Context:
                    {context}

                    Question:
                    {question}

                    Answer:""",
                    input_variables=['context','question']
                )

                # 6) building rag chain
                st.write("Building RAG pipeline...")

                rag_chain = build_rag_chain(
                    retriever=retriever,
                    prompt=prompt,
                    model=llm
                )

                # 7) store in session
                st.session_state.rag_chain = rag_chain # saves the built rag chain
                st.session_state.video_id = video_id # saves the video id
                st.session_state.messages = [] # clears the old conversation related to old processed video

                st.success(
                    "Video processed successfully! "
                    "You can now ask questions"
                )

        except ValueError as e: # error related to youtube transcripting
            st.error(str(e))

        except Exception as e:
            st.error(f"Something went wrong: {e}")


# chat history
# with st.chat_message("user") : Create a chat bubble for the user.
# with st.chat_message("assistant") : Create a chat bubble for the assistant.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# chat input
question = st.chat_input(
    placeholder="Ask something about the video..."
)

if question:
    # checking whether video is processed
    if st.session_state.rag_chain is None:
        st.warning("Please process the YouTube video first.")
    else:
        # display user question
        with st.chat_message("user"):
            st.markdown(question)

        # saving user question
        st.session_state.messages.append({
            "role":"user",
            "content":question
        })

        # generating answer
        with st.chat_message("assistant"):
            with st.spinner(
                "Searching the video and generating answer..."
            ):
                answer = st.session_state.rag_chain.invoke(question)
            st.markdown(answer)

        # saving assistant response
        st.session_state.messages.append({
            "role":"assistant",
            "content":answer
        })