from langchain_core.prompts import PromptTemplate

def load_history_prompt():
    return PromptTemplate(
                    template="""
                    You are given a conversation between a user and an assistant.

                    Use the conversation history to rewrite the user's
                    current question into a standalone question.

                    Do not answer the question.

                    If the current question is already standalone,
                    return it unchanged.

                    Conversation history:
                    {history}

                    Current question:
                    {question}

                    Standalone question:
                    """,
                    input_variables=["history", "question"]
                )



def load_gen_prompt():
    return PromptTemplate(
                    template="""You are a question-answering assistant.

                    Answer the user's question using ONLY the
                    provided video transcript context and conversation history.

                    Rules:

                    1. Use the video transcript as the primary source of truth.
                    2. Use conversation history to understand references
                    to previous questions and answers.
                    3. Do not use outside knowledge.
                    4. If the answer cannot be found in the video context,
                    say:
                    "I don't know based on the provided video."
                    5. Do not invent information.
                    6. Keep the answer clear and concise.

                    Conversation History:
                    {history}

                    Video Context:
                    {context}

                    Question:
                    {question}

                    Answer:""",
                    input_variables=[
                        "history",
                        "context",
                        "question"
                    ]
                )

