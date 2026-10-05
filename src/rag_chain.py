from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from src.reranker import rerank_documents

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(retriever, prompt, model, history_prompt, reranker):

    parser = StrOutputParser()

    rewrite_chain = history_prompt | model | parser 
    # gets history and current question as inputs and returns the rewritten question

    def prepare_input(inputs):
        history = inputs["history"]
        question = inputs["question"]

        rewritten_question = rewrite_chain.invoke({"history":history,"question":question}) # gets the rewritten question

        documents = retriever.invoke(rewritten_question) # retrieves 10 docs(candidates)

        reranked_documents = rerank_documents( # returns the best 4 candidates out of 10
            query=rewritten_question,
            documents=documents,
            reranker=reranker,
            top_k=4
        )

        context = format_docs(reranked_documents)

        return {
            "history": history,
            "context": context,
            "question": rewritten_question,
            "documents":reranked_documents    # returns the reranked docs for citations
        }

    prepared_input = RunnableLambda(prepare_input)

    def generate_answer(inputs):
        history = inputs["history"]
        context = inputs["context"]
        question = inputs["question"]
        documents = inputs["documents"]

        answer = (prompt | model | parser).invoke({
            "history":history,
            "context":context,
            "question":question
        })
        return {
            "answer":answer,
            "sources":[
                doc.page_content for doc in documents
            ]
        }

    gen_chain = RunnableLambda(generate_answer)

    rag_chain = prepared_input | gen_chain

    return rag_chain