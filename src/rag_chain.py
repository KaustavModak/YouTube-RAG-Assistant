from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(retriever, prompt, model, history_prompt):

    parser = StrOutputParser()

    rewrite_chain = history_prompt | model | parser 
    # gets history and current question as inputs and returns the rewritten question

    def prepare_input(inputs):
        history = inputs["history"]
        question = inputs["question"]

        rewritten_question = rewrite_chain.invoke({"history":history,"question":question})

        return {
            "history":history,
            "question":rewritten_question
        }

    prepared_input = RunnableLambda(prepare_input)

    parallel_chain = RunnableParallel({
        "history":RunnableLambda(lambda x:x["history"]),
        "context":RunnableLambda(lambda x:x["question"]) | retriever | RunnableLambda(format_docs),
        "question": RunnableLambda(lambda x:x["question"])
    })

    gen_chain = prompt | model | parser

    rag_chain = prepared_input | parallel_chain | gen_chain

    return rag_chain