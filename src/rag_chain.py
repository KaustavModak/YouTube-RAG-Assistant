from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(retriever, prompt, model):

    parallel_chain = RunnableParallel({
        'context':retriever | RunnableLambda(format_docs),
        'question':RunnablePassthrough()
    })

    parser = StrOutputParser()

    gen_chain = prompt | model | parser

    rag_chain = parallel_chain | gen_chain

    return rag_chain