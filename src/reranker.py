from sentence_transformers import CrossEncoder

def get_reranker():
    reranker = CrossEncoder(
        model_name_or_path= "BAAI/bge-reranker-base"
    )
    return reranker

def rerank_documents(query, documents, reranker, top_k=4):

    pairs = [
        [query,doc.page_content] for doc in documents
    ]
    # pairs = [
    #     [query, doc1.page_content],
    #     [query, doc2.page_content],
    #     ...
    #     [query, doc10.page_content]
    # ]

    scores = reranker.predict(pairs) 
    # produces score for each pair -> [0.32, 0.15, 0.91, 0.47, ...]
    # higher score -> more relevent

    ranked_documents = sorted(
        zip(documents,scores),
        key=lambda x:x[1], # on the basis of scores
        reverse=True       # descending order
    )

    return [
        doc for doc,score in ranked_documents[:top_k] # returns top_k
    ]