class RAGRetriever:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve_context(self, query: str, top_k: int = 3):
        query_embedding = self.embedder.generate_embedding(query)
        results = self.vector_store.search(query_embedding)
        return results
