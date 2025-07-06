import faiss
import pickle
import numpy as np
from embeddings import CodeBERTEmbedder
# Assuming CodeBERTEmbedder is correctly defined in a previous cell or imported

class VectorStore:
    def __init__(self, index_path="vector.index", texts_path="texts.pkl"):
        self.index_path = index_path
        self.texts_path = texts_path
        # Assuming the embedding dimension is 768 based on 'bert-base-uncased'
        self.index = faiss.IndexFlatL2(768)
        self.texts = []
        self.embedder = CodeBERTEmbedder()

    # Corrected search method:
    def search(self, query_embedding, topk=3): # Now accepts query_embedding directly
        # Reshape the query embedding for FAISS
        # Ensure query_embedding is a numpy array of float32
        if not isinstance(query_embedding, np.ndarray):
             query_embedding = np.array(query_embedding, dtype=np.float32)
        elif query_embedding.dtype != np.float32:
             query_embedding = query_embedding.astype(np.float32)

        query_embedding = query_embedding.reshape(1, -1)

        # Perform the search using the provided query embedding
        # Ensure the index is not empty before searching
        if self.index.ntotal == 0:
            print("Warning: FAISS index is empty. Cannot perform search.")
            return [] # Return an empty list if index is empty

        distances, indices = self.index.search(query_embedding, topk)

        # Retrieve the results based on indices
        results = [self.texts[i] for i in indices[0] if i < len(self.texts)] # Add index check

        return results

    def add(self, text):
        embedding = self.embedder.generate_embedding(text)
        self.index.add(embedding.reshape(1, -1))
        self.texts.append(text)

    def add_batch(self, texts):
        all_embeddings = []
        for text in texts:
            embedding = self.embedder.generate_embedding(text)
            all_embeddings.append(embedding)
        embeddings_array = np.vstack(all_embeddings)
        self.index.add(embeddings_array)
        self.texts.extend(texts)

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.texts_path, 'wb') as f:
            pickle.dump(self.texts, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.texts_path, 'rb') as f:
            self.texts = pickle.load(f)