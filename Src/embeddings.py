from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Ensure PyTorch uses a single thread for better efficiency in small-scale tasks
torch.set_num_threads(1)

# Embedding Generation
class CodeBERTEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def generate_embedding(self, text):
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embedding.numpy()

    def batch_generate_embeddings(self, texts, batch_size=2):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = self.tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

# embedder = CodeBERTEmbedder()
# test_text = "Find the maximum subarray sum"
# embedding = embedder.generate_embedding(test_text)
# print(f"Embedding shape: {embedding.shape}")

# # Test batch embedding
# texts = [
#  "Find the maximum subarray sum",
#  "Implement a segment tree",
#  "Solve the knapsack problem"
# ]
# embeddings = embedder.batch_generate_embeddings(texts)
# print(f"Batch embeddings shape: {embeddings.shape}")

