from transformers import AutoTokenizer, AutoModelForCausalLM

class CPChatbot:
    def __init__(self, retriever, system_message, model_name="gpt2"):
        self.retriever = retriever
        self.system_message = system_message
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.max_model_len = self.model.config.n_positions  # Usually 1024 for GPT-2

        # Set the padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def chat(self, query):
        retrieved_contexts = self.retriever.retrieve_context(query)
        context_text = "\n".join(retrieved_contexts)
        prompt = f"{self.system_message}\nContext:\n{context_text}\n\nUser Query: {query}\nAnswer: "

        # Tokenize with truncation
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_model_len,
            padding=True
        )

        output_ids = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=self.max_model_len,  # Cannot exceed model's limit
            max_new_tokens=100,  # <-- sets how many tokens to generate
            pad_token_id=self.tokenizer.eos_token_id,  # Avoid warning
            do_sample=True,  # Optional: adds randomness
            top_k=50,
            top_p=0.95
        )
        
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response[len(prompt):].strip()  # Return only the generated part