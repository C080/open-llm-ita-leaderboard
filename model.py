from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

path = "E:/text-generation-webui-main/models/saiga-7b"

class Agent:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.memory = []

    def answer(self, prompt):
        completion = llm_generate(self.model, self.tokenizer, prompt, self.device)
        self.memory.append(completion)
        return completion
    
    def remember(self, prompt):
        
        # Transform memory and prompt into TF-IDF features
        vectorizer = TfidfVectorizer().fit(self.memory + [prompt])
        tfidf = vectorizer.transform(self.memory + [prompt])

        # Compute cosine similarity between prompt and all memories
        similarities = cosine_similarity(tfidf[-1], tfidf[:-1])

        # Find the index of the most similar memory
        most_similar_index = similarities.argmax()

        # Return the most similar memory
        return self.memory[most_similar_index]
    


def load_model():
    model_params = {
        'low_cpu_mem_usage': True,
        'torch_dtype': torch.bfloat16,
        'use_flash_attention_2': False,
        'load_in_8bit': False, # da fixare bitsandbytes
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(path, **model_params).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path)
    print("Model loaded")
    return model, tokenizer

def llm_generate(model, tokenizer, prompt, device):
    chat = [
    {"role": "user", "content": f"{prompt}"},
    ]
    model_inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    #model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    model_inputs = model_inputs.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=500, do_sample = True, temperature = 1.5, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.batch_decode(generated_ids)[0].split('[/INST]')[1] # un po una porcata
    return answer


