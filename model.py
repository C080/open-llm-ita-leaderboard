from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Agent:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.memory = []

    def answer(self, prompt, batch_generate=1):
        #completion = llm_generate(self.model, self.tokenizer, prompt, self.device)
        completions = llm_batch_generate(self.model, self.tokenizer, prompt, self.device, batch_generate)
        #self.memory = completions
        return completions
    
    def memorize(self, message):
        self.memory = message
    
    def remeber(self):
        return self.memory
    
    def find_similarity_in_memory(self, similarity_threshold = 0.8):
        
        # Transform memory and prompt into TF-IDF features
        vectorizer = TfidfVectorizer().fit(self.memory)
        tfidf = vectorizer.transform(self.memory)

        # Compute the cosine similarity matrix (comparing all memory elements)
        similarity_matrix = cosine_similarity(tfidf)

        # Extract similar pairs above the threshold
        similar_pairs = []
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] >= similarity_threshold:
                    similar_pairs.append([similarity_matrix[i][j], i ,j])
        
        similar_pairs.sort(reverse=True, key=lambda x: x[0])

        return similar_pairs
    


def load_model(path):
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
    model_inputs = model_inputs.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=500, do_sample = True, temperature = 1.5, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.batch_decode(generated_ids)[0].split('[/INST]')[1] # un po una porcata
    return answer

def llm_batch_generate(model, tokenizer, prompt, device, num_sequences=3):
    chat = [
        {"role": "user", "content": f"{prompt}"},
    ]
    model_inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    model_inputs = model_inputs.to(device)
    generated_ids = model.generate(
        model_inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=1.5,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_sequences,  # Specify number of sequences to return
    )
    
    answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Split answers if the tokenizer includes special tokens that you wish to remove
    # Note: The line below assumes your special token is '[/INST]', adjust if different
    answers = [answer.split('[/INST]')[1] if '[/INST]' in answer else answer for answer in answers]
    
    return answers


