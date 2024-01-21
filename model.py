from transformers import AutoModelForCausalLM, AutoTokenizer 
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

class Agent:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.memory = []

    def answer(self, prompt, max_tokens, batch_generate=1):
            # Un po fuorviante puo ritornare una lista di risposte ma anche una singola stringa
            return llm_batch_generate(self.model, self.tokenizer, prompt, self.device, max_tokens, batch_generate)
    
    def memorize(self, message):
        self.memory = message
    
    def remember(self):
        return self.memory
    
    def forget(self):
        self.memory = []
    
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

    start_time = time.time()

    # I love Tim Dettmers
    bnb_config = {'load_in_4bit':True,
                 'bnb_4bit_use_double_quant':True,
                 'bnb_4bit_quant_type':"nf4",
                 'bnb_4bit_compute_dtype': torch.bfloat16}

    model = AutoModelForCausalLM.from_pretrained(path, **bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(path)
    end_time = time.time()  # End the timer
    print(f"Model loading: {end_time - start_time} seconds")  
    return model, tokenizer

def llm_batch_generate(model, tokenizer, prompt, device, max_token, num_sequences):

    start_time = time.time()
    chat = [
        {"role": "user", "content": f"{prompt}"},
    ]
    model_inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    model_inputs = model_inputs.to(device)
    generated_ids = model.generate(
        model_inputs,
        max_new_tokens= max_token + 100,
        do_sample=True,
        temperature=1.5,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_sequences,  # Specify number of sequences to return
    )
    
    answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Split answers if the tokenizer includes special tokens that you wish to remove
    # Note: The line below assumes your special token is '[/INST]', adjust if different
    answers = [answer.split('[/INST]')[1] if '[/INST]' in answer else answer for answer in answers]

    # If only one sequence was generated, return a string instead of a list
    if len(answers) == 1:
        answers =  answers[0]

    end_time = time.time()  # End the timer
    print(f"Batch inference: {end_time - start_time} seconds")  
    
    return answers


