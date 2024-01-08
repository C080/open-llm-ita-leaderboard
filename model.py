from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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
    
    def remember(self):
        return self.memory[:-1]
    


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
    generated_ids = model.generate(model_inputs, max_new_tokens=500)
    return tokenizer.batch_decode(generated_ids)[0]

def prep_list_of_prompts(job_descriptions_prompt):
    print("Preparing list of prompts...")
    job_descriptions_prompt_list = job_descriptions_prompt.split("!STOP!")
    job_descriptions_prompt_list = [job_description.strip() for job_description in job_descriptions_prompt_list]
    print(f"Generated {len(job_descriptions_prompt_list)} prompt of job descriptions")
    return job_descriptions_prompt_list


def generate_father_prompt(row):
    print("-"*50)
    print(f"Generating job description prompt for the role of {row['preferredLabel']}...")
    meta_prompt = "Generate 3 different prompts for the creation of multiple job descriptions based on the following informations:\n"
    meta_prompt += f"Occupation: {row['preferredLabel']}\n"
    meta_prompt += f"Description: {row['description']}\n"
    meta_prompt += f"Alt labels: {row['altLabels']}\n"
    meta_prompt += "All the job descriptions must be different from each other. Separate each job description with the following word: !STOP!"
    print("-"*50)
    return meta_prompt