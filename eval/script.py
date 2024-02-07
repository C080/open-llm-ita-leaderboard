import os
os.environ["HF_HOME"] = "E:/huggingface"
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import evaluate
import time

# Load dataset and model
dataset = load_dataset("squad_it")
model_name = "maestrale"
model_path = "mii-llm/maestrale-chat-v0.3-alpha"

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bnb_config = {'load_in_4bit':True, 'bnb_4bit_use_double_quant':True, 'bnb_4bit_quant_type':"nf4", 'bnb_4bit_compute_dtype': torch.bfloat16}

model = AutoModelForCausalLM.from_pretrained(model_path, **bnb_config)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Batch Size
bs = 8

# FUNCTIONS
def build_question(context, question):
    return f"Dato il seguente testo:\n{context}\nRispondi brevemente a questa domanda:\n{question}"

def build_answer(answer):
    return f"Risposta breve: {answer}"
    
def forward_model(*, prompt, model, tokenizer):
    if "pad_token" not in tokenizer.special_tokens_map:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer(prompt, return_tensors='pt', padding=True).input_ids.to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids, 
            max_new_tokens=32, 
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
            prompt_lookup_num_tokens=10, 
            temperature=0.7, 
            do_sample=True, 
            top_k=50, 
            top_p=0.95, 
            repetition_penalty=1.2
        )
    output_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    return [text[len(p):].strip() for text, p in zip(output_texts, prompt)]


# MAIN PART
ds = dataset["test"].shuffle(seed=42)
predicted_answers = []
theoretical_answers = []

bar = tqdm(range(0, len(ds), bs), total=len(ds)/bs)
for i in bar:
    batch = ds[i:i+bs]
    ids = batch["id"]
    contexts = batch["context"]
    questions = batch["question"]
    batch_answers = batch["answers"]

    shots_text = [build_question(s["context"], s["question"]) + tokenizer.eos_token + build_answer(s["answers"]["text"][0]) for s in ds.select(range(3))]
    prompts = [ " ".join(shots_text) + build_question(c, q) for c, q in zip(contexts, questions)]
    model_outputs = forward_model(prompt=prompts, model=model, tokenizer=tokenizer)
    predicted_answers.extend([{"id":id, "prediction_text": pred} for id, pred in zip(ids, model_outputs)])
    theoretical_answers.extend([{"id":id, "answers": ans} for id, ans in zip(ids, batch_answers)])

# Save the results
os.makedirs('./cache', exist_ok=True)
with open(f"./cache/generated-squad-{model_name}.json", "w") as f:
    json.dump({"predicted_answers":predicted_answers, "theoretical_answers":theoretical_answers}, f, indent=4)

# Evaluate the results
metric = evaluate.load("squad")
results = metric.compute(predictions=predicted_answers, references=theoretical_answers)

# Print final report
print("=== REPORT ===")
print("current date:", time.strftime("%d/%m/%Y %H:%M:%S"))
print("Dataset: SQuAD-it")
print("Model:", model_name)
print(results)
print("==========================")