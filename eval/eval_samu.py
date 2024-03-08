import os
import time
os.environ["HF_HOME"] = "E:/huggingface"
import torch
import random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate

# Define paths and model name
HF_HOME = "E:/huggingface"
CACHE_DIR = "./cache"
MODEL_NAME = "DeepMount00/Gemma_QA_ITA"

# Set up the environment
os.environ["HF_HOME"] = HF_HOME
os.makedirs(CACHE_DIR, exist_ok=True)

# Load and preprocess the dataset
squad_it_train = load_dataset("squad_it", split='train', cache_dir=CACHE_DIR, trust_remote_code=True)
squad_it_test = load_dataset("squad_it", split='test', cache_dir=CACHE_DIR, trust_remote_code=True)  

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

SAMPLES_TO_EVALUATE  = 10

# Function to prepare three-shot prompt
def prepare_three_shot_prompt(train_examples, context, question):
    shots = random.sample(train_examples, 3)
    prompt = ""
    for shot in shots:
        prompt += f"contesto: {shot['context']}\ndomanda: {shot['question']}\nrisposta\n {shot['answers']['text'][0]}\n\n"
    prompt += f"contesto: {context}\ndomanda: {question}\nrisposta\n"
    return prompt

# Function to evaluate on test set and save results
def evaluate_on_test_set(test_dataset, train_dataset, model, SAMPLES_TO_EVALUATE):
    predicted_answers = {}
    theoretical_answers = {}
    train_examples = [example for example in train_dataset]
    max_length = model.config.max_position_embeddings
    
    for i, example in enumerate(test_dataset.select([x for x in range(0,SAMPLES_TO_EVALUATE,1)])):
        context = example['context']
        question = example['question']
        true_answer = example['answers']['text'][0]
        
        prompt = prepare_three_shot_prompt(train_examples, context, question)
        
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
        output_max_length = inputs.input_ids.size(1) + 50
        if output_max_length > max_length:
            output_max_length = max_length
        
        output = model.generate(**inputs, max_length=output_max_length, eos_token_id = 107)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        #answer_start_idx = generated_text.rfind("risposta:") + len("risposta:")
        generated_answer = generated_text[len(prompt):].strip()

        # Save results to dictionaries
        predicted_answers[str(i)] = generated_answer
        theoretical_answers[str(i)] = {"answers": [{"text": true_answer, "answer_start": example['answers']['answer_start'][0]}]}

        # Prepare predictions and references for metric computation
        references = [{"id": key , "answers": value["answers"]} for key, value in theoretical_answers.items()]
        predictions = [{"id": key, "prediction_text": value} for key, value in predicted_answers.items()]

        # Compute the metric
        metric = evaluate.load("squad")
        results = metric.compute(predictions=predictions, references=references)

    return results

    # Evaluate the model and print the report
results = evaluate_on_test_set(squad_it_test, squad_it_train, model, SAMPLES_TO_EVALUATE)
current_date = time.strftime("%d/%m/%Y %H:%M:%S")
report = f"""=== REPORT ===
Current date: {current_date}
Dataset: SQuAD-it
Model: {MODEL_NAME}
{results}
==========================
"""
print(report)