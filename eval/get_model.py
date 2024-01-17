import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, PeftConfig


def get_model(model_name):
    if model_name == "mistral-7b-fauno":

        model_path = "/home/m.cimino/federico/mistral-finetune2/out2/checkpoint-582/"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, 
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return model, tokenizer
    
    if model_name == "llamantino":

        model = AutoModelForCausalLM.from_pretrained(
            "swap-uniba/LLaMAntino-2-chat-7b-hf-UltraChat-ITA",
            torch_dtype=torch.bfloat16, 
            load_in_8bit=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "swap-uniba/LLaMAntino-2-chat-7b-hf-UltraChat-ITA",
        )
        return model, tokenizer
        
    
    if model_name == "saiga-7b":

        model_path = "E:/text-generation-webui-main/models/saiga-7b"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, 
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return model, tokenizer

    if model_name == "mistral-7b-onlygen":

        model_path = "/home/m.cimino/federico/mistral-finetune2/out-final-onlygen/checkpoint-419"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, 
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return model, tokenizer


    elif model_name == "fauno":
        
        model = LlamaForCausalLM.from_pretrained(
            "baffo32/decapoda-research-llama-7B-hf",
            torch_dtype=torch.bfloat16, 
        )
        tokenizer = LlamaTokenizer.from_pretrained(
            "baffo32/decapoda-research-llama-7B-hf",
            add_eos_token=True,
            padding_side="left",
        )
        
        model = PeftModel.from_pretrained(model, "/home/m.cimino/federico/Fauno-Italian-LLM-7B")

        return model, tokenizer
    
    elif model_name == "camoscio":
        
        model = LlamaForCausalLM.from_pretrained(
            "baffo32/decapoda-research-llama-7B-hf",
            torch_dtype=torch.bfloat16, 
        )
        tokenizer = LlamaTokenizer.from_pretrained(
            "baffo32/decapoda-research-llama-7B-hf",
            add_eos_token=True,
            padding_side="left",
        )
        
        model = PeftModel.from_pretrained(model, "/home/m.cimino/federico/camoscio-7b-llama")

        return model, tokenizer
    
    elif model_name == "mistral-7b":
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            torch_dtype=torch.bfloat16, 
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            add_eos_token=True,
            padding_side="left",
        )

        return model, tokenizer
    
    elif model_name == "cerbero":
        #model_path = "/home/m.cimino/federico/mistral-finetune2/out-final-test"
        model_path = "/home/m.cimino/federico/mistral-finetune2/out-final-full/checkpoint-920"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, 
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return model, tokenizer

    elif model_name == "cerbero-openchat":
        #model_path = "/home/m.cimino/federico/mistral-finetune2/out-final-test"
        model_path = "/home/m.cimino/federico/mistral-finetune2/out-final-full-openchat/checkpoint-920"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, 
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return model, tokenizer

    else:
        raise ValueError("Model not found")

    

if __name__ == "__main__":
    models = ["llamantino"]
    for model_name in models:
        print(f"Getting model {model_name}")
        model, tokenizer = get_model(model_name)
        print("===")
