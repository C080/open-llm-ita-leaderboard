from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

def LoRA_extract(lora):
    A = {}
    B = {}
    for n, w in lora.items():
        l = n.split('.')
        layer = '.'.join(l[2:7]+[l[-1]])
        if l[7] == 'lora_A':
            A[layer] = w
        elif l[7] == 'lora_B':
            B[layer] = w
    return A, B

lora_path = "E:/text-generation-webui-main/loras/dante_saiga_1/"
path = "E:/text-generation-webui-main/models/saiga-7b"

model_params = {
    'low_cpu_mem_usage': True,
    'torch_dtype': torch.bfloat16,
    'use_flash_attention_2': False,
    # 'load_in_8bit': True
}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(path, **model_params)
model = model.cuda()
lora_model = torch.load(lora_path + 'adapter_model.bin')
tokenizer = AutoTokenizer.from_pretrained(path)

with open(lora_path + "adapter_config.json","r") as f:
    lora_config = json.load(f)
alpha = lora_config['lora_alpha']/lora_config['r']
A, B = LoRA_extract(lora_model)

original_weights = dict(model.named_parameters())
for name, param in original_weights.items():
    if name in A.keys() and name in B.keys():
        print(f'Found layer:\n{name}\n***\n')
        param.data.add_(alpha * B[name] @ A[name]) # a causa del modo in cui PyTorch gestisce i tensori se cambio l'iteratore cambia anche il dict

model.save_pretrained('saiga-7b-dante-qlora')

# prompt = "Traduci in volgare fiorentino: \"Mamma mi sono fatto la cacca addosso\""

# model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
# model.to(device)

# generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
# print(tokenizer.batch_decode(generated_ids)[0])