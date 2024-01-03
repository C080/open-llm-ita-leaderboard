
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

path = "E:/text-generation-webui-main/models/saiga-7b"
model_params = {
    'low_cpu_mem_usage': True,
    'torch_dtype': torch.bfloat16,
    'use_flash_attention_2': False,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(path, **model_params)
model = model.cuda()
tokenizer = AutoTokenizer.from_pretrained(path)

prompt = "My favourite condiment is"

model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
model.to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
tokenizer.batch_decode(generated_ids)[0]

