
def llamantino_prompt(conversation, do_continue=False):
    assert conversation[-1]['role'] == "user"
    ST, ET = "<s>", "</s>" #Start/End Text
    SUM, EUM = "[INST]", "[/INST]" #Start/End User Message
    prompt = ""
    for message in conversation:
        if message['role'] == "user":
            prompt += f"{ST}{SUM}{message['text']}{EUM}"
        elif message['role'] == "assistant":
            prompt += f"{message['text']}{ET}"
        else:
            raise ValueError(f"Role not found,\nFounded {message['role']}")
    

    if do_continue:
        prompt += "\n" # non mi convince
    
    return prompt

def loquace_prompt(conversation, do_continue=False):
    assert conversation[-1]['role'] == "user"
    ST, ET = "", "" #Start/End Text
    SUM, EUM = "", "" #Start/End User Message
    prompt = ""
    for message in conversation:
        if message['role'] == "user":
            prompt += f"{ST}{SUM}### Instruction: {message['text']}{EUM}"
        elif message['role'] == "assistant":
            prompt += f"### Response: {message['text']}{ET}"
        else:
            raise ValueError(f"Role not found,\nFounded {message['role']}")
    

    if do_continue:
        prompt += "\n" # non mi convince
    
    return prompt

# llamantino prompt template
# """
# <s>[INST] <<SYS>>
# {your_system_message}
# <</SYS>>

# {user_message_1} [/INST] {model_reply_1}</s><s>[INST] {user_message_2} [/INST]
# """


def mistral_ita_prompt0(conversation, do_continue=False):
    prompt = ""
    B_INST, E_INST = "<s>[INST] ", " [/INST]"
    for i, message in enumerate(conversation):
        if message["role"] == "user":
            if i != len(conversation) - 1:
                prompt += f"{B_INST}{message['text']}{E_INST}\n{conversation[i + 1]['text']}</s>\n"
            else:
                prompt += f"{B_INST}{message['text']}{E_INST}</s>"
        elif message["role"] == "assistant":
            continue
        else:
            raise ValueError("Role not found")
    assert conversation[-1]["role"] == "user"

    if do_continue:
        prompt += "<s>" 
    
    return prompt

def zefiro_prompt(conversation, do_continue=False):

    prompt = ""
    for message in conversation:
        if message['role'] == 'user':
            prompt += f"<|user|>\n{message['text']}</s>\n"
        elif message['role'] == 'assistant':
            prompt += f"<|assistant|>\n{message['text']}</s>\n"
    # if do_continue:
    #     prompt += "<s>" 
    
    return prompt

    ### NEW PROMPT 000
def mistral_ita_prompt(conversation, do_continue=False):
    B_INST, E_INST = "[INST] ", " [/INST]"
    prompt = "<s>"
    for message in conversation:
        if message['role'] == 'user':
            prompt += f"{B_INST}{message['text']}{E_INST} "
        elif message['role'] == 'assistant':
            prompt += f"{message['text']}</s>"
    # if do_continue:
    #     prompt += "<s>" 
    
    return prompt


def maestrale_prompt(conversation, do_continue=False):
    B_INST, E_INST = "<|im_start|>", "<|im_end|>"
    prompt = ""
    for message in conversation:
        if message['role'] == 'user':
            prompt += f"{B_INST}user\n{message['text']}{E_INST}\n"
        elif message['role'] == 'assistant':
            prompt += f"{B_INST}assistant\n{message['text']}{E_INST}\n"
    # if do_continue:
    #     prompt += "<s>" 
    
    return prompt


def fauno_prompt(conversation, do_continue=False):
    prompt = ""
    for message in conversation:
        if message["role"] == "user":
            prompt += f"[|Umano|] {message['text']}\n"
        elif message["role"] == "ai":
            prompt += f"[|AI|] {message['text']}\n"
        else:
            raise ValueError("Role not found")
    assert conversation[-1]["role"] == "user"
    
    if do_continue:
        prompt += "[|AI|]"

    return prompt

def cerbero_prompt(conversation, do_continue=False):
    prompt = ""
    for message in conversation:
        if message["role"] == "user":
            prompt += f"[|Umano|] {message['text']}\n"
        elif message["role"] == "ai":
            prompt += f"[|Assistente|] {message['text']}\n"
        else:
            raise ValueError("Role not found")
    assert conversation[-1]["role"] == "user"
    
    if do_continue:
        prompt += "[|Assistente|]"

    return prompt


def camoscio_prompt(conversation, do_continue=False):
    def generate_prompt(instruction, input=None):
        if input:
            return f"""Di seguito è riportata un'istruzione che descrive un task, insieme ad un input che fornisce un contesto più ampio. Scrivete una risposta che completi adeguatamente la richiesta.

### Istruzione:
{instruction}

### Input:
{input}

### Risposta:"""
        else:
            return f"""Di seguito è riportata un'istruzione che descrive un task. Scrivete una risposta che completi adeguatamente la richiesta.

### Istruzione:
{instruction}

### Risposta:"""
        
    prompt = ""
    for message in conversation:
        if message["role"] == "user":
            prompt += generate_prompt(message["text"])
        elif message["role"] == "ai":
            prompt += f"\n{message['text']}\n\n"
        else:
            raise ValueError("Role not found")
    assert conversation[-1]["role"] == "user"
    
    if do_continue:
        prompt += "\n"

    return prompt



def get_prompt(model_name):
    if model_name == "mistral-7b-fauno":
        return fauno_prompt, "[|Umano|]"
    elif model_name == "fauno":
        return fauno_prompt, "[|Umano|]"
    elif model_name == "camoscio":
        return camoscio_prompt, "\n"
    elif model_name == "mistral-7b":
        return fauno_prompt, "\n"
    elif model_name == "cerbero":
        return cerbero_prompt, "[|Umano|]"
    elif model_name == "mistral-7b-onlygen":
        return cerbero_prompt, "\n"
    elif model_name == "cerbero-openchat":
        return cerbero_prompt, "[|Umano|]"
    elif model_name == "saiga-7b":
        return cerbero_prompt, "[|Umano|]"
    elif model_name == 'llamantino':
        return llamantino_prompt, "[\INST]"
    elif model_name == 'mistral-ita-7b':
        return mistral_ita_prompt, "[\INST]"
    elif model_name == 'zefiro':
        return zefiro_prompt, "[\INST]"
    elif model_name == 'loquace':
        return loquace_prompt, "[\INST]"
    elif model_name == 'maestrale':
        return maestrale_prompt, "[\INST]"
    else:
        raise ValueError("Model not found")

if __name__ == "__main__":
    models = ["maestrale"]
    for model_name in models:
        print(f"Getting model {model_name}")
        prompt, stop = get_prompt(model_name)
        print(prompt([
            dict(
                role="user",
                text="SHOT 1"
            ),
            dict(
                role="assistant",
                text="ANSWER 1"
            ),
             dict(
                role="user",
                text="SHOT 2"
            ),
            dict(
                role="assistant",
                text="ANSWER 2"
            ),
             dict(
                role="user",
                text="SHOT 3"
            ),
            dict(
                role="assistant",
                text="ANSWER 3"
            ),
            dict(
                role="user",
                text="DOMANDA"
            )
        ], do_continue=False))
        #print("stop:"+stop)
        print("===")