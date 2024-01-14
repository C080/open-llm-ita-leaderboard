
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
    else:
        raise ValueError("Model not found")

if __name__ == "__main__":
    models = ["cerbero"]
    for model_name in models:
        print(f"Getting model {model_name}")
        prompt, stop = get_prompt(model_name)
        print(prompt([
            dict(
                role="user",
                text="Fai qualcosa"
            ),
            dict(
                role="ai",
                text="Qualcosa fatta"
            ),
            dict(
                role="user",
                text="Ora fai qualcos'altro"
            )
        ], do_continue=True))
        print("stop:"+stop)
        print("===")