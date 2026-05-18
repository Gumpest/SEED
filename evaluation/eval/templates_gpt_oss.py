
def create_prompt_with_gpt_oss_format(messages, bos="<|start|>", eos="<|end|>", add_bos=True):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|start|>system<|message|>" + message["content"] + "<|end|>"
        elif message["role"] == "user":
            formatted_text += "<|start|>user<|message|>" + message["content"] + "<|end|>"
        elif message["role"] == "assistant":
            formatted_text += "<|start|>assistant<|message|>" + message["content"].strip() + "<|end|>"
        else:
            raise ValueError(f"Unsupported role: {message['role']}")
    # Force final channel
    formatted_text += "<|start|>assistant<|channel|>final<|message|>"
    return formatted_text
