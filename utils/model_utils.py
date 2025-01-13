import numpy as np
from typing import List


def chunk_as_size(lst, chunk_lst):
    sizes = [len(x) for x in chunk_lst]
    assert sum(sizes) == len(lst)

    rnt = []
    i = 0
    for s in sizes:
        rnt.append(lst[i : i + s])
        i += s

    return rnt


def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def chunks_by_num(lst, n):
    rnt = np.array_split(lst, n)
    return [x.tolist() for x in rnt]


def flatten_list(lst):
    new_lst = []
    for item in lst:
        new_lst += item
    return new_lst


# llama2-chat
def llama2chat_format_chat_prompt(message: str):
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe. "
        "Please ensure that your responses are socially unbiased and positive in nature. "
        "If a question does not make any sense, or is not factually coherent, explain "
        "why instead of answering something not correct. If you don't know the answer "
        "to a question, please don't share false information."
    )
    lines = [
        "<s>[INST] <<SYS>>",
        DEFAULT_SYSTEM_PROMPT,
        "<</SYS>>",
        "",
        f"{message} [/INST]",
    ]
    return "\n".join(lines)


# kv, nq + longchat
def longchat_format_chat_prompt(input):
    from fastchat.model import get_conversation_template

    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], input + '\n')
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt


def wrap_prompts(prompts: List[str], model_name: str):
    if "llama" in model_name and "chat" in model_name:
        wrapper = llama2chat_format_chat_prompt
    elif "longchat" in model_name:
        wrapper = longchat_format_chat_prompt
    else:
        return prompts

    wrapped_prompts = [
        wrapper(pm) for pm in prompts
    ]
    return wrapped_prompts
