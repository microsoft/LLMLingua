import json
import os
import random
import re
import string
import time

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from azure.identity import ManagedIdentityCredential, get_bearer_token_provider 
from openai import AzureOpenAI 


# Default API configuration - can be overridden by user
DEFAULT_API_CONFIG = {
    "scope": "https://cognitiveservices.azure.com/.default",
    "client_id": "YOUR_CLIENT_ID_HERE",  # Replace with your client ID
    "api_details": {
        "primary": {
            "api_base": "YOUR_PRIMARY_API_BASE_HERE",  # Replace with your primary API base
            "api_version": "2024-02-01",
        },
        "secondary": {
            "api_base": "YOUR_SECONDARY_API_BASE_HERE",  # Replace with your secondary API base
            "api_version": "2024-02-01",
        }
    }
}

# Global variables for API configuration
api_config = DEFAULT_API_CONFIG.copy()
token_provider = None
client = None
api_base = None

def initialize_api_config(custom_config=None):
    """Initialize API configuration with custom settings or defaults"""
    global api_config, token_provider, client, api_base
    
    if custom_config:
        api_config.update(custom_config)
    
    # Initialize token provider
    token_provider = get_bearer_token_provider(
        ManagedIdentityCredential(client_id=api_config["client_id"]), 
        api_config["scope"]
    )
    
    # Initialize client with default endpoint
    client = AzureOpenAI( 
        api_version=api_config["api_details"]["primary"]["api_version"], 
        azure_endpoint=api_config["api_details"]["primary"]["api_base"], 
        azure_ad_token_provider=token_provider 
    )
    
    api_base = api_config["api_details"]["primary"]["api_base"]

# Initialize with default config
initialize_api_config()

def query_llm(
    prompt,
    model_name,
    max_tokens,
    **kwargs,
):
    SLEEP_TIME_FAILED = 1
    global api_base, client, api_config

    request = {
        "temperature": kwargs["temperature"] if "temperature" in kwargs else 0.0,
        "top_p": kwargs["top_p"] if "top_p" in kwargs else 1.0,
        "seed": kwargs["seed"] if "seed" in kwargs else 42,
        "max_tokens": max_tokens,
        "n": 1,
        "stream": False,
    }
    request["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    answer = None
    response = None
    while answer is None:
        try:
            response = client.chat.completions.create(model=model_name, **request)
            answer = response.choices[0].message.content
            
        except Exception as e:
            print(f"error: {e}, response: {response}")
            answer = None
            if "content management" in str(e):
                print(f"error: {e}, response: {response}")
                print("returning None")
                break
            elif "content" in str(e):
                print(f"error: {e}, response: {response}")
                print("returning None")
                break
            elif "filtered" in str(response):
                print(f"error: {e}, response: {response}")
                print("returning None")
                break
            elif "repetitive patterns" in str(e):
                print(f"error: {e}, response: {response}")
                print("returning None")
                break
            elif "exceeded token rate limit" in str(e) or "exceeded call rate limit" in str(e) or "rate limit" in str(e):
                # change api details
                # print(f"error: {e}, response: {response}")
                print("changing api details")
                if api_base == api_config["api_details"]["primary"]["api_base"]:
                    api_base = api_config["api_details"]["secondary"]["api_base"]
                    client = AzureOpenAI( 
                        api_version=api_config["api_details"]["secondary"]["api_version"], 
                        azure_endpoint=api_config["api_details"]["secondary"]["api_base"], 
                        azure_ad_token_provider=token_provider 
                    )
                else:
                    api_base = api_config["api_details"]["primary"]["api_base"]
                    client = AzureOpenAI( 
                        api_version=api_config["api_details"]["primary"]["api_version"], 
                        azure_endpoint=api_config["api_details"]["primary"]["api_base"], 
                        azure_ad_token_provider=token_provider 
                    )
            time.sleep(SLEEP_TIME_FAILED)
    # sleep(SLEEP_TIME_SUCCESS)
    return answer


class TokenClfDataset(Dataset):
    def __init__(
        self,
        texts,
        max_len=512,
        tokenizer=None,
        model_name="bert-base-multilingual-cased",
    ):
        self.len = len(texts)
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_name = model_name
        if "bert-base-multilingual-cased" in model_name:
            self.cls_token = "[CLS]"
            self.sep_token = "[SEP]"
            self.unk_token = "[UNK]"
            self.pad_token = "[PAD]"
            self.mask_token = "[MASK]"
        elif "xlm-roberta-large" in model_name:
            self.bos_token = "<s>"
            self.eos_token = "</s>"
            self.sep_token = "</s>"
            self.cls_token = "<s>"
            self.unk_token = "<unk>"
            self.pad_token = "<pad>"
            self.mask_token = "<mask>"
        else:
            raise NotImplementedError()

    def __getitem__(self, index):
        text = self.texts[index]
        tokenized_text = self.tokenizer.tokenize(text)

        tokenized_text = (
            [self.cls_token] + tokenized_text + [self.sep_token]
        )  # add special tokens

        if len(tokenized_text) > self.max_len:
            tokenized_text = tokenized_text[: self.max_len]
        else:
            tokenized_text = tokenized_text + [
                self.pad_token for _ in range(self.max_len - len(tokenized_text))
            ]

        attn_mask = [1 if tok != self.pad_token else 0 for tok in tokenized_text]

        ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(attn_mask, dtype=torch.long),
        }

    def __len__(self):
        return self.len


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_begin_of_new_word(token, model_name, force_tokens, token_map):
    if "bert-base-multilingual-cased" in model_name \
            or "tinybert" in model_name.lower() \
            or "mobilebert" in model_name.lower():
        if token.lstrip("##") in force_tokens or token.lstrip("##") in set(
            token_map.values()
        ):
            return True
        return not token.startswith("##")
    elif "xlm-roberta-large" in model_name:
        if (
            token in string.punctuation
            or token in force_tokens
            or token in set(token_map.values())
        ):
            return True
        return token.startswith("▁")
    else:
        raise NotImplementedError()


def replace_added_token(token, token_map):
    for ori_token, new_token in token_map.items():
        token = token.replace(new_token, ori_token)
    return token


def get_pure_token(token, model_name):
    if "bert-base-multilingual-cased" in model_name \
            or "tinybert" in model_name.lower() \
            or "mobilebert" in model_name.lower():
        return token.lstrip("##")
    elif "xlm-roberta-large" in model_name:
        return token.lstrip("▁")
    else:
        raise NotImplementedError()


def process_structured_json_data(json_data, json_config):
    if isinstance(json_config, str):
        with open(json_config, "r") as file:
            json_config = yaml.safe_load(file)
    elif not isinstance(json_config, dict):
        raise ValueError(
            "Invalid json config file. It should be a dictionary or a path to a yaml file."
        )
    assert set(json_data.keys()) == set(
        json_config.keys()
    ), "Keys in json data and json config file do not match."
    context = ["<llmlingua, compress=False>{</llmlingua>"]
    forced_context_ids = [0]
    for i, (k, v) in enumerate(json_data.items()):
        if not json_config[k]["pair_remove"]:
            forced_context_ids.append(i + 1)
        rate, compress, value_type = (
            json_config[k]["rate"],
            json_config[k]["compress"],
            json_config[k]["value_type"],
        )
        if not compress:
            rate = 1
        context.append(precess_jsonKVpair(k, v, value_type, rate))
    context[-1] = context[-1][:-14] + "</llmlingua>"
    context.append("<llmlingua, compress=False>}</llmlingua>")
    forced_context_ids.append(len(json_data) + 1)

    return context, forced_context_ids


def precess_jsonKVpair(k, v, value_type, rate):
    if rate == 1:
        return (
            "<llmlingua, compress=False>"
            + f"{json.dumps({k:v})[1:-1]}, "
            + "</llmlingua>"
        )
    if value_type == "str" or value_type == "string":
        v = str(v)
        new_v = (
            f"</llmlingua><llmlingua, rate={rate}>"
            + v
            + "</llmlingua><llmlingua, compress=False>"
        )
        return (
            "<llmlingua, compress=False>"
            + f"{json.dumps({k:new_v})[1:-1]}, "
            + "</llmlingua>"
        )
    elif value_type in ["int", "float", "integer", "number"]:
        if value_type in ["int", "integer"]:
            v = int(v)
        if value_type in ["float", "number"]:
            v = float(v)
        return (
            "<llmlingua, compress=False>"
            + f'"{k}": </llmlingua><llmlingua, rate={rate}>{v}</llmlingua><llmlingua, compress=False>, </llmlingua>'
        )
    elif value_type == "bool" or value_type == "boolean":
        if v in ["True", "true", "TRUE", True]:
            v = "true"
        elif v in ["False", "false", "FALSE", False]:
            v = "false"
        else:
            raise ValueError(f"Invalid boolean value: {v}")
        new_v = (
            f"</llmlingua><llmlingua, rate={rate}>"
            + v
            + "</llmlingua><llmlingua, compress=False>"
        )
        return (
            "<llmlingua, compress=False>"
            + f"{json.dumps({k:new_v})[1:-1]}, "
            + "</llmlingua>"
        )
    elif value_type == "list" or value_type == "List":
        return (
            "<llmlingua, compress=False>"
            + f'"{k}": {process_sequence_data(rate, "[", "]", v)}'
        )
    elif value_type == "dict" or value_type == "dictionary":
        return (
            "<llmlingua, compress=False>"
            + f'"{k}": {process_sequence_data(rate, "[", "]", v, is_dict=True)}'
        )
    elif value_type == "set":
        raise ValueError(f"Invalid value type: {value_type}")
        # return '<llmlingua, compress=False>' + f'"{k}": {process_sequence_data(rate, "{", "}", v)}'
    elif value_type == "tuple":
        return (
            "<llmlingua, compress=False>"
            + f'"{k}": {process_sequence_data(rate, "(", ")", v)}'
        )
    else:
        raise ValueError(f"Invalid value type: {value_type}")


def process_sequence_data(rate, start, end, sequence, is_dict=False):
    res = f'{start}"'
    n = len(sequence)
    if not is_dict:
        for i, item in enumerate(sequence):
            item = str(item)
            res += f"</llmlingua><llmlingua, rate={rate}>{item}</llmlingua><llmlingua, compress=False>"
            if i != n - 1:
                res += '", "'
    else:
        for i, (k, v) in enumerate(sequence.items()):
            item = f"{k}: {v}"
            item.replace('"', "'")
            res += f"</llmlingua><llmlingua, rate={rate}>{item}</llmlingua><llmlingua, compress=False>"
            if i != n - 1:
                res += '", "'
    res += f'"{end}, </llmlingua>'
    return res


def remove_consecutive_commas(text):
    text = re.sub(r",\s*", ",", text)
    text = re.sub(r",+", ",", text)
    return text