# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import json
import re
import yaml


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
