# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import json
import os

from datasets import load_dataset

dataset = load_dataset("huuuyeah/meetingbank", split="train")
data = []
for idx, instance in enumerate(dataset):
    temp = {}
    temp["idx"] = idx
    temp["prompt"] = instance["transcript"]
    temp["summary"] = instance["summary"]
    data.append(temp)
os.makedirs("../../../results/meetingbank/origin/", exist_ok=True)
json.dump(
    data,
    open("../../../results/meetingbank/origin/meetingbank_train_formated.json", "w"),
    indent=4,
)
