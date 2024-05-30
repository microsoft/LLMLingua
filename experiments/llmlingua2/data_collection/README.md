### Use our collected data

We release our collected GPT-4 compression result at [HF](https://huggingface.co/datasets/microsoft/MeetingBank-LLMCompressed) after review. To load data, simply use

```python
from datasets import load_dataset
data = load_dataset("microsoft/MeetingBank-LLMCompressed", split="train")
print(len(data))
for idx, sample in enumerate(data):
    # concatenation of all chunks
    prompt = sample["prompt"]
    compressed_prompt = sample["compressed_prompt"]
```
**prompt** is the original meeting transcript. **compressed_prompt** is the compression result after merging all compressed chunks of a transcript.

To load compressed chunks along with original chunks, simply use
```python
from datasets import load_dataset
data = load_dataset("microsoft/MeetingBank-LLMCompressed", split="train")
print(len(data))
for idx, sample in enumerate(data):
    # chunk list
    prompt_list = sample["prompt_list"]
    compressed_prompt_list = sample["compressed_prompt_list"]
```

### Construct your custom compression dataset

First, format your data to a list of dict, with each dict containing at least two keys: *idx* and *prompt*. [**format_data.py**](format_data.py) illustrates how we format the meetingbank data.

Then, instruct GPT-4 to compress the original context.

```bash
python compress.py --load_origin_from <your data path> \
--chunk_size 512 \
--compressor llmcomp \
--model_name gpt-4-32k \
--save_path <compressed data save path>

```

Then, assign label to the original words and filter out poor compression samples.


```bash
python label_word.py \
--load_prompt_from <compressed data save path> \
--window_size 400 \
--save_path <labeled data save path> \

```

Filter out some poorly compressed / labeled samples.
```bash
python filter.py --load_path <labeled data save path> \
--save_path <kept data save path>
```
