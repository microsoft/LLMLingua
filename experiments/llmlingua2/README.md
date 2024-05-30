# LLMLingua-2 Experiments

## Getting Started

To get started with LLMLingua-2 experiments, simply install it using pip:

```bash
pip install llmlingua
```

To collect your own data using GPT-4, install the following packages:
```bash
pip install openai==0.28

pip install spacy
python -m spacy download en_core_web_sm
```

To train your own compressor on the collected data, install:
```bash
pip install scikit-learn
pip install tensorboard
```

## Data collection

We release our collected GPT-4 compression result at [HF](https://huggingface.co/datasets/microsoft/MeetingBank-LLMCompressed) after review. We also provide the whole data collection pipeline at [**collect_data.sh**](data_collection/collect_data.sh) to help you construct your custom compression dataset.

## Model Training

To train a compressor on the collected data, simply run [**train.sh**](model_training/train.sh)

## Evaluation

We provide a script [**compress.sh**](evaluation/scripts/compress.sh) to compress the original context on several benchmarks. After compression, run [**evaluate.sh**](evaluation/scripts/evaluate.sh) to evalate on down-stream task using the compressed prompt.
