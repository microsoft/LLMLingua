# FastKV LongBench + LLMLingua-2 + Llama 3.1 8B Instruct

This setup runs LLMLingua-2 compression on the LongBench data stored in a sibling
FastKV checkout, then evaluates with `meta-llama/Llama-3.1-8B-Instruct` at
adaptive 20% and 30% prompt-token budgets.

Expected checkout layout:

```text
<workdir>/FastKV
<workdir>/LLMLingua
```

Required local data and prompt source:

```text
FastKV/data/LongBench/*.jsonl
FastKV/eval/run_longbench.py
```

Before running, authenticate to Hugging Face with an account that has access to
Llama 3.1 8B Instruct. Do not commit tokens.

```bash
huggingface-cli login
```

Run from either repository:

```bash
cd FastKV
bash scripts/run_llmlingua2_llama31_8b_20_30.sh
```

or

```bash
cd LLMLingua
bash experiments/llmlingua2/evaluation/scripts/run_fastkv_longbench_llama31_8b_20_30.sh
```

Useful overrides:

```bash
FASTKV_ROOT=/path/to/FastKV LLMLINGUA_REPO=/path/to/LLMLingua DATA_DIR=/path/to/LongBench SAVE_DIR=/path/to/results RATES="0.20 0.30" DATASETS="narrativeqa,qasper" MAX_NUM_EXAMPLES=10 bash scripts/run_llmlingua2_llama31_8b_20_30.sh
```

Default output:

```text
LLMLingua/results/fastkv_longbench_llmlingua2/llama3_1_8b_instruct/adaptive20
LLMLingua/results/fastkv_longbench_llmlingua2/llama3_1_8b_instruct/adaptive30
```

The adaptive budget is computed per sample as a percentage of the full downstream
prompt length after applying the FastKV LongBench prompt and chat template. The
runner subtracts prompt overhead and passes the remaining target token count to
LLMLingua-2 for context compression.
