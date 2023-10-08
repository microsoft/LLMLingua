# LLMLingua Documentation

## Principles

- The most important thing is **the sensitivity to compression varies among different components in a prompt**, such as instructions and questions being more sensitive, while context or documents are less sensitive. Therefore, it is advisable to separate the components within the prompt and input them into demonstrations, instructions, and questions.
- **Divide demonstrations and context into independent granularities**, such as documents in multi-document QA and examples in few-shot learning. This approach will be beneficial for the budget controller and document reordering.
- **Preserving essential characters in the scenario as required by the rule**, we will provide support soon.
- Try experimenting with different target compression ratios or other hyperparameters to optimize the performance.

## Initialization

```python
from llmlingua import PromptCompressor

llm_lingua = PromptCompressor(
    model_name: str = "NousResearch/Llama-2-7b-hf",
    device_map: str = "cuda",
    use_auth_token: bool = False,
    open_api_config: dict = {}, 
)
```
### Parameters

- model_name(str), the name of small language model from huggingface. Default set to "NousResearch/Llama-2-7b-hf";
- device_map(str), the device environment for using small models, like 'cuda', 'cpu', 'balanced', 'balanced_low_0', 'auto'. Default set to "cuda";
- use_auth_token(bool, optional), controls the usage of huggingface auto_token. Default set to False;
- open_api_config(dict, optional), the config of openai which use in OpenAI Embedding in coarse-level prompt compression. Default set to {};

## Function Call

```python
compressed_prompt = llm_lingua.compress_prompt(
    context: List[str],
    instruction: str = "",
    question: str = "",
    ratio: float = 0.5,
    target_token: float = -1,
    iterative_size: int = 200,
    force_context_ids: List[int] = None,
    force_context_number: int = None,
    use_sentence_level_filter: bool = False,
    use_context_level_filter: bool = True,
    use_token_level_filter: bool = True,
    keep_split: bool = False,
    keep_first_sentence: int = 0,
    keep_last_sentence: int = 0,
    keep_sentence_number: int = 0,
    high_priority_bonus: int = 100,
    context_budget: str = "+100",
    token_budget_ratio: float = 1.4,
    condition_in_question: str = "none",
    reorder_context: str = "original",
    dynamic_context_compression_ratio: float = 0.0,
    condition_compare: bool = False,
    add_instruction: bool = False,
    rank_method: str = "longllmlingua",
    concate_question: bool = True,
)

# > {'compressed_prompt': 'Question: Sam bought a dozen boxes, each with 30 highlighter pens inside, for $10 each box. He reanged five of boxes into packages of sixlters each and sold them $3 per. He sold the rest theters separately at the of three pens $2. How much did make in total, dollars?\nLets think step step\nSam bought 1 boxes x00 oflters.\nHe bought 12 * 300ters in total\nSam then took 5 boxes 6ters0ters.\nHe sold these boxes for 5 *5\nAfterelling these  boxes there were 3030 highlighters remaining.\nThese form 330 / 3 = 110 groups of three pens.\nHe sold each of these groups for $2 each, so made 110 * 2 = $220 from them.\nIn total, then, he earned $220 + $15 = $235.\nSince his original cost was $120, he earned $235 - $120 = $115 in profit.\nThe answer is 115',
#  'origin_tokens': 2365,
#  'compressed_tokens': 211,
#  'ratio': '11.2x',
#  'saving': ', Saving $0.1 in GPT-4.'}
```

### Parameters

- **context**(str or List[str]), the context, documents or demonstrations in the prompt, low sensitivity to compression;
- instruction(str), general instruction in the prompt before the context, high sensitivity to compression;
- **question**(str), general question in the prompt after the context, high sensitivity to compression;
- **ratio**(float, optional), target compression ratio, the larger the value, the fewer tokens will be retained, mutually exclusive with **target_token**, default set to 0.5;
- **target_token**(float), target compression token number, mutually exclusive with **ratio**, default set to -1;
- **iterative_size**(int), the segment size in Iterative Token-level Prompt Compression, default set to 200;
- **force_context_ids**(List[int], optional), the index list forcefully retains of **context**, default set to None,
- **force_context_number**(int, optional), the context number forcefully retains in Coarse-level Prompt Compression, default set to None,
- **use_sentence_level_filter**(bool, optional), controls the usage of the sentence-level prompt compression, default set to False;
- **use_context_level_filter**(bool, optional), controls the usage of the coarse-level prompt compression, default set to True;
- **use_token_level_filter**(bool, optional), controls the usage of the token-level prompt compression, default set to True;
- **keep_split**(bool, optional), control whether to retain all the newline separators "\n\n" in the prompt, default set to False;
- **keep_first_sentence**(bool, optional), control whether to retain the first k sentence in each context, default set to False;
- **keep_last_sentence**(bool, optional), control whether to retain the last k sentence in each context, default set to False;
- **keep_sentence_number**(int, optional), control the retain sentence number in each context, default set to 0;
- **high_priority_bonus**(int, optional), control the ppl bonus of the ratin sentence, only use when **keep_first_sentence** or **keep_last_sentence** is True, default set to 100;
- **context_budget**(str, optional), the budget in Coarse-level Prompt Compression, supported operators, like "*1.5" or "+100", default set to "+100";
- **token_budget_ratio**(float, optional), the budget ratio in sentence-level Prompt Compression, default set to 1.4;
- **condition_in_question**(str, optional), control whether use the question-aware coarse-level prompt compression, support "none", "after", "before". In the LongLLMLingua, it is necessary to set to "after" or "before", default set to "none";
- **reorder_context**(str, optional), control whether use the document reordering before compression in LongLLMLingua, support "original", "sort", "two_stage", default set to "original";
- **dynamic_context_compression_ratio**(float, optional), control the ratio of dynamic context compression in LongLLMLingua, default set to 0.0;
- **condition_compare**(bool, optional), control whether use the Iterative Token-level Question-aware Fine-Grained Compression in LongLLMLingua, default set to False,
- **add_instruction**(bool, optional), control whether add the instuct before prompt in Iterative Token-level Question-aware Fine-Grained Compression, default set to False;
- **rank_method**(bool, optional), control the rank method use in Coarse-level Prompt Compression, support "llmlingua", "longllmlingua", "bm25", "gzip", "sentbert", "openai", default set to "llmlingua";
- **concate_question**(bool, optional), control whether include the question in the compressed prompt, default set to True;

### Response

- **compressed_prompt**(str), the compressed prompt;
- **origin_tokens**(int), the token number of original prompt;
- **compressed_tokens**(int), the token number of compressed prompt;
- **ratio**(str), the actual compression ratio;
- **saving**(str), the saving cost in GPT-4.

## Post-precessing

```python
compressed_prompt = llm_lingua.recover(
    original_prompt: str,
    compressed_prompt: str,
    response: str,
)
```

### Parameters

- **original_prompt**(str), the original prompt;
- **compressed_prompt**(str), the compressed prompt;
- **response**(str), the response of the compressed prompt from black-box LLMs;

### Response

- **recovered_response**(str), the recovered response;