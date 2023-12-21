# (Long)LLMLingua Documentation

## Principles

1. **Prompt Sensitivity**: Different components of a prompt, like instructions and questions, vary in sensitivity to compression. Contexts or documents, for example, are less sensitive. It's advisable to separate these components in the prompt for demonstrations, instructions, and questions.
2. **Granular Division**: For multi-document QA and few-shot learning, divide demonstrations and contexts into independent granularities. This helps with budget control and document reordering.
3. **Essential Character Preservation**: Preserve essential characters as required by the scenario rules. Support for this feature is forthcoming.
4. **Optimization through Experimentation**: Experiment with various target compression ratios and other hyperparameters to optimize performance.

## Initialization

Initialize (Long)LLMLingua with the following parameters:

```python
from llmlingua import PromptCompressor

llm_lingua = PromptCompressor(
    model_name="NousResearch/Llama-2-7b-hf",  # Default model
    device_map="cuda",  # Device environment (e.g., 'cuda', 'cpu', 'mps')
    model_config={},  # Configuration for the Huggingface model
    open_api_config={},  # Configuration for OpenAI Embedding
)
```

### Parameters

- **model_name** (str): Name of the small language model from Huggingface. Defaults to "NousResearch/Llama-2-7b-hf".
- **device_map** (str): The computing environment. Options include 'cuda', 'cpu', 'mps', 'balanced', 'balanced_low_0', 'auto'. Default is 'cuda'.
- **model_config** (dict, optional): Configuration for the Huggingface model. Defaults to {}.
- **open_api_config** (dict, optional): Configuration for OpenAI Embedding in coarse-level prompt compression. Defaults to {}.

## Function Call

Utilize (Long)LLMLingua for prompt compression with a range of customizable parameters:

```python
compressed_prompt = llm_lingua.compress_prompt(
    context: List[str],  # Context or documents (low compression sensitivity)
    instruction: str = "",  # Instruction (high compression sensitivity)
    question: str = "",  # Question (high compression sensitivity)
    ratio: float = 0.5,  # Target compression ratio (default: 0.5)
    target_token: float = -1,  # Target compression token count (default: -1)
    # Additional parameters to fine-tune the compression
    iterative_size: int = 200,  # Segment size for iterative token-level compression
    force_context_ids: List[int] = None,  # Forced retention of specific context indices
    force_context_number: int = None,  # Forced retention of a specific number of contexts
    use_sentence_level_filter: bool = False,  # Enables sentence-level compression
    use_context_level_filter: bool = True,  # Enables context-level compression
    use_token_level_filter: bool = True,  # Enables token-level compression
    keep_split: bool = False,  # Retains newline separators in the prompt
    keep_first_sentence: int = 0,  # Retains the first 'k' sentences in each context
    keep_last_sentence: int = 0,  # Retains the last 'k' sentences in each context
    keep_sentence_number: int = 0,  # Retains a specific number of sentences in each context
    high_priority_bonus: int = 100,  # Priority bonus for ranking sentences
    context_budget: str = "+100",  # Budget for context-level compression
    token_budget_ratio: float = 1.4,  # Budget ratio for sentence-level compression
    condition_in_question: str = "none",  # Enables question-aware compression
    reorder_context: str = "original",  # Method for reordering context before compression
    dynamic_context_compression_ratio: float = 0.0,  # Ratio for dynamic context compression
    condition_compare: bool = False,  # Enables iterative token-level question-aware fine-grained compression
    add_instruction: bool = False,  # Adds instruction before the prompt
    rank_method: str = "longllmlingua",  # Method for ranking in coarse-level compression
    concate_question: bool = True,  # Includes the question in the compressed prompt
)
```


### Parameters

- **context** (str or List[str]): Contexts, documents, or demonstrations in the prompt, exhibiting low sensitivity to compression.
- **instruction** (str): General instruction within the prompt, displaying high sensitivity to compression.
- **question** (str): General question within the prompt, also highly sensitive to compression.
- **ratio** (float, optional): The target compression ratio, where larger values result in fewer retained tokens. The default is set to 0.5.
- **target_token** (float): The target number of tokens post-compression. This parameter is mutually exclusive with **ratio**. Default is -1.
- **iterative_size** (int): Segment size for Iterative Token-level Prompt Compression. The default is set to 200.
- **force_context_ids** (List[int], optional): Indexes of context elements to be forcefully retained. Default is None.
- **force_context_number** (int, optional): The number of contexts to be forcefully retained in Coarse-level Prompt Compression. Default is None.
- **use_sentence_level_filter** (bool, optional): Enables sentence-level prompt compression. Default is False.
- **use_context_level_filter** (bool, optional): Enables context-level prompt compression. Default is True.
- **use_token_level_filter** (bool, optional): Enables token-level prompt compression. Default is True.
- **keep_split** (bool, optional): Determines whether to retain all newline separators ("\n\n") in the prompt. Default is False.
- **keep_first_sentence** (int, optional): Specifies whether to retain the first 'k' sentences in each context. Default is 0.
- **keep_last_sentence** (int, optional): Specifies whether to retain the last 'k' sentences in each context. Default is 0.
- **keep_sentence_number** (int, optional): Specifies the number of sentences to retain in each context. Default is 0.
- **high_priority_bonus** (int, optional): Assigns a priority bonus to sentences retained by the **keep_first_sentence** or **keep_last_sentence** settings. Default is 100.
- **context_budget** (str, optional): Budget for Coarse-level Prompt Compression, with supported operators like "*1.5" or "+100". Default is "+100".
- **token_budget_ratio** (float, optional): Budget ratio for sentence-level Prompt Compression. Default is 1.4.
- **condition_in_question** (str, optional): Determines the use of question-aware coarse-level prompt compression. Options include "none", "after", "before". Default is "none".
- **reorder_context** (str, optional): Method for document reordering before compression in LongLLMLingua. Options include "original", "sort", "two_stage". Default is "original".
- **dynamic_context_compression_ratio** (float, optional): Ratio for dynamic context compression in LongLLMLingua. Default is 0.0.
- **condition_compare** (bool, optional): Enables Iterative Token-level Question-aware Fine-Grained Compression in LongLLMLingua. Default is False.
- **add_instruction** (bool, optional): Determines whether to add an instruction before the prompt in Iterative Token-level Question-aware Fine-Grained Compression. Default is False.
- **rank_method** (str, optional): Selects the ranking method for Coarse-level Prompt Compression, with support for various embedding and reranker methods, as well as LLMLingua and LongLLMLingua. Default is "llmlingua".
    - "llmlingua": Employs the coarse-grained prompt compression technique of **LLMLingua**.
    - "longllmlingua": Utilizes the question-aware coarse-grained prompt compression method in **LongLLMLingua** (recommended).
    - Traditional Retrieval Methods:
        - "bm25": A bag-of-words retrieval function that ranks documents based on the occurrence of query terms, irrespective of their proximity within the documents.
        - "gzip": A retrieval method based on GZIP compression. For further information, see [GZIP Retrieval Method](https://aclanthology.org/2023.findings-acl.426).
    - Embedding-Based Retrieval Methods:
        - "sentbert": An embedding-based retrieval method. Learn more at [SentenceBERT](https://www.sbert.net).
        - "openai": Utilizes "text-embedding-ada-002" as the embedding model from OpenAI.
        - "bge": An embedding-based retrieval method using "BAAI/bge-large-en-v1.5". For additional information, visit [BGE-Large-EN-V1.5](https://huggingface.co/BAAI/bge-large-en-v1.5).
        - "voyageai": An embedding-based retrieval method provided by VoyageAI. More details at [VoyageAI](https://www.voyageai.com).
        - "jinza": An embedding-based retrieval method using "jinaai/jina-embeddings-v2-base-en". Further details are available at [JinaAI Embeddings](https://huggingface.co/jinaai/jina-embeddings-v2-base-en).
    - Reranker Methods:
        - "bge_reranker": A reranker-based method using "BAAI/bge-reranker-large". More information can be found at [BGE Reranker Large](https://huggingface.co/BAAI/bge-reranker-large).
        - "bge_llmembedder": A reranker-based method using "BAAI/llm-embedder". For more details, refer to [BAAI LLM Embedder](https://huggingface.co/BAAI/llm-embedder).
        - "cohere": A reranker-based method using "rerank-english-v2.0" from Cohere. Learn more at [Cohere Rerank](https://cohere.com/rerank).
- **concate_question** (bool, optional): Determines whether to include the question in the compressed prompt. Default is True.


### Response

- **compressed_prompt** (str): The compressed prompt.
- **origin_tokens** (int): Number of tokens in the original prompt.
- **compressed_tokens** (int): Number of tokens in the compressed prompt.
- **ratio** (str): Actual compression ratio.
- **saving** (str): Savings in GPT-4 cost.

## Post-Processing

Recover the original response from a compressed prompt:

```python
recovered_response = llm_lingua.recover(
    original_prompt: str,
    compressed_prompt: str,
    response: str,
)
```

### Parameters

- **original_prompt** (str): The original prompt.
- **compressed_prompt** (str): The compressed prompt.
- **response** (str): The response from black-box LLMs based on the compressed prompt.

### Response

- **recovered_response** (str): The recovered response, integrating the original prompt's context.

## Advanced Usage

### Utilizing Quantized Small Models

(LLong)LLMLingua supports the use of quantized small models such as `TheBloke/Llama-2-7b-Chat-GPTQ`, which require less than 8GB of GPU memory.

To begin, ensure you install the necessary packages with:

```bash
pip install optimum auto-gptq
```

Then, initialize your model as follows:

```python
from llmlingua import PromptCompressor

llm_lingua = PromptCompressor("TheBloke/Llama-2-7b-Chat-GPTQ", model_config={"revision": "main"})
```

### Integration with LlamaIndex

Thanks to the contributions of Jerry Liu (@jerryjliu), (Long)LLMLingua can be seamlessly integrated into LlamaIndex. Here's an example of how to initialize (Long)LLMLingua within LlamaIndex:

```python
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import CompactAndRefine
from llama_index.indices.postprocessor import LongLLMLinguaPostprocessor

node_postprocessor = LongLLMLinguaPostprocessor(
    instruction_str="Given the context, please answer the final question",
    target_token=300,
    rank_method="longllmlingua",
    additional_compress_kwargs={
        "condition_compare": True,
        "condition_in_question": "after",
        "context_budget": "+100",
        "reorder_context": "sort",  # Enables document reordering
        "dynamic_context_compression_ratio": 0.4, # Enables dynamic compression ratio
    },
)
```

For a more detailed guide, please refer to [RAGLlamaIndex Example](https://github.com/microsoft/LLMLingua/blob/main/examples/RAGLlamaIndex.ipynb).
