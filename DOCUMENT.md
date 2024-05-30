# LLMLingua Series Documentation

## Principles

1. **Prompt Sensitivity**: Different components of a prompt, like instructions and questions, vary in sensitivity to compression. Contexts or documents, for example, are less sensitive. It's advisable to separate these components in the prompt for demonstrations, instructions, and questions.
2. **Granular Division**: For multi-document QA and few-shot learning, divide demonstrations and contexts into independent granularities. This helps with budget control and document reordering.
3. **Essential Character Preservation**: Preserve essential characters as required by the scenario rules. **Support for this feature is now available in Structured Prompt Compression and LLMLingua-2**.
4. **Optimization through Experimentation**: Experiment with various target compression ratios and other hyperparameters to optimize performance.

## Basic Usage

With **LLMLingua**, you can easily compress your prompts. Here’s how you can do it:

```python
from llmlingua import PromptCompressor

llm_lingua = PromptCompressor()
compressed_prompt = llm_lingua.compress_prompt(prompt, instruction="", question="", target_token=200)

# > {'compressed_prompt': 'Question: Sam bought a dozen boxes, each with 30 highlighter pens inside, for $10 each box. He reanged five of boxes into packages of sixlters each and sold them $3 per. He sold the rest theters separately at the of three pens $2. How much did make in total, dollars?\nLets think step step\nSam bought 1 boxes x00 oflters.\nHe bought 12 * 300ters in total\nSam then took 5 boxes 6ters0ters.\nHe sold these boxes for 5 *5\nAfterelling these  boxes there were 3030 highlighters remaining.\nThese form 330 / 3 = 110 groups of three pens.\nHe sold each of these groups for $2 each, so made 110 * 2 = $220 from them.\nIn total, then, he earned $220 + $15 = $235.\nSince his original cost was $120, he earned $235 - $120 = $115 in profit.\nThe answer is 115',
#  'origin_tokens': 2365,
#  'compressed_tokens': 211,
#  'ratio': '11.2x',
#  'saving': ', Saving $0.1 in GPT-4.'}

## Or use the phi-2 model,
llm_lingua = PromptCompressor("microsoft/phi-2")

## Or use the quantation model, like TheBloke/Llama-2-7b-Chat-GPTQ, only need <8GB GPU memory.
## Before that, you need to pip install optimum auto-gptq
llm_lingua = PromptCompressor("TheBloke/Llama-2-7b-Chat-GPTQ", model_config={"revision": "main"})
```

To try **LongLLMLingua** in your scenarios, you can use

```python
from llmlingua import PromptCompressor

llm_lingua = PromptCompressor()
compressed_prompt = llm_lingua.compress_prompt(
    prompt_list,
    question=question,
    ratio=0.55,
    # Set the special parameter for LongLLMLingua
    condition_in_question="after_condition",
    reorder_context="sort",
    dynamic_context_compression_ratio=0.3, # or 0.4
    condition_compare=True,
    context_budget="+100",
    rank_method="longllmlingua",
)
```

To try **LLMLingua-2** in your scenarios, you can use

```python
from llmlingua import PromptCompressor

llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
)
compressed_prompt = llm_lingua.compress_prompt(prompt, rate=0.33, force_tokens = ['\n', '?'])

## Or use LLMLingua-2-small model
llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
)
```

## Advanced Usage

### Utilizing Small Models

### Using phi-2

Thanks to the efforts of the community, phi-2 is now available for use in LLMLingua.

Before using it, please update your transformers to the GitHub version by running `pip install -U git+https://github.com/huggingface/transformers.git`.

```python
llm_lingua = PromptCompressor("microsoft/phi-2")
```

### Quantized Models

(Long)LLMLingua supports the use of quantized small models such as `TheBloke/Llama-2-7b-Chat-GPTQ`, which require less than 8GB of GPU memory.

To begin, ensure you install the necessary packages with:

```bash
pip install optimum auto-gptq
```

Then, initialize your model as follows:

```python
from llmlingua import PromptCompressor

llm_lingua = PromptCompressor("TheBloke/Llama-2-7b-Chat-GPTQ", model_config={"revision": "main"})
```

### Structured Prompt Compression

Split text into sections, decide on whether to compress and its rate. Use `<llmlingua></llmlingua>` tags for context segmentation, with optional `rate` and `compress` parameters.

```python
structured_prompt = """<llmlingua, compress=False>Speaker 4:</llmlingua><llmlingua, rate=0.4> Thank you. And can we do the functions for content? Items I believe are 11, three, 14, 16 and 28, I believe.</llmlingua><llmlingua, compress=False>
Speaker 0:</llmlingua><llmlingua, rate=0.4> Item 11 is a communication from Council on Price recommendation to increase appropriation in the general fund group in the City Manager Department by $200 to provide a contribution to the Friends of the Long Beach Public Library. Item 12 is communication from Councilman Super Now. Recommendation to increase appropriation in the special advertising and promotion fund group and the city manager's department by $10,000 to provide support for the end of summer celebration. Item 13 is a communication from Councilman Austin. Recommendation to increase appropriation in the general fund group in the city manager department by $500 to provide a donation to the Jazz Angels . Item 14 is a communication from Councilman Austin. Recommendation to increase appropriation in the general fund group in the City Manager department by $300 to provide a donation to the Little Lion Foundation. Item 16 is a communication from Councilman Allen recommendation to increase appropriation in the general fund group in the city manager department by $1,020 to provide contribution to Casa Korero, Sew Feria Business Association, Friends of Long Beach Public Library and Dave Van Patten. Item 28 is a communication. Communication from Vice Mayor Richardson and Council Member Muranga. Recommendation to increase appropriation in the general fund group in the City Manager Department by $1,000 to provide a donation to Ron Palmer Summit. Basketball and Academic Camp.</llmlingua><llmlingua, compress=False>
Speaker 4:</llmlingua><llmlingua, rate=0.6> We have a promotion and a second time as councilman served Councilman Ringa and customers and they have any comments.</llmlingua>"""
compressed_prompt = llm_lingua.structured_compress_prompt(structured_prompt, instruction="", question="", rate=0.5)
print(compressed_prompt['compressed_prompt'])

# > Speaker 4:. And can we do the functions for content? Items I believe are11,,116 28,.
# Speaker 0: a from Council on Price to increase the fund group the Manager0 provide a the the1 is Councilman Super Now. the special group the provide the summerman a the Jazzels a communication from Councilman Austin. Recommendation to increase appropriation in the general fund group in the City Manager department by $300 to provide a donation to the Little Lion Foundation. Item 16 is a communication from Councilman Allen recommendation to increase appropriation in the general fund group in the city manager department by $1,020 to provide contribution to Casa Korero, Sew Feria Business Association, Friends of Long Beach Public Library and Dave Van Patten. Item 28 is a communication. Communication from Vice Mayor Richardson and Council Member Muranga. Recommendation to increase appropriation in the general fund group in the City Manager Department by $1,000 to provide a donation to Ron Palmer Summit. Basketball and Academic Camp.
# Speaker 4: We have a promotion and a second time as councilman served Councilman Ringa and customers and they have any comments.
```

### Compress Json data

You can specify the compression method for each key and value by passing a config or a yaml config file. Each key must include four parameters: `rate` indicates the compression ratio for the corresponding value, `compress` indicates whether the corresponding value is compressed, `value_type` indicates the data type of the value, and `pair_remove` indicates whether the key-value pair can be completely deleted.

```python
json_data = {
    "id": 987654,
    "name": "John Doe",
    "skills": ["Java","Python","Machine Learning","Cloud Computing","AI Development"],
    "biography": "John Doe, born in New York in 1985, is a renowned software engineer with over 10 years of experience in the field. John graduated from MIT with a degree in Computer Science and has since worked with several Fortune 500 companies. He has a passion for developing innovative software solutions and has contributed to numerous open source projects. John is also an avid writer and speaker at tech conferences, sharing his insights on emerging technologies and their impact on the business world. In his free time, John enjoys hiking, reading science fiction novels, and playing the piano. At TechCorp, John was responsible for leading a team of software engineers and overseeing the development of scalable web applications. He played a key role in driving the adoption of cloud technologies within the company, significantly enhancing the efficiency of their digital operations. In his John on developingedge AI and implementing machine learning solutions for various business applications. He was instrumental in developing a predictive analytics tool that transformed the company's approach to data-driven decision making."
}
json_config = {
    "id": {
        "rate": 1,
        "compress": False,
        "value_type": "int",
        "pair_remove": True
    },
    "name": {
        "rate": 0.7,
        "compress": False,
        "value_type": "str",
        "pair_remove": False
    },
    "skills": {
        "rate": 0.2,
        "compress": True,
        "value_type": "list",
        "pair_remove": True
    },
    "biography": {
        "rate": 0.3,
        "compress": True,
        "value_type": "str",
        "pair_remove": True
    }
}
compressed_prompt = llm_lingua.compress_json(json_data, json_config, use_keyvalue_level_filter=True)
print(compressed_prompt['compressed_prompt'])
# > {'id': 987654, 'name': 'John Doe', 'skills': ['', '', '', '', 'AI'], 'biography': ",York in a has several for developing has avid and speaker at,on and enjoys reading fiction playing. At Tech John for and of scalable He in the of cloud technologies,significantly enhancing the efficiency of their digital operations. In his John on developingedge AI and implementing machine learning solutions for various business applications. He was instrumental in developing a predictive analytics tool that transformed the company's approach to data-driven decision making."}
```

### Integration with LangChain

Thanks to the contributions of Ayo Ayibiowu (@thehapyone), (Long)LLMLingua can be seamlessly integrated into LangChain. Here's an example of how to initialize (Long)LLMLingua within LangChain:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers.document_compressors import LLMLinguaCompressor
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

compressor = LLMLinguaCompressor(model_name="openai-community/gpt2", device_map="cpu")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.get_relevant_documents(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)
```

For a more detailed guide, please refer to [Notebook](https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/retrievers/llmlingua.ipynb).

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

### Training Your Own LLMLingua-2

Not performing well on some domain-specific tasks? Don't worry, we've released code to help you build your own training data by instructing GPT-4 to compress your custom corpus and train compressors on the distilled data.

#### Data collection

First, format your data to a list of dict, with each dict containing at least two keys: _idx_ and _prompt_. [**format_data.py**](./experiments/llmlingua2/data_collection/format_data.py) illustrates how we format the meetingbank data.

Then, instruct GPT-4 to compress the original context.

```bash
cd experiments/llmlingua2/data_collection/
python compress.py --load_origin_from <your data path> \
--chunk_size 512 \
--compressor llmcomp \
--model_name gpt-4-32k \
--save_path <compressed data save path>

```

Then, assign label to the original words and filter out poor compression samples.

```bash
cd experiments/llmlingua2/data_collection/
python label_word.py \
--load_prompt_from <compressed data save path> \
--window_size 400 \
--save_path <labeled data save path> \

```

Filter out some poorly compressed / labeled samples.

```bash
cd experiments/llmlingua2/data_collection/
python filter.py --load_path <labeled data save path> \
--save_path <kept data save path>
```

#### Compressor Training

The [**model_training**](./experiments/llmlingua2/model_training) folder contains the code to train compressor on the distilled data.

```bash
cd cd experiments/llmlingua2/model_training/
python train_roberta.py --data_path <kept data save path>
```

## Detailed of Pramater

### Initialization

Initialize **LLMLingua**, **LongLLMLingua**, and **LLMLingua-2** with the following parameters:

```python
from llmlingua import PromptCompressor

llm_lingua = PromptCompressor(
    model_name="NousResearch/Llama-2-7b-hf", # Default model, use "microsoft/llmlingua-2-xlm-roberta-large-meetingbank" or "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank" for LLMLingua-2
    device_map="cuda",  # Device environment (e.g., 'cuda', 'cpu', 'mps')
    model_config={},  # Configuration for the Huggingface model
    open_api_config={},  # Configuration for OpenAI Embedding
    use_llmlingua2=False, # Whether to use llmlingua-2
)
```

#### Parameters

- **model_name** (str): Name of the small language model from Huggingface, use "microsoft/llmlingua-2-xlm-roberta-large-meetingbank" or "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank" for LLMLingua-2. Defaults to "NousResearch/Llama-2-7b-hf".
- **device_map** (str): The computing environment. Options include 'cuda', 'cpu', 'mps', 'balanced', 'balanced_low_0', 'auto'. Default is 'cuda'.
- **model_config** (dict, optional): Configuration for the Huggingface model. Defaults to {}.
- **open_api_config** (dict, optional): Configuration for OpenAI Embedding in coarse-level prompt compression. Defaults to {}.
- **use_llmlingua2** (bool, optional): Whether to use llmlingua-2 for prompt compression. Defaults is False.

### Function Call

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
    # Parameters for LLMLingua-2
    target_context: int = -1,  # Context Budget for Coarse-level Prompt Compression
    context_level_rate: float = 1.0, # Compression rate for Coarse-level Prompt Compression
    context_level_target_token: int = -1, # Token Budget for Coarse-level Prompt Compression
    return_word_label: bool = False, # Whether to return words with corresponding labels. Default is False.
    word_sep: str = '\t\t|\t\t', # The sep token used in fn_labeled_original_prompt to partition words.
    label_sep: str = " ", # The sep token used in fn_labeled_original_prompt to partition word and label.
    token_to_word: str = 'mean', # How to convert token probability to word probability. Default is 'mean'.
    force_tokens: List[str] = [], # List of specific tokens to always include in the compressed result. Default is [].
    force_reserve_digit: bool = False, # Whether to forcibly reserve tokens that containing digit (0,...,9). Default is False.
    drop_consecutive: bool = False, # Whether to drop tokens which are in 'force_tokens' but appears consecutively in compressed prompt. Default is False
    chunk_end_tokens: List[str] = [".", "\n"] # The early stop tokens for segmenting chunk. Default is [".", "\n"].
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
- **context_budget** (str, optional): Budget for Coarse-level Prompt Compression, with supported operators like "\*1.5" or "+100". Default is "+100".
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
- **target_context** (int): The maximum number of contexts to be achieved in context level compression. Default is -1 (no compression on context level).
- **context_level_rate** (float): The compression rate target to be achieved in context level. Default is 1.0 (no compression on context level).
- **context_level_target_token** (int): The maximum number of tokens to be achieved in context level compression. Default is -1 (no compression on context level).
- **return_word_label** (bool): Whether to return words with corresponding labels. Default is False.
- **word_sep** (str): The sep token used in fn_labeled_original_prompt to partition words. Only used when return_word_label==True. Default is '\t\t|\t\t'
- **label_sep** (str): The sep token used in fn_labeled_original_prompt to partition word and label. Only used when return_word_label==True. Default is ' '
- **token_to_word** (str): The method to convert token probability to word probability. Default is 'mean'
- **force_tokens** (List[str], optional): List of specific tokens to always include in the compressed result. Default is [].
- **force_reserve_digit** (bool, optional): Whether to forcibly reserve tokens that containing digit (0,...,9). Default is False.
- **drop_consecutive** (bool, optinal): Whether to drop tokens which are in 'force_tokens' but appears consecutively in compressed prompt. Default is False.
- **chunk_end_tokens** (List[str], optinal): The early stop tokens for segmenting chunk. Default is [".", "\n"].

### Response

- **compressed_prompt** (str): The compressed prompt.
- **origin_tokens** (int): Number of tokens in the original prompt.
- **compressed_tokens** (int): Number of tokens in the compressed prompt.
- **ratio** (str): Actual compression ratio.
- **saving** (str): Savings in GPT-4 cost.

Additional Response Parameter for LLMLingua-2.

- **fn_labeled_original_prompt** (str): original words along with their labels indicating whether to reserve in compressed prompt, in the format (word1 label_sep label2 word_sep word2 label_sep label2 ...). Only return when return_word_label==True.
- **compressed_prompt_list** (str): List of the compressed prompt.

### Post-Processing

Recover the original response from a compressed prompt:

```python
recovered_response = llm_lingua.recover(
    original_prompt: str,
    compressed_prompt: str,
    response: str,
)
```

#### Parameters

- **original_prompt** (str): The original prompt.
- **compressed_prompt** (str): The compressed prompt.
- **response** (str): The response from black-box LLMs based on the compressed prompt.

#### Response

- **recovered_response** (str): The recovered response, integrating the original prompt's context.
