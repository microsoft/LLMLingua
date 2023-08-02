<p align="center" width="100%">
<img src="images/LLMLingua_logo.png" alt="LLMLingua" style="width: 20%; min-width: 100px; display: block; margin: auto;">
</p>

# LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models

This repo contains the code for LLMLingua, a project that compresses prompts and speeds up inference for LLMs with minimal loss of performance.

[LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models]() ().
_Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang and Lili Qiu_


PS: We also release a hackathon demo to show our idea. Please check [here](https://hackbox.microsoft.com/hackathons/hackathon2023/project/26540).


## 🎥 Overview

![image](./images/LLMLingua_motivation.png)

- Have you ever tried to input a long text and ask ChatGPT to summarize it, only to be told that it exceeds the token limit? ​
- Have you ever spent a lot of time fine-tuning the personality of ChatGPT, only to find that it forgets the previous instructions after a few rounds of dialogue? ​
- Have you ever used the GPT3.5/4 API for experiments, and got good results, but also received a huge bill after a few days? ​

Large language models, such as ChatGPT and GPT-4, impress us with their amazing generalization and reasoning abilities, but they also come with some drawbacks, such as the prompt length limit and the prompt-based pricing scheme.​

![image](./images/LLMLingua_framework.png)

Now you can use **LLMLingua**!​

A simple and efficient method to compress prompt up to **20x**.​

- 💰 **Saving cost**, not only prompt, but also the generation length;​
- 📝 **Support longer contexts**;​
- ⚖️ **Robustness**, no need any training for the LLMs;​
- 🕵️ **Keeping** the original prompt knowledge like ICL, reasoning, etc.​
- 📜 **KV-Cache compression**, speedup inference;​

![image](./images/LLMLingua_demo.png)

If you find this repo helpful, please cite the following paper:

```bibtex
@inproceedings{jiang-etal-2023-llmlingua,
    title = "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models",
    author = "Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang and Lili Qiu",
}
```

## 🎯 Quick Start

Install LLMLingua,

```bash
pip install -e .
```
    
Then, you can use LLMLingua to compress your prompt,
    
```python
from llmlingua import PromptCompressor

llmlingua = PromptCompressor()
compressed_prompt = llmlingua.compress_prompt(prompt, instruction="", question="", target_token=200)

# > {'compressed_prompt': 'Question: Sam bought a dozen boxes, each with 30 highlighter pens inside, for $10 each box. He reanged five of boxes into packages of sixlters each and sold them $3 per. He sold the rest theters separately at the of three pens $2. How much did make in total, dollars?\nLets think step step\nSam bought 1 boxes x00 oflters.\nHe bought 12 * 300ters in total\nSam then took 5 boxes 6ters0ters.\nHe sold these boxes for 5 *5\nAfterelling these  boxes there were 3030 highlighters remaining.\nThese form 330 / 3 = 110 groups of three pens.\nHe sold each of these groups for $2 each, so made 110 * 2 = $220 from them.\nIn total, then, he earned $220 + $15 = $235.\nSince his original cost was $120, he earned $235 - $120 = $115 in profit.\nThe answer is 115',
#  'origin_tokens': 2365,
#  'compressed_tokens': 211,
#  'ratio': '11.2x',
#  'saving': ', Saving $0.1 in GPT-4.'}
```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
