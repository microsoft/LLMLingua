# LLMLingua's Responsible AI FAQ 

## What is LLMLingua? 

- LLMLingua is a simple and efficient method to compress prompt up to 20x and keeping the original prompt knowledge like ICL, reasoning, etc. 
- LLMLingua takes user-defined prompts and compression goals as input, and outputs a compressed prompt, which may often result in a form of expression that is difficult for humans to understand. 

## What can LLMLingua do?  

- LLMLingua can simultaneously reduce the length of prompts and the output of LLMs (20%-30%), thus saving API calls; 
- Compressed prompts from LLMLingua can be directly used with black-box LLMs, such as ChatGPT, GPT-4, and Claude; 
- By compressing prompts, LLMLingua allows for more information to be included within the original token length, thereby improving model performance; 
- LLMLingua relies on a small language model, like GPT-2 or LLaMA-7b, for perplexity calculations, which is a relatively low-cost approach; 
- Compressed prompts generated by LLMLingua can be understood by LLMs, preserving their original capabilities in downstream tasks and keeping the original prompt knowledge like ICL, reasoning, etc. LLMs can also recover the essential information from the compressed prompts; 
- LLMLingua is a robustness method, no need any training for the LLMs; 
- Additionally, LLMLingua can be used to compress KV-Cache, which speeds up inference. 

## What is/are LLMLingua’s intended use(s)? 

- Users who call black-box LLM APIs similar to GPT-4, those who utilize ChatGPT to handle longer content, as well as model deployers and cloud service providers, can benefit from these techniques. 

## How was LLMLingua evaluated? What metrics are used to measure performance? 

- In our experiments, we conducted a detailed evaluation of the performance of compressed prompts across various tasks, particularly in those involving LLM-specific capabilities, such as In-Context Learning, reasoning tasks, summarization, and conversation tasks. We assessed our approach using compression ratio and performance loss as evaluation metrics. 

## What are the limitations of LLMLingua? How can users minimize the impact of LLMLingua’s limitations when using the system? 

- The potential harmful, false or biased responses using the compressed prompts would likely be unchanged. Thus using LLMLingua has no inherent benefits or risks when it comes to those types of responsible AI issues. 
- LLMLingua may struggle to perform well at particularly high compression ratios, especially when the original prompts are already quite short. 

## What operational factors and settings allow for effective and responsible use of LLMLingua? 

- Users can set parameters such as the boundaries between different components (instruction, context, question) in the prompt, compression goals, and the small model used for compression calculations. Afterward, they can input the compressed prompt into black-box LLMs for use. 

## What is instruction, context, and question?

In our approach, we divide the prompts into three distinct modules: instruction, context, and question. Each prompt necessarily contains a question, but the presence of context and instruction is not always guaranteed.

- Question: This refers to the directives given by the user to the LLMs, such as inquiries, questions, or requests. Positioned after the instruction and context modules, the question module has a high sensitivity to compression.
- Context: This module provides the supplementary context needed to address the question, such as documents, demonstrations, web search results, or API call results. Located between the instruction and question modules, its sensitivity to compression is relatively low.
- Instruction: This module consists of directives given by the user to the LLMs, such as task descriptions. Placed before the instruction and context modules, the instruction module exhibits a high sensitivity to compression.
