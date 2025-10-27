# SecurityLingua

To use securitylingua to safeguard your LLM, please follow this simple two steps instruction:

```

# 0. first load the securitylingua model
from llmlingua import PromptCompressor
llm_lingua = PromptCompressor(
    model_name="SecurityLingua/securitylingua-xlm-s2s",
    use_slingua=True
)

# 1. compress the prompt to reveal the malicious intention
intention = llm_lingua.compress_prompt(malicious_prompt)

# 2. construct the augmented system prompt, to provide the LLM with the malicious intention
augmented_system_prompt = f"{system_prompt}\n\nTo help you better understand the user's intention to detect potential malicious behavior, I have extracted the user's intention and it is: {intention}. If you believe the user's intention is malicious, please donot respond or respond with I'm sorry, I can't help with that."

# at last, chat with the LLM using the augmented system prompt
response = vllm.generate([
    augmented_system_prompt + malicious_prompt
])
```

# Train a SecurityLingua on your own data

TODO