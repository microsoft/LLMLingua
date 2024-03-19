# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from time import sleep

from utils import load_model_and_tokenizer

SLEEP_TIME_SUCCESS = 10
SLEEP_TIME_FAILED = 62


class PromptCompressor:
    def __init__(
        self,
        model_name,
        user_prompt,
        system_prompt=None,
        temperature=0.3,
        top_p=1.0,
        n_max_token=32700,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        print(self.system_prompt)
        print(self.user_prompt)

        self.model, self.tokenizer = load_model_and_tokenizer(
            self.model_name, chat_completion=True
        )
        self.n_max_token = n_max_token

    def query_template(self, text, n_max_new_token=4096):
        if self.user_prompt and "{text_to_compress}" in self.user_prompt:
            prompt = self.user_prompt.format(text_to_compress=text)
        else:
            prompt = text

        len_sys_prompt = 0
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}]
            len_sys_prompt = len(self.tokenizer.encode(self.system_prompt))
        token_ids = self.tokenizer.encode(prompt)
        if len(token_ids) > (self.n_max_token - n_max_new_token - len_sys_prompt):
            half = int((self.n_max_token - n_max_new_token - len_sys_prompt) / 2) - 1
            prompt = self.tokenizer.decode(token_ids[:half]) + self.tokenizer.decode(
                token_ids[-half:]
            )
        messages.append({"role": "user", "content": prompt})
        return messages

    def compress(self, text, n_max_new_token=4096):
        messages = self.query_template(text, n_max_new_token)
        comp = None
        while comp is None:
            try:
                request = {
                    "messages": messages,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": n_max_new_token,
                }
                response = self.model.create(engine=self.model_name, **request)
                if "choices" not in response:
                    print(response)
                comp = response["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"error: {e}")
                sleep(SLEEP_TIME_FAILED)
        # sleep(SLEEP_TIME_SUCCESS)
        return comp
