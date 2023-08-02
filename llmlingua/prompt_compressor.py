from collections import defaultdict
from typing import List

import nltk
import torch

import tiktoken
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


class PromptCompressor:
    def __init__(
        self, model_name: str = "NousResearch/Llama-2-7b-hf", device_map: str = "cuda"
    ):
        self.load_model(model_name, device_map)

    def load_model(self, model_name: str, device_map: str = "cuda"):
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = (
            config.pad_token_id if config.pad_token_id else tokenizer.eos_token_id
        )
        if device_map == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                config=config,
                ignore_mismatched_sizes=True,
            ).cuda()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype="auto",
                pad_token_id=tokenizer.pad_token_id,
                offload_folder="/tmp/offload",
                offload_state_dict=True,
                cache_dir="/tmp/cache",
                use_auth_token=True,
            )
        self.tokenizer = tokenizer
        self.model = model

    def get_ppl(
        self,
        text: str,
        granularity: str = "sentence",
        input_ids=None,
        attention_mask=None,
    ):
        if input_ids is None:
            tokenized_text = self.tokenizer(text, return_tensors="pt")
            input_ids = tokenized_text["input_ids"].cuda()
            attention_mask = tokenized_text["attention_mask"].cuda()
        with torch.no_grad():
            response = self.model(input_ids, attention_mask=attention_mask)

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        shift_logits = response.logits[
            ..., :-1, :
        ].contiguous()  # batch_size x seq_len x vocab_size
        shift_labels = input_ids[..., 1:].contiguous()  # batch_size x seq_len
        # Flatten the tokens
        active = (attention_mask == 1)[..., :-1].view(-1)
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)
        if granularity == "token":
            return loss
        elif granularity == "sentence":
            return loss.mean()

    def __call__(self, *args, **kwargs):
        return self.compress_prompt(*args, **kwargs)

    def compress_prompt(
        self,
        demonstrations: List[str],
        instruction: str = "",
        question: str = "",
        ratio: float = 0.5,
        target_token: float = -1,
        iterative_size: int = 200,
        length_ratio: float = 0.0,
        force_demonstrations_ids: List[int] = None,
        use_sentence_level_filter: bool = False,
        use_demonstrate_level_filter: bool = True,
        keep_split: bool = False,
        keep_first_sentence: int = 0,
        keep_last_sentence: int = 0,
        keep_sentence_number: int = 0,
        high_priority_bonus: int = 100,
        token_budget_ratio: float = 1.4,
        condition_in_question: bool = False,
    ):
        if isinstance(demonstrations, str):
            demonstrations = [demonstrations]
        origin_tokens = len(
            encoding.encode(
                "\n\n".join([instruction] + demonstrations + [question]).strip()
            )
        )
        demonstrations_tokens_length = [
            self.get_token_length(demonstration) for demonstration in demonstrations
        ]
        instruction_tokens_length, question_tokens_length = self.get_token_length(
            instruction
        ), self.get_token_length(question)
        if target_token == -1:
            target_token = (
                (
                    instruction_tokens_length
                    + question_tokens_length
                    + sum(demonstrations_tokens_length)
                )
                * (1 - ratio)
                - instruction_tokens_length
                - question_tokens_length
            )
        if len(demonstrations) > 1 and use_demonstrate_level_filter:
            demonstrations = self.control_demonstrations_budget(
                demonstrations,
                demonstrations_tokens_length,
                target_token,
                length_ratio,
                force_demonstrations_ids,
            )
        if use_sentence_level_filter:
            demonstrations = self.control_sentence_budget(
                demonstrations,
                target_token,
                keep_first_sentence=keep_first_sentence,
                keep_last_sentence=keep_last_sentence,
                keep_sentence_number=keep_sentence_number,
                high_priority_bonus=high_priority_bonus,
                token_budget_ratio=token_budget_ratio,
            )
        if condition_in_question:
            demonstrations = [question] + demonstrations
            start = self.get_token_length(question) + 2
        else:
            start = 0
        demonstrations = self.iterative_compress_prompt(
            demonstrations,
            target_token,
            iterative_size=iterative_size,
            keep_split=keep_split,
            start=start,
        )

        context = self.tokenizer.batch_decode(demonstrations[0])[0].replace("<s> ", "")
        if instruction:
            context = instruction + "\n\n" + context
        if question:
            context = context + "\n\n" + question

        compressed_tokens = len(encoding.encode(context))
        saving = (origin_tokens - compressed_tokens) * 0.06 / 1000
        return {
            "compressed_prompt": context,
            "origin_tokens": origin_tokens,
            "compressed_tokens": compressed_tokens,
            "ratio": f"{origin_tokens/compressed_tokens:.1f}x",
            "saving": f", Saving ${saving:.1f} in GPT-4.",
        }

    def get_token_length(self, text: str):
        return len(self.tokenizer(text).input_ids)

    def control_demonstrations_budget(
        self,
        demonstrations: List[str],
        demonstrations_tokens_length: List[int],
        target_token: float,
        length_ratio: float = 0.0,
        force_demonstrations_ids: List[int] = None,
    ):
        if force_demonstrations_ids is not None:
            return [demonstrations[ii] for ii in force_demonstrations_ids]
        demonstrations_ppl = [
            self.get_ppl(d) - dl * 2 / 250 * length_ratio
            for d, dl in zip(demonstrations, demonstrations_tokens_length)
        ]
        if target_token < 0:
            target_token = 100
        target_token += 100
        res = []
        for idx, ppl in sorted(enumerate(demonstrations_ppl), key=lambda x: -x[1]):
            target_token -= demonstrations_tokens_length[idx]
            res.append(demonstrations[idx])
            if target_token < 0:
                break
        return res

    def control_sentence_budget(
        self,
        demonstrations: List[str],
        target_token: float,
        keep_first_sentence: int = 0,
        keep_last_sentence: int = 0,
        keep_sentence_number: int = 0,
        high_priority_bonus: int = 100,
        token_budget_ratio: float = 1.4,
    ):
        def keep_sentence(dem_idx: int, sent_keep: int):
            idxs = sorted(dem_g[dem_idx], key=lambda x: sentence_ppl[x])[:sent_keep]
            for idx in idxs:
                sentence_ppl[idx] += high_priority_bonus

        sentences = [
            nltk.sent_tokenize(demonstration) for demonstration in demonstrations
        ]
        dem_g, s2de, idx = defaultdict(set), defaultdict(int), 0
        for idx_d, s in enumerate(sentences):
            for _ in s:
                dem_g[idx_d].add(idx)
                s2de[idx] = idx_d
                idx += 1

        demonstrations_sentences = [s for ii in sentences for s in ii]
        sentence_tokens_length = [
            self.get_token_length(sentence) for sentence in demonstrations_sentences
        ]
        if len(sentence_tokens_length) == 1:
            return demonstrations
        sentence_ppl = [
            self.get_ppl(sentence).cpu().numpy().item()
            for sentence in demonstrations_sentences
        ]
        if keep_first_sentence:
            sentence_ppl[:keep_first_sentence] = [
                ii + high_priority_bonus for ii in sentence_ppl[:keep_first_sentence]
            ]
        if keep_last_sentence:
            sentence_ppl[-keep_last_sentence:] = [
                ii + high_priority_bonus for ii in sentence_ppl[-keep_last_sentence:]
            ]
        if keep_sentence_number:
            for dem_idx in range(len(sentences)):
                keep_sentence(dem_idx, keep_sentence_number)

        N = len(demonstrations_sentences)
        sentence_flags = [False] * N
        if target_token < 0:
            target_token = 100
        target_token *= token_budget_ratio
        res = []
        for idx, ppl in sorted(enumerate(sentence_ppl), key=lambda x: -x[1]):
            target_token -= sentence_tokens_length[idx]
            sentence_flags[idx] = True
            if target_token < 0:
                break
        idx = 0
        res = []
        for s in sentences:
            tmp = [jj for ii, jj in enumerate(s) if sentence_flags[idx + ii]]
            res.append("\n".join(tmp))
            idx += len(s)
        return res

    def get_compressed_input(
        self,
        loss,
        input_ids,
        attention_mask,
        end=200,
        iterative_size=200,
        threshold=0.5,
        keep_flag=None,
        split_token_id: int = 13,
    ):
        need_idx = torch.concat([loss > threshold, loss[:1] > 0])
        need_idx[end:] = 1
        need_idx[: end - iterative_size] = 1
        last = -1
        if keep_flag is not None:
            for ii in range(end - iterative_size, end):
                if need_idx[ii] != 1:
                    continue
                now = input_ids[0][ii].detach().cpu().item()
                if (
                    now == split_token_id
                    and last == split_token_id
                    and keep_flag[ii].detach().cpu().item() == 0
                ):
                    need_idx[ii] = 0
                else:
                    last = now
        compressed_input_ids = input_ids[attention_mask == 1][need_idx].unsqueeze(0)
        compressed_attention_mask = attention_mask[attention_mask == 1][
            need_idx
        ].unsqueeze(0)
        if keep_flag is not None:
            keep_flag = keep_flag[need_idx]
        end -= (need_idx[:end] == 0).sum()
        return compressed_input_ids, compressed_attention_mask, keep_flag, end

    def get_estimate_threshold_base_distribution(self, ppl, target_token: int):
        target_token = max(0, min(len(ppl) - 1, int(target_token)))
        return ppl.sort(descending=True).values[target_token].detach().cpu().item()

    def iterative_compress_prompt(
        self,
        demonstrations: List[str],
        target_token: float,
        iterative_size: int = 200,
        keep_split: bool = False,
        split_token_id: int = 13,
        start: int = 0,
    ):
        demonstrations = "\n\n".join(demonstrations)
        tokenized_text = self.tokenizer(demonstrations, return_tensors="pt")
        input_ids = tokenized_text["input_ids"].cuda()
        attention_mask = tokenized_text["attention_mask"].cuda()

        N = (attention_mask == 1).sum()
        end = iterative_size + start
        compressed_input_ids, compressed_attention_mask = input_ids, attention_mask
        threshold, keep_flag = None, None
        if keep_split:
            input_ids_numpy = input_ids.cpu().detach().numpy()[0]
            N = len(input_ids_numpy)
            keep_flag = [
                int(
                    (
                        ii > 0
                        and input_ids_numpy[ii] == split_token_id
                        and input_ids_numpy[ii - 1] == split_token_id
                    )
                    or (
                        ii < N - 1
                        and input_ids_numpy[ii] == split_token_id
                        and input_ids_numpy[ii + 1] == split_token_id
                    )
                )
                for ii in range(N)
            ]
            keep_flag = torch.tensor(keep_flag).cuda()
        while end < compressed_input_ids.shape[1]:
            loss = self.get_ppl(
                "", "token", compressed_input_ids, compressed_attention_mask
            )
            # if threshold is None:
            threshold = self.get_estimate_threshold_base_distribution(
                loss, target_token
            )
            if keep_split:
                loss[keep_flag[:-1] == 1] = 100

            (
                compressed_input_ids,
                compressed_attention_mask,
                keep_flag,
                end,
            ) = self.get_compressed_input(
                loss,
                compressed_input_ids,
                compressed_attention_mask,
                end,
                iterative_size=iterative_size,
                threshold=threshold,
                keep_flag=keep_flag,
                split_token_id=split_token_id,
            )
            end += iterative_size
        return compressed_input_ids[:, start:], compressed_attention_mask[:, start:]
