import bisect
from collections import defaultdict
from typing import List

import numpy as np
import torch

import nltk
import tiktoken
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


class PromptCompressor:
    def __init__(
        self,
        model_name: str = "NousResearch/Llama-2-7b-hf",
        device_map: str = "cuda",
        model_config: dict = {},
        open_api_config: dict = {},
    ):
        self.load_model(model_name, device_map, model_config)
        self.retrieval_model = None
        self.retrieval_model_name = None
        self.open_api_config = open_api_config
        self.cache_bos_num = 10
        self.prefix_bos_num = 100

    def load_model(
        self, model_name: str, device_map: str = "cuda", model_config: dict = {}
    ):
        trust_remote_code = model_config.get("trust_remote_code", True)
        if "trust_remote_code" not in model_config:
            model_config["trust_remote_code"] = trust_remote_code
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if model_config.get("pad_to_left", True):
            tokenizer.padding_side = "left"
            tokenizer.pad_token_id = (
                config.pad_token_id if config.pad_token_id else tokenizer.eos_token_id
            )
        self.device = (
            device_map if any(key in device_map for key in ["cuda", "cpu"]) else "cuda"
        )
        if "cuda" in device_map or "cpu" in device_map:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto" if device_map == "cuda" else torch.float32,
                device_map=device_map,
                config=config,
                ignore_mismatched_sizes=True,
                **model_config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype="auto",
                pad_token_id=tokenizer.pad_token_id,
                offload_folder="/tmp/offload",
                offload_state_dict=True,
                cache_dir="/tmp/cache",
                **model_config
            )
        self.tokenizer = tokenizer
        self.model = model
        self.context_idxs = []
        self.max_position_embeddings = config.max_position_embeddings

    def get_ppl(
        self,
        text: str,
        granularity: str = "sentence",
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        return_kv=False,
        end=None,
        condition_mode: str = "none",
        condition_pos_id: int = 0,
    ):
        if input_ids is None:
            tokenized_text = self.tokenizer(text, return_tensors="pt")
            input_ids = tokenized_text["input_ids"].to(self.device)
            attention_mask = tokenized_text["attention_mask"].to(self.device)
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
        else:
            past_length = 0
        if end is None:
            end = input_ids.shape[1]
        end = min(end, past_length + self.max_position_embeddings)
        with torch.no_grad():
            response = self.model(
                input_ids[:, past_length:end],
                attention_mask=attention_mask[:, :end],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = response.past_key_values

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        shift_logits = response.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., past_length + 1 : end].contiguous()
        # Flatten the tokens
        active = (attention_mask[:, past_length:end] == 1)[..., :-1].view(-1)
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)
        if condition_mode == "before":
            loss = loss[:condition_pos_id]
        elif condition_mode == "after":
            loss = loss[condition_pos_id:]
        res = loss.mean() if granularity == "sentence" else loss
        return (res, past_key_values) if return_kv else res

    def __call__(self, *args, **kwargs):
        return self.compress_prompt(*args, **kwargs)

    def compress_prompt(
        self,
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
        rank_method: str = "llmlingua",
        concate_question: bool = True,
    ):
        if not context:
            context = [" "]
        if isinstance(context, str):
            context = [context]
        assert not (
            rank_method == "longllmlingua" and not question
        ), "In the LongLLMLingua, it is necessary to set a question."
        if condition_compare and "_condition" not in condition_in_question:
            condition_in_question += "_condition"
        if rank_method == "longllmlingua":
            if condition_in_question == "none":
                condition_in_question = "after"
        elif rank_method == "llmlingua":
            condition_in_question = (
                "none"
                if "_condition" not in condition_in_question
                else "none_condition"
            )
        origin_tokens = len(
            encoding.encode("\n\n".join([instruction] + context + [question]).strip())
        )
        context_tokens_length = [self.get_token_length(c) for c in context]
        instruction_tokens_length, question_tokens_length = self.get_token_length(
            instruction
        ), self.get_token_length(question)
        if target_token == -1:
            target_token = (
                (
                    instruction_tokens_length
                    + question_tokens_length
                    + sum(context_tokens_length)
                )
                * (1 - ratio)
                - instruction_tokens_length
                - (question_tokens_length if concate_question else 0)
            )
        condition_flag = "_condition" in condition_in_question
        condition_in_question = condition_in_question.replace("_condition", "")

        if len(context) > 1 and use_context_level_filter:
            context, dynamic_ratio = self.control_context_budget(
                context,
                context_tokens_length,
                target_token,
                force_context_ids,
                force_context_number,
                question,
                condition_in_question,
                reorder_context=reorder_context,
                dynamic_context_compression_ratio=dynamic_context_compression_ratio,
                rank_method=rank_method,
                context_budget=context_budget,
            )
        else:
            dynamic_ratio = [0.0] * len(context)

        if use_sentence_level_filter:
            context = self.control_sentence_budget(
                context,
                target_token,
                keep_first_sentence=keep_first_sentence,
                keep_last_sentence=keep_last_sentence,
                keep_sentence_number=keep_sentence_number,
                high_priority_bonus=high_priority_bonus,
                token_budget_ratio=token_budget_ratio,
                question=question,
                condition_in_question=condition_in_question,
                rank_method=rank_method,
            )

        if condition_flag:
            prefix = question + "\n\n" + instruction if add_instruction else question
            if (
                self.get_token_length(prefix) + 2 + iterative_size * 2
                > self.max_position_embeddings
            ):
                tokens = self.tokenizer(prefix, add_special_tokens=False).input_ids
                prefix = self.tokenizer.decode(
                    tokens[: self.prefix_bos_num]
                    + tokens[
                        len(tokens)
                        - self.max_position_embeddings
                        + 2
                        + self.prefix_bos_num
                        + 2 * iterative_size :
                    ]
                )
            start = self.get_token_length(prefix) + 2
            context = [prefix] + context
        else:
            start = 0

        if use_token_level_filter:
            context = self.iterative_compress_prompt(
                context,
                target_token,
                iterative_size=iterative_size,
                keep_split=keep_split,
                start=start,
                dynamic_ratio=dynamic_ratio,
                condition_compare=condition_compare,
            )
            compressed_prompt = (
                self.tokenizer.batch_decode(context[0])[0]
                .replace("<s> ", "")
                .replace("<s>", "")
            )
        else:
            compressed_prompt = "\n\n".join(context)

        res = []
        if instruction:
            res.append(instruction)
        if compressed_prompt.strip():
            res.append(compressed_prompt)
        if question and concate_question:
            res.append(question)

        compressed_prompt = "\n\n".join(res)

        compressed_tokens = len(encoding.encode(compressed_prompt))
        saving = (origin_tokens - compressed_tokens) * 0.06 / 1000
        return {
            "compressed_prompt": compressed_prompt,
            "origin_tokens": origin_tokens,
            "compressed_tokens": compressed_tokens,
            "ratio": f"{origin_tokens/compressed_tokens:.1f}x",
            "saving": f", Saving ${saving:.1f} in GPT-4.",
        }

    def get_token_length(self, text: str, add_special_tokens: bool = True):
        return len(
            self.tokenizer(text, add_special_tokens=add_special_tokens).input_ids
        )

    def get_condition_ppl(
        self,
        text: str,
        question: str,
        condition_in_question: str = "none",
        granularity: str = "sentence",
    ):
        if condition_in_question == "none":
            return self.get_ppl(text, granularity=granularity)
        elif condition_in_question == "before":
            return self.get_ppl(
                question + text,
                granularity=granularity,
                condition_mode="after",
                condition_pos_id=self.get_token_length(question) - 1,
            )
        elif condition_in_question == "after":
            return self.get_ppl(
                text + question,
                granularity=granularity,
                condition_mode="after",
                condition_pos_id=self.get_token_length(text) - 1,
            )

    def get_dynamic_compression_ratio(
        self,
        context: list,
        target_token: float,
        iterative_size: int,
        dynamic_ratio: list,
        start: int,
    ):
        def get_ratio(base: float, delta: float):
            return max(min(1, base + delta), 0)

        context_length = [self.get_token_length(ii, False) + 2 for ii in context]
        if start:
            context_length = context_length[1:]
        tau = target_token / (sum(context_length) + 1)
        res, idx, last, last_target = [], 0, 1, []
        while idx < len(context_length):
            if last + context_length[idx] >= iterative_size:
                last_target.append(
                    (iterative_size - last, get_ratio(tau, dynamic_ratio[idx]))
                )
                res.append(last_target)
                last = last + context_length[idx] - iterative_size
                if last > iterative_size:
                    k = last // iterative_size
                    res.extend(
                        [[(iterative_size, get_ratio(tau, dynamic_ratio[idx]))]] * k
                    )
                    last -= k * iterative_size

                last_target = (
                    [(last, get_ratio(tau, dynamic_ratio[idx]))] if last else []
                )
            else:
                last += context_length[idx]
                last_target.append(
                    (context_length[idx], get_ratio(tau, dynamic_ratio[idx]))
                )
            idx += 1
        if last_target:
            res.append(last_target)
        return res

    def control_context_budget(
        self,
        context: List[str],
        context_tokens_length: List[int],
        target_token: float,
        force_context_ids: List[int] = None,
        force_context_number: int = None,
        question: str = "",
        condition_in_question: str = "none",
        reorder_context: str = "original",
        dynamic_context_compression_ratio: float = 0.0,
        rank_method: str = "longllmlingua",
        context_budget: str = "+100",
    ):
        if force_context_ids is not None:
            return [context[ii] for ii in force_context_ids]
        demostrations_sort = self.get_rank_results(
            context,
            question,
            rank_method,
            condition_in_question,
            context_tokens_length,
        )

        if target_token < 0:
            target_token = 100
        target_token = eval("target_token" + context_budget)
        res = []
        used = force_context_ids if force_context_ids is not None else []

        self.context_idxs.append([x for idx, (x, _) in enumerate(demostrations_sort)])
        for idx, _ in demostrations_sort:
            if idx >= len(context_tokens_length):
                continue
            target_token -= context_tokens_length[idx]
            if idx not in used:
                used.append(idx)
            if target_token < 0 or (
                force_context_number is not None and len(res) >= force_context_number
            ):
                break
        original_used = used
        if reorder_context == "original":
            used = sorted(used)
        elif reorder_context == "two_stage":
            l, r = [_ for idx, _ in enumerate(used) if idx % 2 == 0], [
                _ for idx, _ in enumerate(used) if idx % 2 == 1
            ]
            used = l + r[::-1]

        if dynamic_context_compression_ratio > 0:
            N = len(used)
            if condition_in_question:
                rank = [
                    i
                    for i, _ in self.get_rank_results(
                        context,
                        question,
                        "longllmlingua",
                        "after",
                        context_tokens_length,
                    )
                ]
                used = sorted(used, key=lambda x: rank.index(x))
            dynamic_ratio = [
                i * (abs(dynamic_context_compression_ratio) / (N - 1)) if N > 1 else 0
                for i in range(-(N - 1), N, 2)
            ][::-1]
            dynamic_ratio_map = {i: j for i, j in zip(original_used, dynamic_ratio)}
            dynamic_ratio = [dynamic_ratio_map[i] for i in used]
        else:
            dynamic_ratio = [0.0] * len(used)

        res = [context[idx] for idx in used if idx < len(context)]
        return res, dynamic_ratio

    def control_sentence_budget(
        self,
        context: List[str],
        target_token: float,
        keep_first_sentence: int = 0,
        keep_last_sentence: int = 0,
        keep_sentence_number: int = 0,
        high_priority_bonus: int = 100,
        token_budget_ratio: float = 1.4,
        question: str = "",
        condition_in_question: str = "none",
        rank_method: str = "longllmlingua",
    ):
        def keep_sentence(dem_idx: int, sent_keep: int):
            idxs = sorted(dem_g[dem_idx], key=lambda x: sentence_ppl[x])[:sent_keep]
            for idx in idxs:
                sentence_ppl[idx] += high_priority_bonus

        sentences = [nltk.sent_tokenize(c) for c in context]
        dem_g, s2de, idx = defaultdict(set), defaultdict(int), 0
        for idx_d, s in enumerate(sentences):
            for _ in s:
                dem_g[idx_d].add(idx)
                s2de[idx] = idx_d
                idx += 1

        context_sentences = [s for ii in sentences for s in ii]
        sentence_tokens_length = [
            self.get_token_length(sentence) for sentence in context_sentences
        ]
        N = len(context_sentences)
        flags = list(range(len(context_sentences)))
        if len(sentence_tokens_length) == 1:
            return context
        if rank_method == "longllmlingua":
            sentence_ppl = [
                self.get_condition_ppl(sentence, question, condition_in_question)
                .cpu()
                .numpy()
                .item()
                for sentence in context_sentences
            ]
            if keep_first_sentence:
                sentence_ppl[:keep_first_sentence] = [
                    ii + high_priority_bonus
                    for ii in sentence_ppl[:keep_first_sentence]
                ]
            if keep_last_sentence:
                sentence_ppl[-keep_last_sentence:] = [
                    ii + high_priority_bonus
                    for ii in sentence_ppl[-keep_last_sentence:]
                ]
            if keep_sentence_number:
                for dem_idx in range(len(sentences)):
                    keep_sentence(dem_idx, keep_sentence_number)
            sort_direct = -1 if condition_in_question == "none" else 1
            sent_sort = sorted(
                enumerate(sentence_ppl), key=lambda x: sort_direct * x[1]
            )
        else:
            sent_sort = self.get_rank_results(
                context_sentences,
                question,
                rank_method,
                condition_in_question,
                [0] * len(context_sentences),
            )

        sentence_flags = [False] * N
        if target_token < 0:
            target_token = 100
        target_token *= token_budget_ratio
        res = []
        for idx, _ in sent_sort:
            idx = flags[idx]
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
        start: int = 0,
        self_loss=None,
        self_input_ids=None,
        self_attention_mask=None,
    ):
        if self_loss is not None:
            need_idx = torch.concat(
                [
                    loss[:start] > 0,
                    self_loss[: loss[start:].shape[0]] - loss[start:] > threshold,
                    loss[:1] > 0,
                ]
            )
        else:
            need_idx = torch.concat([loss > threshold, loss[:1] > 0])
        need_idx[end:] = 1
        need_idx[: end - iterative_size] = 1
        loss = loss[need_idx[:-1]]
        if self_loss is not None:
            if need_idx.shape[0] < self_loss.shape[0] + start + 1:
                need_idx = torch.cat(
                    [
                        need_idx,
                        torch.ones(
                            self_loss.shape[0] - need_idx.shape[0] + start + 1,
                            dtype=torch.bool,
                        ).to(need_idx.device),
                    ]
                )
            self_loss = self_loss[need_idx[start:-1]]

        if need_idx.shape[0] < input_ids.shape[1]:
            need_idx = torch.cat(
                [
                    need_idx,
                    torch.ones(
                        input_ids.shape[1] - need_idx.shape[0], dtype=torch.bool
                    ).to(need_idx.device),
                ]
            )
        elif need_idx.shape[0] > input_ids.shape[1]:
            need_idx = need_idx[: input_ids.shape[1]]

        if keep_flag is not None:
            need_idx[keep_flag == 1] = 1
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

        if self_loss is not None:
            self_compressed_input_ids = self_input_ids[self_attention_mask == 1][
                need_idx[start:]
            ].unsqueeze(0)
            self_compressed_attention_mask = self_attention_mask[
                self_attention_mask == 1
            ][need_idx[start:]].unsqueeze(0)
        else:
            self_compressed_input_ids, self_compressed_attention_mask = None, None
        if keep_flag is not None:
            if len(keep_flag) > len(need_idx):
                keep_flag = torch.cat(
                    [
                        keep_flag[:start],
                        keep_flag[start : len(need_idx) + start][need_idx],
                        keep_flag[start + len(need_idx) :],
                    ]
                )
            else:
                keep_flag = keep_flag[need_idx]
        end -= (need_idx[:end] == 0).sum()
        return (
            compressed_input_ids,
            compressed_attention_mask,
            keep_flag,
            end,
            loss,
            self_loss,
            self_compressed_input_ids,
            self_compressed_attention_mask,
        )

    def get_estimate_threshold_base_distribution(
        self, ppl, ratio: float, condition_flag: bool = False
    ):
        ppl = ppl[ppl != 10000]
        target_token = max(0, min(len(ppl) - 1, int(len(ppl) * ratio) - 1))
        return (
            ppl.sort(descending=not condition_flag)
            .values[target_token]
            .detach()
            .cpu()
            .item()
        )

    def iterative_compress_prompt(
        self,
        context: List[str],
        target_token: float,
        iterative_size: int = 200,
        keep_split: bool = False,
        split_token_id: int = 13,
        start: int = 0,
        dynamic_ratio: list = None,
        condition_compare: bool = False,
    ):
        iterative_ratios = self.get_dynamic_compression_ratio(
            context, target_token, iterative_size, dynamic_ratio, start
        )
        context = "\n\n".join(context)
        tokenized_text = self.tokenizer(context, return_tensors="pt")
        input_ids = tokenized_text["input_ids"].to(self.device)
        attention_mask = tokenized_text["attention_mask"].to(self.device)

        N = (attention_mask == 1).sum()
        compressed_input_ids, compressed_attention_mask = input_ids, attention_mask
        if condition_compare:
            self_input_ids, self_attention_mask = (
                input_ids[:, start:],
                attention_mask[:, start:],
            )
            self_compressed_input_ids, self_compressed_attention_mask = (
                self_input_ids,
                self_attention_mask,
            )

        end = min(iterative_size + start, compressed_input_ids.shape[1])
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
            keep_flag = torch.tensor(keep_flag).to(self.device)
        past_key_values, past_loss, ready_end = None, None, 0
        self_past_key_values, self_past_loss, self_ready_end = None, None, 0
        pop_compressed_input_ids, pop_self_compressed_input_ids = None, None
        idx = 0
        while end <= compressed_input_ids.shape[1]:
            if end > self.max_position_embeddings and past_key_values is not None:
                # KV-Cache Compression
                e, s = end - self.max_position_embeddings, self.cache_bos_num
                if pop_compressed_input_ids is None:
                    pop_compressed_input_ids = compressed_input_ids[:, :e]
                else:
                    pop_compressed_input_ids = torch.cat(
                        [pop_compressed_input_ids, compressed_input_ids[:, :e]], dim=-1
                    )
                compressed_input_ids = compressed_input_ids[:, e:]
                compressed_attention_mask = compressed_attention_mask[:, e:]
                past_key_values = [
                    [
                        torch.cat([k[..., :s, :], k[..., s + e :, :]], dim=-2),
                        torch.cat([v[..., :s, :], v[..., s + e :, :]], dim=-2),
                    ]
                    for k, v in past_key_values
                ]
                end, ready_end = end - e, ready_end - e
                if condition_compare:
                    s = min(s, self_past_key_values[0][0].shape[2] - e)
                    self_ready_end -= e
                    if pop_self_compressed_input_ids is None:
                        pop_self_compressed_input_ids = self_compressed_input_ids[:, :e]
                    else:
                        pop_self_compressed_input_ids = torch.cat(
                            [
                                pop_self_compressed_input_ids,
                                self_compressed_input_ids[:, :e],
                            ],
                            dim=-1,
                        )
                    self_compressed_input_ids = self_compressed_input_ids[:, e:]
                    self_compressed_attention_mask = self_compressed_attention_mask[
                        :, e:
                    ]
                    self_past_key_values = [
                        [
                            torch.cat([k[..., :s, :], k[..., s + e :, :]], dim=-2),
                            torch.cat([v[..., :s, :], v[..., s + e :, :]], dim=-2),
                        ]
                        for k, v in self_past_key_values
                    ]

            loss, past_key_values = self.get_ppl(
                "",
                "token",
                compressed_input_ids,
                compressed_attention_mask,
                past_key_values=past_key_values,
                return_kv=True,
                end=end if idx else None,
            )
            if past_loss is not None:
                if end - 1 > len(past_loss):
                    past_loss = torch.cat(
                        [past_loss, torch.zeros_like(loss)[: end - 1 - len(past_loss)]]
                    )
                past_loss[ready_end : end - 1] = loss
                loss = past_loss
            else:
                past_loss = loss
            if idx:
                past_key_values = [
                    [k[:, :, : end - iterative_size], v[:, :, : end - iterative_size]]
                    for k, v in past_key_values
                ]
            else:
                past_key_values = None

            if condition_compare:
                self_loss, self_past_key_values = self.get_ppl(
                    "",
                    "token",
                    self_compressed_input_ids,
                    self_compressed_attention_mask,
                    past_key_values=self_past_key_values,
                    return_kv=True,
                    end=end - start if idx else None,
                )
                if self_past_loss is not None:
                    if end - start - 1 > len(self_past_loss):
                        self_past_loss = torch.cat(
                            [
                                self_past_loss,
                                torch.zeros_like(self_loss)[
                                    : end - 1 - start - len(self_past_loss)
                                ],
                            ]
                        )
                    self_past_loss[self_ready_end : end - start - 1] = self_loss
                    self_loss = self_past_loss
                else:
                    self_past_loss = self_loss
                if idx:
                    self_past_key_values = [
                        [
                            k[:, :, : end - iterative_size - start],
                            v[:, :, : end - iterative_size - start],
                        ]
                        for k, v in self_past_key_values
                    ]
                else:
                    self_past_key_values = None

                self_ready_end = (
                    end - start - iterative_size if not (start and idx == 0) else 0
                )
            ready_end = end - iterative_size if not (start and idx == 0) else 0

            for delta_end, ratio in iterative_ratios[idx]:
                loss = past_loss
                if condition_compare:
                    self_loss = self_past_loss
                    threshold = self.get_estimate_threshold_base_distribution(
                        self_loss[: loss[start:].shape[0]] - loss[start:], ratio, False
                    )
                else:
                    threshold = self.get_estimate_threshold_base_distribution(
                        loss, ratio, False
                    )

                (
                    compressed_input_ids,
                    compressed_attention_mask,
                    keep_flag,
                    end,
                    past_loss,
                    self_past_loss,
                    self_compressed_input_ids,
                    self_compressed_attention_mask,
                ) = self.get_compressed_input(
                    loss,
                    compressed_input_ids,
                    compressed_attention_mask,
                    end - iterative_size + delta_end,
                    iterative_size=delta_end,
                    threshold=threshold,
                    keep_flag=keep_flag,
                    split_token_id=split_token_id,
                    start=start,
                    self_loss=self_loss if condition_compare else None,
                    self_input_ids=self_compressed_input_ids
                    if condition_compare
                    else None,
                    self_attention_mask=self_compressed_attention_mask
                    if condition_compare
                    else None,
                )
                end += iterative_size
            idx += 1
        if pop_compressed_input_ids is not None:
            compressed_input_ids = torch.cat(
                [pop_compressed_input_ids, compressed_input_ids], dim=-1
            )
        return compressed_input_ids[:, start:], compressed_attention_mask[:, start:]

    def recover(
        self,
        original_prompt: str,
        compressed_prompt: str,
        response: str,
    ):
        def match_from_compressed(response_word):
            response_input_ids = self.tokenizer(
                response_word, add_special_tokens=False
            )["input_ids"]
            response_set, response_c = set(response_input_ids), defaultdict(list)
            for idx in range(M):
                if original_input_ids[idx] in response_set:
                    response_c[original_input_ids[idx]].append(idx)
            res, res_min, res_c = None, float("inf"), 1
            n = len(response_input_ids)
            for l in response_c[response_input_ids[0]]:
                x, y, c = 0, l, 1
                for x in range(1, n):
                    idx = bisect.bisect_right(response_c[response_input_ids[x]], y)
                    if (
                        idx >= len(response_c[response_input_ids[x]])
                        or response_c[response_input_ids[x]][idx] - y > 10
                    ):
                        continue
                    c += 1
                    y = response_c[response_input_ids[x]][idx]
                if c > res_c:
                    res_c = c
                    res_min = y - l + 1
                    res = (l, y + 1)
                elif c == res_c and y - l + 1 < res_min:
                    res_min = y - l + 1
                    res = (l, y + 1)

            if res is None:
                return response_word
            # while l > 0 and not self.tokenizer.convert_ids_to_tokens(original_input_ids[l]).startswith("_"):
            #     l -= 1
            # while r < M - 1 and not self.tokenizer.convert_ids_to_tokens(original_input_ids[l]).startswith("_"):
            #     l -= 1
            return self.tokenizer.decode(original_input_ids[res[0] : res[1]])

        response_words = response.split(" ")

        original_input_ids = self.tokenizer(original_prompt, add_special_tokens=False)[
            "input_ids"
        ]
        N, M = len(response_words), len(original_input_ids)
        recovered_response_words = []
        l = 0
        while l < N:
            if response_words[l] not in compressed_prompt:
                recovered_response_words.append(response_words[l])
                l += 1
                continue
            r = l
            while (
                r + 1 < N and " ".join(response_words[l : r + 2]) in compressed_prompt
            ):
                r += 1

            match_words = match_from_compressed(" ".join(response_words[l : r + 1]))
            recovered_response_words.append(match_words)
            l = r + 1
        return " ".join(recovered_response_words)

    def get_rank_results(
        self,
        context: list,
        question: str,
        rank_method: str,
        condition_in_question: str,
        context_tokens_length: list,
    ):
        def get_distance_bm25(corpus, query):
            from rank_bm25 import BM25Okapi

            tokenized_corpus = [doc.split(" ") for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.split(" ")
            doc_scores = bm25.get_scores(tokenized_query)
            idx = [(ii, 0) for ii in (-doc_scores).argsort()]
            return idx

        def get_distance_gzip(corpus, query):
            def get_score(x, y):
                cx, cy = len(gzip.compress(x.encode())), len(gzip.compress(y.encode()))
                cxy = len(gzip.compress(f"{x} {y}".encode()))
                return (cxy - min(cx, cy)) / max(cx, cy)

            import gzip

            doc_scores = [get_score(doc, query) for doc in corpus]
            idx = [(ii, 0) for ii in np.argsort(doc_scores)]
            return idx

        def get_distance_sentbert(corpus, query):
            from sentence_transformers import SentenceTransformer, util

            if self.retrieval_model is None or self.retrieval_model_name != rank_method:
                self.retrieval_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
                self.retrieval_model_name = rank_method
            doc_embeds = self.retrieval_model.encode(corpus)
            query = self.retrieval_model.encode(query)
            doc_scores = -util.dot_score(doc_embeds, query).cpu().numpy().reshape(-1)
            idx = [(ii, 0) for ii in np.argsort(doc_scores)]
            return idx

        def get_distance_openai(corpus, query):
            import openai
            from sentence_transformers import util

            openai.api_key = self.open_api_config.get("api_key", "")
            openai.api_base = self.open_api_config.get(
                "api_base", "https://api.openai.com/v1"
            )
            openai.api_type = self.open_api_config.get("api_type", "open_ai")
            openai.api_version = self.open_api_config.get("api_version", "2023-05-15")
            engine = self.open_api_config.get("engine", "text-embedding-ada-002")

            def get_embed(text):
                return openai.Embedding.create(
                    input=[text.replace("\n", " ")], engine=engine
                )["data"][0]["embedding"]

            doc_embeds = [get_embed(i) for i in corpus]
            query = get_embed(query)
            doc_scores = -util.dot_score(doc_embeds, query).cpu().numpy().reshape(-1)
            idx = [(ii, 0) for ii in np.argsort(doc_scores)]
            return idx

        def get_distance_sentbert_bge(corpus, query):
            from sentence_transformers import SentenceTransformer, util

            if self.retrieval_model is None or self.retrieval_model_name != rank_method:
                self.retrieval_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
                self.retrieval_model_name = rank_method
            doc_embeds = self.retrieval_model.encode(
                [i for i in corpus], normalize_embeddings=True
            )
            query = self.retrieval_model.encode(query, normalize_embeddings=True)
            doc_scores = -util.dot_score(doc_embeds, query).cpu().numpy().reshape(-1)
            idx = [(ii, 0) for ii in np.argsort(doc_scores)]
            return idx

        def get_distance_bge_ranker(corpus, query):
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            pairs = [[i, query] for i in corpus]
            if self.retrieval_model is None or self.retrieval_model_name != rank_method:
                tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
                model = (
                    AutoModelForSequenceClassification.from_pretrained(
                        "BAAI/bge-reranker-large"
                    )
                    .eval()
                    .to(self.device)
                )
                self.retrieval_model = [tokenizer, model]
                self.retrieval_model_name = rank_method
            with torch.no_grad():
                inputs = self.retrieval_model[0](
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                ).to(self.device)
                scores = (
                    self.retrieval_model[1](**inputs, return_dict=True)
                    .logits.view(
                        -1,
                    )
                    .float()
                )
            idx = [(ii, 0) for ii in np.argsort(-scores.cpu())]
            return idx

        def get_distance_bge_llmembedder(corpus, query):
            from transformers import AutoModel, AutoTokenizer

            if self.retrieval_model is None or self.retrieval_model_name != rank_method:
                tokenizer = AutoTokenizer.from_pretrained("BAAI/llm-embedder")
                model = (
                    AutoModel.from_pretrained("BAAI/llm-embedder")
                    .eval()
                    .to(self.device)
                )
                self.retrieval_model = [tokenizer, model]
                self.retrieval_model_name = rank_method

            instruction_qa_query = (
                "Represent this query for retrieving relevant documents: "
            )
            instruction_qa_key = "Represent this document for retrieval: "
            queries = [instruction_qa_query + query for _ in corpus]
            keys = [instruction_qa_key + key for key in corpus]
            with torch.no_grad():
                query_inputs = self.retrieval_model[0](
                    queries,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                ).to(self.device)
                key_inputs = self.retrieval_model[0](
                    keys,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                ).to(self.device)
                query_outputs = self.retrieval_model[1](**query_inputs)
                key_outputs = self.retrieval_model[1](**key_inputs)
                # CLS pooling
                query_embeddings = query_outputs.last_hidden_state[:, 0]
                key_embeddings = key_outputs.last_hidden_state[:, 0]
                # Normalize
                query_embeddings = torch.nn.functional.normalize(
                    query_embeddings, p=2, dim=1
                )
                key_embeddings = torch.nn.functional.normalize(
                    key_embeddings, p=2, dim=1
                )
                similarity = query_embeddings @ key_embeddings.T
            idx = [(ii, 0) for ii in np.argsort(-similarity[0].cpu())]
            return idx

        def get_distance_jinza(corpus, query):
            from numpy.linalg import norm

            from transformers import AutoModel

            def cos_sim(a, b):
                return (a @ b.T) / (norm(a) * norm(b))

            if self.retrieval_model is None or self.retrieval_model_name != rank_method:
                model = (
                    AutoModel.from_pretrained(
                        "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
                    )
                    .eval()
                    .to(self.device)
                )
                self.retrieval_model = model
                self.retrieval_model_name = rank_method

            doc_embeds = self.retrieval_model.encode(corpus)
            query = self.retrieval_model.encode(query)
            doc_scores = cos_sim(doc_embeds, query)
            idx = [(ii, 0) for ii in np.argsort(-doc_scores)]
            return idx

        def get_distance_voyageai(corpus, query):
            import voyageai
            from sentence_transformers import util

            voyageai.api_key = self.open_api_config.get("voyageai_api_key", "")

            def get_embed(text):
                return voyageai.get_embedding(text, model="voyage-01")

            doc_embeds = [get_embed(i) for i in corpus]
            query = get_embed(query)
            doc_scores = -util.dot_score(doc_embeds, query).cpu().numpy().reshape(-1)
            idx = [(ii, 0) for ii in np.argsort(doc_scores)]
            return idx

        def get_distance_cohere(corpus, query):
            import cohere

            api_key = self.open_api_config.get("cohere_api_key", "")
            co = cohere.Client(api_key)
            results = co.rerank(
                model="rerank-english-v2.0", query=query, documents=corpus, top_n=20
            )
            c_map = {jj: ii for ii, jj in enumerate(corpus)}
            doc_rank = [c_map[ii.document["text"]] for ii in results]
            idx = [(ii, 0) for ii in doc_rank]
            return idx

        def get_distance_longllmlingua(corpus, query):
            context_ppl = [
                self.get_condition_ppl(
                    d,
                    query
                    + " We can get the answer to this question in the given documents.",
                    condition_in_question,
                )
                - dl * 2 / 250 * 0
                for d, dl in zip(corpus, context_tokens_length)
            ]
            sort_direct = -1 if condition_in_question == "none" else 1
            ys = sorted(enumerate(context_ppl), key=lambda x: sort_direct * x[1])
            return ys

        method = None
        if rank_method == "bm25":
            method = get_distance_bm25
        elif rank_method == "gzip":
            method = get_distance_gzip
        elif rank_method == "sentbert":
            method = get_distance_sentbert
        elif rank_method == "openai":
            method = get_distance_openai
        elif rank_method in ["longllmlingua", "llmlingua"]:
            method = get_distance_longllmlingua
        elif rank_method == "bge":
            method = get_distance_sentbert_bge
        elif rank_method == "bge_reranker":
            method = get_distance_bge_ranker
        elif rank_method == "bge_llmembedder":
            method = get_distance_bge_llmembedder
        elif rank_method == "jinza":
            method = get_distance_jinza
        elif rank_method == "voyageai":
            method = get_distance_voyageai
        elif rank_method == "cohere":
            method = get_distance_cohere
        return method(context, question)
