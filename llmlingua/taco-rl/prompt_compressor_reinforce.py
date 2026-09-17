# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import re
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from typing import List
from llmlingua.prompt_compressor import PromptCompressor
from utils import TokenClfDataset, get_pure_token, is_begin_of_new_word, replace_added_token

class PromptCompressorReinforce(PromptCompressor):
    """
    PromptCompressorReinforce extends PromptCompressor with reinforcement learning capabilities.
    
    This class overrides the compression methods to track actions, log probabilities, and entropy
    for reinforcement learning training. It maintains the same interface as PromptCompressor
    but provides additional RL-specific outputs.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actions = []
        self.old_log_probs = []
        self.old_logits = None
        self.entropy = []
        self.compressed_prompt_list_2 = []
        self.compression_ratios = []
    
    def compress_prompt_llmlingua2(
        self,
        context: List[str],
        rate: float = 0.5,
        target_token: int = -1,
        use_context_level_filter: bool = False,
        use_token_level_filter: bool = True,
        target_context: int = -1,
        context_level_rate: float = 1.0,
        context_level_target_token: int = -1,
        force_context_ids: List[int] = [],
        return_word_label: bool = False,
        word_sep: str = "\t\t|\t\t",
        label_sep: str = " ",
        token_to_word: str = "mean",
        force_tokens: List[str] = [],
        force_reserve_digit: bool = False,
        drop_consecutive: bool = False,
        chunk_end_tokens: List[str] = [".", "\n"],
    ):
        """
        Override the llmlingua2 compression method to track RL-specific information.
        """
        assert len(force_tokens) <= self.max_force_token
        token_map = {}
        for i, t in enumerate(force_tokens):
            if len(self.tokenizer.tokenize(t)) != 1:
                token_map[t] = self.added_tokens[i]
        chunk_end_tokens = copy.deepcopy(chunk_end_tokens)
        for c in chunk_end_tokens:
            if c in token_map:
                chunk_end_tokens.append(token_map[c])
        chunk_end_tokens = set(chunk_end_tokens)

        if type(context) == str:
            context = [context]
        context = copy.deepcopy(context)

        if len(context) == 1 and use_context_level_filter:
            use_context_level_filter = False

        n_original_token = 0
        context_chunked = []
        for i in range(len(context)):
            n_original_token += self.get_token_length(
                context[i], use_oai_tokenizer=True
            )
            for ori_token, new_token in token_map.items():
                context[i] = context[i].replace(ori_token, new_token)
            context_chunked.append(
                self.__chunk_context(context[i], chunk_end_tokens=chunk_end_tokens)
            )

        org_context = None
        if use_context_level_filter:
            # want use_context_level_filter but do not specify any parameters in context level?
            # we will set context_level_rate = (rate + 1.0) / 2 if specify rate or target_token * 2 if specify target_token
            if (
                target_context <= 0
                and context_level_rate >= 1.0
                and context_level_target_token <= 0
            ):
                if target_token < 0 and rate < 1.0:
                    context_level_rate = (
                        (rate + 1.0) / 2 if use_token_level_filter else rate
                    )
                if target_token >= 0:
                    context_level_target_token = (
                        target_token * 2 if use_token_level_filter else target_token
                    )

            if target_context >= 0:
                context_level_rate = min(target_context / len(context), 1.0)
            if context_level_target_token >= 0:
                context_level_rate = min(
                    context_level_target_token / n_original_token, 1.0
                )

            context_probs, context_words = self.__get_context_prob(
                context_chunked,
                token_to_word=token_to_word,
                force_tokens=force_tokens,
                token_map=token_map,
                force_reserve_digit=force_reserve_digit,
            )

            threshold = np.percentile(
                context_probs, int(100 * (1 - context_level_rate))
            )

            reserved_context = []
            context_label = [False] * len(context_probs)
            for i, p in enumerate(context_probs):
                if p >= threshold or (
                    force_context_ids is not None and i in force_context_ids
                ):
                    reserved_context.append(context_chunked[i])
                    context_label[i] = True
            n_reserved_token = 0
            for chunks in reserved_context:
                for c in chunks:
                    n_reserved_token += self.get_token_length(c, use_oai_tokenizer=True)
            if target_token >= 0:
                rate = min(target_token / n_reserved_token, 1.0)

            org_context = copy.deepcopy(reserved_context)

            if use_token_level_filter:
                compressed_context, word_list, word_label_list, actions, old_log_probs, old_logits, compressed_prompt_list_2, entropy = self.__compress(
                    reserved_context,
                    reduce_rate=max(0, 1 - rate),
                    token_to_word=token_to_word,
                    force_tokens=force_tokens,
                    token_map=token_map,
                    force_reserve_digit=force_reserve_digit,
                    drop_consecutive=drop_consecutive,
                )
            else:
                compressed_context, word_list, word_label_list, actions, old_log_probs, old_logits, compressed_prompt_list_2, entropy = self.__compress(
                    reserved_context,
                    reduce_rate=0,
                    token_to_word=token_to_word,
                    force_tokens=force_tokens,
                    token_map=token_map,
                    force_reserve_digit=force_reserve_digit,
                    drop_consecutive=drop_consecutive,
                )

            n_compressed_token = 0
            compression_ratios=[]
            for c,o in zip(compressed_prompt_list_2,org_context):
                org_token_length = self.get_token_length(o, use_oai_tokenizer=True)
                compressed_token_length = self.get_token_length(c, use_oai_tokenizer=True)
                fin_dict = {"original":org_token_length, "compressed":compressed_token_length, "original_context":o, "compressed_context":c}
                compression_ratios.append(fin_dict)
            for c in compressed_context:
                n_compressed_token += self.get_token_length(c, use_oai_tokenizer=True)
            saving = (n_original_token - n_compressed_token) * 0.06 / 1000
            ratio = (
                1 if n_compressed_token == 0 else n_original_token / n_compressed_token
            )
            res = {
                "compressed_prompt": "\n\n".join(compressed_context),
                "compressed_prompt_list": compressed_context,
                "compression_ratios":compression_ratios,
                "compressed_prompt_list_2":compressed_prompt_list_2,
                "entropy": entropy,
                "origin_tokens": n_original_token,
                "compressed_tokens": n_compressed_token,
                "ratio": f"{ratio:.1f}x",
                "rate": f"{1 / ratio * 100:.1f}%",
                "saving": f", Saving ${saving:.1f} in GPT-4.",
                "actions": actions,
                "old_log_probs": old_log_probs,
                "old_logits": old_logits,
            }
            if return_word_label:
                words = []
                labels = []
                j = 0
                for i in range(len(context)):
                    if context_label[i]:
                        words.extend(word_list[j])
                        labels.extend(word_label_list[j])
                        j += 1
                    else:
                        words.extend(context_words[i])
                        labels.extend([0] * len(context_words[i]))
                word_label_lines = word_sep.join(
                    [f"{word}{label_sep}{label}" for word, label in zip(words, labels)]
                )
                res["fn_labeled_original_prompt"] = word_label_lines
            return res

        if target_token > 0:
            rate = min(target_token / n_original_token, 1.0)

        org_context = copy.deepcopy(context_chunked)

        if use_token_level_filter:
            compressed_context, word_list, word_label_list, actions, old_log_probs, old_logits, compressed_prompt_list_2, entropy = self.__compress(
                context_chunked,
                reduce_rate=max(0, 1 - rate),
                token_to_word=token_to_word,
                force_tokens=force_tokens,
                token_map=token_map,
                force_reserve_digit=force_reserve_digit,
                drop_consecutive=drop_consecutive,
            )
        else:
            compressed_context, word_list, word_label_list, actions, old_log_probs, old_logits, compressed_prompt_list_2, entropy = self.__compress(
                context_chunked,
                reduce_rate=0,
                token_to_word=token_to_word,
                force_tokens=force_tokens,
                token_map=token_map,
                force_reserve_digit=force_reserve_digit,
                drop_consecutive=drop_consecutive,
            )

        n_compressed_token = 0
        compression_ratios=[]
        for c,o in zip(compressed_prompt_list_2, org_context):
            org_token_length = self.get_token_length(o[0], use_oai_tokenizer=True)
            compressed_token_length = self.get_token_length(c, use_oai_tokenizer=True)
            fin_dict = {"original":org_token_length, "compressed":compressed_token_length, "original_context":o[0], "compressed_context":c}
            compression_ratios.append(fin_dict)
        for c in compressed_context:
            n_compressed_token += self.get_token_length(c, use_oai_tokenizer=True)
        saving = (n_original_token - n_compressed_token) * 0.06 / 1000
        ratio = 1 if n_compressed_token == 0 else n_original_token / n_compressed_token
        res = {
            "compressed_prompt": "\n\n".join(compressed_context),
            "compressed_prompt_list": compressed_context,
            "compression_ratios":compression_ratios,
            "compressed_prompt_list_2":compressed_prompt_list_2,
            "entropy": entropy,
            "origin_tokens": n_original_token,
            "compressed_tokens": n_compressed_token,
            "ratio": f"{ratio:.1f}x",
            "rate": f"{1 / ratio * 100:.1f}%",
            "saving": f", Saving ${saving:.1f} in GPT-4.",
            "actions": actions,
            "old_log_probs": old_log_probs,
            "old_logits": old_logits,
        }
        if return_word_label:
            words = []
            labels = []
            for w_list, l_list in zip(word_list, word_label_list):
                words.extend(w_list)
                labels.extend(l_list)

            word_label_lines = word_sep.join(
                [f"{word}{label_sep}{label}" for word, label in zip(words, labels)]
            )
            res["fn_labeled_original_prompt"] = word_label_lines
        return res

    def __chunk_context(self, origin_text, chunk_end_tokens):
        # leave 2 token for CLS and SEP
        max_len = self.max_seq_len - 2
        origin_list = []
        origin_tokens = self.tokenizer.tokenize(origin_text)
        n = len(origin_tokens)
        st = 0
        while st < n:
            if st + max_len > n - 1:
                chunk = self.tokenizer.convert_tokens_to_string(origin_tokens[st:n])
                origin_list.append(chunk)
                break
            else:
                ed = st + max_len
                for j in range(0, ed - st):
                    if origin_tokens[ed - j] in chunk_end_tokens:
                        ed = ed - j
                        break
                chunk = self.tokenizer.convert_tokens_to_string(
                    origin_tokens[st : ed + 1]
                )
                origin_list.append(chunk)
                st = ed + 1
        return origin_list

    def __merge_token_to_word_for_actions(
        self, tokens, token_actions, force_tokens, token_map, force_reserve_digit
    ):
        """
        Merge tokens to words while tracking actions for RL training.
        """
        words = []
        word_actions = []
        word_actions_no_force = []

        for token, action in zip(tokens, token_actions):
            if token in self.special_tokens:
                continue
            # add a new word
            elif is_begin_of_new_word(token, self.model_name, force_tokens, token_map):
                pure_token = get_pure_token(token, self.model_name)
                action_no_force = action
                if pure_token in force_tokens or pure_token in set(token_map.values()):
                    action = 1
                token = replace_added_token(token, token_map)
                words.append(token)
                word_actions.append(
                    [
                        1
                        if force_reserve_digit and bool(re.search(r"\d", token))
                        else action
                    ]
                )
                word_actions_no_force.append([action_no_force])
            # concatenate with previous token
            else:
                pure_token = get_pure_token(token, self.model_name)
                words[-1] += pure_token
                word_actions[-1].append(
                    1
                    if force_reserve_digit and bool(re.search(r"\d", token))
                    else action
                )
                word_actions_no_force[-1].append(action)

        return words, word_actions, word_actions_no_force

    def __token_action_to_word_action(self, token_actions, convert_mode="all"):
        """
        Convert token actions to word actions for RL training.
        """
        if convert_mode == "all":
            word_action = [all(p) for p in token_actions]
        elif convert_mode == "any":
            word_action = [any(p) for p in token_actions]
        else:
            raise NotImplementedError()

        return word_action

    def __compress(
        self,
        context_list: list,
        reduce_rate: float = 0.5,
        token_to_word: str = "mean",
        force_tokens: List[str] = [],
        token_map: dict = {},
        force_reserve_digit: bool = False,
        drop_consecutive: bool = False,
    ):
        """
        Override the compression method to track RL-specific information.
        """
        def split_string_to_words(input_string):
            pattern = r'\b\w+\b|[<>=/!@#$%^&*()?":{}|\\`~;_+-]'
            result = re.findall(pattern, input_string)
            return result

        if reduce_rate <= 0:
            words, word_labels = [], []
            for i in range(len(context_list)):
                chunk_list = context_list[i]
                chunk_words = []
                chunk_word_labels = []
                for j in range(len(chunk_list)):
                    # replace to original token
                    for ori_token, new_token in token_map.items():
                        chunk_list[j] = chunk_list[j].replace(new_token, ori_token)
                    ws = split_string_to_words(chunk_list[j])
                    chunk_words.extend(ws)
                    chunk_word_labels.extend([1 for _ in range(len(ws))])
                context_list[i] = "".join(chunk_list)
                words.append(chunk_words)
                word_labels.append(chunk_word_labels)
            return context_list, words, word_labels

        chunk_list = []
        for chunks in context_list:
            for c in chunks:
                chunk_list.append(c)

        dataset = TokenClfDataset(
            chunk_list, tokenizer=self.tokenizer, max_len=self.max_seq_len
        )
        dataloader = DataLoader(
            dataset, batch_size=self.max_batch_size, shuffle=False, drop_last=False
        )

        compressed_chunk_list = []
        word_list = []
        word_label_list = []
        actions, old_log_probs, old_logits, entropy = [], [], None, []
        compressed_prompt_list_2 = []
        
        for batch in dataloader:
            ids = batch["ids"].type(torch.long).to(self.model.device)
            mask = batch["mask"].type(torch.long).to(self.model.device) == 1

            outputs = self.model(input_ids=ids, attention_mask=mask)
            
            loss, logits = outputs.loss, outputs.logits
            probs = F.softmax(logits, dim=-1)

            old_probs = probs
            for logits_, old_probs_ in zip(logits, old_probs):
                dist = Categorical(probs=old_probs_)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                actions.append(action)
                old_log_probs.append(log_prob)
                entropy.append(dist.entropy())

            # Create compressed prompt list from actions
            for j in range(ids.shape[0]):
                chunk_actions = actions[j]
                chunk_ids = ids[j]
                chunk_mask = mask[j]
                active_actions = torch.masked_select(chunk_actions, chunk_mask)
                active_ids = torch.masked_select(chunk_ids, chunk_mask)

                tokens = self.tokenizer.convert_ids_to_tokens(
                    active_ids.squeeze().tolist()
                )
                token_actions = [action for action in active_actions.cpu().detach().numpy()]

                words, valid_token_actions, _ = self.__merge_token_to_word_for_actions(
                    tokens=tokens,
                    token_actions=token_actions,
                    force_tokens=force_tokens,
                    token_map=token_map,
                    force_reserve_digit=force_reserve_digit,
                )
                word_actions = self.__token_action_to_word_action(
                    valid_token_actions, convert_mode="all"
                )

                if drop_consecutive:
                    is_token_between = False
                    prev = None
                    for i, (word, word_action) in enumerate(zip(words, word_actions)):
                        if word in force_tokens:
                            if is_token_between:
                                is_token_between = False
                            elif not is_token_between and word == prev:
                                word_actions[i] = 0
                            prev = word
                        else:
                            is_token_between |= word_action
                
                keep_words = []
                for word, word_action in zip(words, word_actions):
                    if word_action:
                        keep_words.append(word)
                keep_str = self.tokenizer.convert_tokens_to_string(keep_words)
                compressed_prompt_list_2.append(keep_str)

        compressed_context_list = []
        original_word_list = []
        original_word_label_list = []
        prev_idx = 0
        for chunk_list in context_list:
            n_chunk = len(chunk_list)
            compressed_context_list.append(
                "".join(compressed_prompt_list_2[prev_idx : prev_idx + n_chunk])
            )
            original_word_list.append([])
            original_word_label_list.append([])
            prev_idx = prev_idx + n_chunk

        return compressed_context_list, original_word_list, original_word_label_list, actions, old_log_probs, old_logits, compressed_prompt_list_2, entropy

    def get_rl_info(self):
        """
        Get reinforcement learning information for training.
        
        Returns:
            dict: Dictionary containing RL-specific information including actions, log probabilities, and entropy.
        """
        return {
            "actions": self.actions,
            "old_log_probs": self.old_log_probs,
            "old_logits": self.old_logits,
            "entropy": self.entropy,
            "compressed_prompt_list_2": self.compressed_prompt_list_2,
            "compression_ratios": self.compression_ratios
        }

    def clear_rl_info(self):
        """
        Clear stored reinforcement learning information.
        """
        self.actions = []
        self.old_log_probs = []
        self.old_logits = None
        self.entropy = []
        self.compressed_prompt_list_2 = []
        self.compression_ratios = []
    