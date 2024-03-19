# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import spacy
import torch
from torch.utils.data import Dataset


class TokenClfDataset(Dataset):
    def __init__(
        self,
        texts,
        labels=None,
        max_len=512,
        tokenizer=None,
        model_name="bert-base-multilingual-cased",
    ):
        self.len = len(texts)
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels
        self.model_name = model_name
        if "bert-base-multilingual-cased" in model_name:
            self.cls_token = "[CLS]"
            self.sep_token = "[SEP]"
            self.unk_token = "[UNK]"
            self.pad_token = "[PAD]"
            self.mask_token = "[MASK]"
        elif "xlm-roberta-large" in model_name:
            self.bos_token = "<s>"
            self.eos_token = "</s>"
            self.sep_token = "</s>"
            self.cls_token = "<s>"
            self.unk_token = "<unk>"
            self.pad_token = "<pad>"
            self.mask_token = "<mask>"
        else:
            raise NotImplementedError()

        self.nlp = spacy.load("en_core_web_sm")

    def __getitem__(self, index):
        text = self.texts[index]
        if self.labels is not None:
            labels = self.labels[index][:]
            tokenized_text, labels = self.tokenize_and_preserve_labels(
                text, labels, self.tokenizer
            )
            assert len(tokenized_text) == len(labels)
            labels.insert(0, False)
            labels.insert(-1, False)
        else:
            tokenized_text = self.tokenizer.tokenize(text)

        tokenized_text = [self.cls_token] + tokenized_text + [self.sep_token]

        if len(tokenized_text) > self.max_len:
            tokenized_text = tokenized_text[: self.max_len]
            if self.labels is not None:
                labels = labels[: self.max_len]
        else:
            tokenized_text = tokenized_text + [
                self.pad_token for _ in range(self.max_len - len(tokenized_text))
            ]
            if self.labels is not None:
                labels = labels + [False for _ in range(self.max_len - len(labels))]

        attn_mask = [1 if tok != self.pad_token else 0 for tok in tokenized_text]

        ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        sample = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(attn_mask, dtype=torch.long),
        }
        if self.labels is not None:
            sample["targets"] = torch.tensor(labels, dtype=torch.long)

        return sample

    def __len__(self):
        return self.len

    def split_string(self, input_string, ignore_tokens=set([","])):
        doc = self.nlp(input_string)
        word_list = []
        for word in doc:
            if word.lemma_ not in ignore_tokens:
                word_list.append(word.lemma_)
        return word_list

    def tokenize_and_preserve_labels(self, text, text_labels, tokenizer):
        """
        Word piece tokenization makes it difficult to match word labels
        back up with individual word pieces. This function tokenizes each
        word one at a time so that it is easier to preserve the correct
        label for each subword. It is, of course, a bit slower in processing
        time, but it will help our model achieve higher accuracy.
        """

        tokenized_text = []
        labels = []

        assert len(self.split_string(text)) == len(text_labels)

        for word, label in zip(self.split_string(text), text_labels):
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_text.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)

        return tokenized_text, labels
