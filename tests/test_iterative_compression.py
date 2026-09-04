# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import unittest

import torch

from llmlingua import PromptCompressor


class _WindowTokenizer:
    def __init__(self, length):
        self.length = length

    def __call__(self, *args, **kwargs):
        input_ids = torch.arange(self.length, dtype=torch.long).unsqueeze(0) % 10
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class IterativeCompressionTester(unittest.TestCase):
    def test_long_prompt_keeps_loss_and_token_windows_aligned(self):
        length, max_position_embeddings, iterative_size = 50, 16, 15
        compressor = PromptCompressor.__new__(PromptCompressor)
        compressor.tokenizer = _WindowTokenizer(length)
        compressor.device = "cpu"
        compressor.max_position_embeddings = max_position_embeddings
        compressor.cache_bos_num = 2

        ratios = [[(iterative_size - 1, 0.5)]] + [[(iterative_size, 0.5)]] * 10
        compressor.get_dynamic_compression_ratio = lambda *args: ratios
        compressor.get_estimate_threshold_base_distribution = lambda *args: 0.5

        def get_ppl(
            text,
            granularity="sentence",
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            return_kv=False,
            end=None,
            **kwargs,
        ):
            past_length = (
                0 if past_key_values is None else past_key_values[0][0].shape[2]
            )
            end = input_ids.shape[1] if end is None else int(end)
            end = min(end, past_length + max_position_embeddings)
            loss = torch.tensor(
                [
                    ((position * 1103515245) % 1000) / 1000
                    for position in range(past_length + 1, end)
                ],
                dtype=torch.float32,
            )
            cache = [
                [
                    torch.zeros(1, 1, end, 1),
                    torch.zeros(1, 1, end, 1),
                ]
            ]
            return loss, cache

        compressor.get_ppl = get_ppl
        compressed_ids, compressed_mask = compressor.iterative_compress_prompt(
            ["prompt"],
            target_token=100,
            iterative_size=iterative_size,
            dynamic_ratio=[0.0],
        )

        self.assertEqual(compressed_ids.shape, compressed_mask.shape)
        self.assertGreater(compressed_ids.shape[1], 0)
        self.assertLess(compressed_ids.shape[1], length)


if __name__ == "__main__":
    unittest.main()
