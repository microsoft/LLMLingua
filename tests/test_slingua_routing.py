# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Unit tests for issue #235: use_slingua=True should route compress_prompt
to compress_prompt_llmlingua2, just like use_llmlingua2=True does.

These tests use sys.modules patching to avoid requiring ML dependencies
(torch, transformers) to be installed.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock


def _install_mock_modules():
    """Insert lightweight stubs for heavy ML dependencies before importing llmlingua.

    Note: patching sys.modules here is process-local. pytest --dist=loadfile
    (used in the Makefile) ensures this file runs in its own worker, so the
    stubs do not leak into integration tests that load real models.
    """
    for mod_name in [
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "torch.utils",
        "torch.utils.data",
        "transformers",
        "numpy",
        "numpy.linalg",
        "nltk",
        "nltk.tokenize",
        "accelerate",
        "tiktoken",
        "yaml",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()


_install_mock_modules()

# Now the import will succeed with the stubs in place
from llmlingua.prompt_compressor import PromptCompressor  # noqa: E402


class SLinguaRoutingTester(unittest.TestCase):
    """
    Verifies that compress_prompt routes to compress_prompt_llmlingua2 when
    use_slingua=True, without requiring a real model download.
    """

    def _make_compressor(self, use_llmlingua2=False, use_slingua=False):
        """Return a PromptCompressor with model loading stubbed out."""
        compressor = PromptCompressor.__new__(PromptCompressor)
        compressor.use_llmlingua2 = use_llmlingua2
        compressor.use_slingua = use_slingua
        compressor.oai_tokenizer = MagicMock()
        return compressor

    def test_use_slingua_routes_to_llmlingua2(self):
        """compress_prompt must call compress_prompt_llmlingua2 when use_slingua=True."""
        compressor = self._make_compressor(use_slingua=True)
        fake_result = {"compressed_prompt": "test", "origin_tokens": 10}
        compressor.compress_prompt_llmlingua2 = MagicMock(return_value=fake_result)

        result = compressor.compress_prompt(["hello world"], rate=0.5)

        compressor.compress_prompt_llmlingua2.assert_called_once()
        self.assertEqual(result, fake_result)

    def test_use_llmlingua2_routes_to_llmlingua2(self):
        """Existing behaviour: compress_prompt routes to compress_prompt_llmlingua2 when use_llmlingua2=True."""
        compressor = self._make_compressor(use_llmlingua2=True)
        fake_result = {"compressed_prompt": "test", "origin_tokens": 10}
        compressor.compress_prompt_llmlingua2 = MagicMock(return_value=fake_result)

        result = compressor.compress_prompt(["hello world"], rate=0.5)

        compressor.compress_prompt_llmlingua2.assert_called_once()
        self.assertEqual(result, fake_result)

    def test_neither_flag_does_not_route_to_llmlingua2(self):
        """When neither use_slingua nor use_llmlingua2 is set, compress_prompt_llmlingua2 is NOT called."""
        compressor = self._make_compressor()
        compressor.compress_prompt_llmlingua2 = MagicMock()

        # The LLMLingua-1 path will raise because model internals are stubs;
        # what matters is that compress_prompt_llmlingua2 was never entered.
        try:
            compressor.compress_prompt(["hello world"], rate=0.5)
        except Exception:
            pass

        compressor.compress_prompt_llmlingua2.assert_not_called()


if __name__ == "__main__":
    unittest.main()
