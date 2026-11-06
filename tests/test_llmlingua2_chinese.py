# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""End-to-end tests for LLMLingua-2 Chinese (中文) prompt compression."""

import re
import unittest

from transformers import AutoTokenizer

from llmlingua import PromptCompressor
from llmlingua.utils import (
    get_pure_token,
    is_begin_of_new_word,
    remove_space_between_cjk,
)


class LLMLingua2ChineseTester(unittest.TestCase):
    """
    End2end tests validating LLMLingua-2 compression behavior on Chinese text.

    The XLM-RoBERTa based llmlingua-2 model is multilingual, so these tests
    exercise Chinese input and assert on observable compression behavior
    (token reduction, structural preservation, force_tokens, digit reserve).
    """

    # A single Chinese news article (医药、卫生 category).
    ZH_PROMPT = (
        "新闻内容：\n（服务·健康）专家提醒：寒冷气候易诱发心脑血管疾病\n"
        "新华社海口２月９日专电（张苏民、李建国）海口市疾病预防控制中心专家介绍，"
        "持续的寒冷气候是心脑血管疾病的杀手，尤其患有高血压或高血脂疾病的老人更应做好防范，"
        "防止脑中风发生。\n"
        "　　在寒冷的气候环境当中要注意保暖，增添衣服，饮食以清淡为主，多食用蔬菜，忌暴食荤类。"
        "尤其过年时，切忌熬夜，平时要加强身体锻炼，劳逸结合。\n"
        "类别：医药、卫生"
    )

    # Two Chinese contexts for multi-context (coarse-to-fine) compression.
    ZH_MULTI_CONTEXT = [
        "新闻内容：\n第38届世界贸易中心年会将于2007年10月21至24日在美国新奥尔良召开，"
        "届时将有来自60多个国家和地区的经贸代表团约600余人与会。\n类别：商业、外贸、海关",
        "新闻内容：\n第十一届全国运动会将于今年10月16日在济南奥体中心开幕，"
        "闭幕时间为10月28日，比赛项目共设33个大项、43个分项、362个小项。\n类别：体育",
    ]

    ZH_WITH_DIGITS = (
        "会议定于2024年3月15日上午9点30分召开，预计持续2小时，参会人数约120人。"
    )

    @classmethod
    def setUpClass(cls):
        cls.llmlingua = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            device_map="cpu",
            use_llmlingua2=True,
        )

    def test_chinese_single_context_reduces_tokens(self):
        # Arrange
        rate = 0.5
        # Act
        result = self.llmlingua.compress_prompt(
            self.ZH_PROMPT,
            rate=rate,
            force_tokens=["\n", "。", "，", "：", "、"],
            force_reserve_digit=False,
            drop_consecutive=False,
        )
        # Assert
        self.assertIn("compressed_prompt", result)
        self.assertGreater(result["origin_tokens"], 0)
        self.assertGreater(result["compressed_tokens"], 0)
        self.assertLess(
            result["compressed_tokens"],
            result["origin_tokens"],
            "Compressed token count must be smaller than the original.",
        )
        self.assertGreater(
            len(result["compressed_prompt"].strip()),
            0,
            "Compressed prompt must not be empty.",
        )
        self.assertTrue(
            any("\u4e00" <= ch <= "\u9fff" for ch in result["compressed_prompt"]),
            "Compressed prompt should still contain Chinese characters.",
        )

    def test_chinese_target_token(self):
        # Arrange
        target = 60
        # Act
        result = self.llmlingua.compress_prompt(
            self.ZH_PROMPT,
            target_token=target,
            force_tokens=["\n", "。", "，", "：", "、"],
            force_reserve_digit=False,
            drop_consecutive=False,
        )
        # Assert: target_token overrides rate; result should be at/near target.
        self.assertLess(result["compressed_tokens"], result["origin_tokens"])
        self.assertLessEqual(
            result["compressed_tokens"],
            target + 40,
            f"compressed_tokens={result['compressed_tokens']} far exceeds target={target}",
        )

    def test_chinese_force_tokens_preserved(self):
        # Arrange: force the period and full-width colon to always be kept.
        # Note: XLM-RoBERTa's tokenizer normalizes the full-width colon "："
        # to the half-width ":", so force_tokens and assertions use ":".
        # Act
        result = self.llmlingua.compress_prompt(
            self.ZH_PROMPT,
            rate=0.4,
            force_tokens=["\n", "。", ":"],
            force_reserve_digit=False,
            drop_consecutive=False,
        )
        # Assert: forced punctuation present in original stays in output.
        self.assertIn(":", result["compressed_prompt"])
        self.assertIn("。", result["compressed_prompt"])

    def test_chinese_force_reserve_digit(self):
        # force_reserve_digit keeps the digits that belong to *retained*
        # tokens intact (it prevents a kept numeric token from being split);
        # it does not force every digit in the input to survive compression.
        # A near-lossless rate keeps the whole numeric sentence, so all input
        # digits must then be present in the output.
        # Act
        result = self.llmlingua.compress_prompt(
            self.ZH_WITH_DIGITS,
            rate=0.9,
            force_tokens=["，", "。"],
            force_reserve_digit=True,
            drop_consecutive=False,
        )
        # Assert: at a near-lossless rate every input digit is preserved.
        digits_in_input = {c for c in self.ZH_WITH_DIGITS if c.isdigit()}
        digits_in_output = {c for c in result["compressed_prompt"] if c.isdigit()}
        self.assertTrue(
            digits_in_input.issubset(digits_in_output),
            f"Digits dropped. input={sorted(digits_in_input)} "
            f"output={sorted(digits_in_output)}",
        )

    def test_chinese_multi_context_level_filter(self):
        # Act
        result = self.llmlingua.compress_prompt(
            self.ZH_MULTI_CONTEXT,
            target_token=80,
            use_context_level_filter=True,
            force_tokens=["\n", "。", "，", "：", "、"],
            force_reserve_digit=False,
            drop_consecutive=True,
        )
        # Assert
        self.assertLess(result["compressed_tokens"], result["origin_tokens"])
        self.assertGreater(len(result["compressed_prompt"].strip()), 0)
        self.assertIn("compressed_prompt_list", result)

    def test_chinese_rate_reported(self):
        # Act
        result = self.llmlingua.compress_prompt(
            self.ZH_PROMPT,
            rate=0.5,
            force_tokens=["\n", "。", "，", "：", "、"],
        )
        # Assert: human-readable ratio/rate fields are well-formed.
        self.assertTrue(result["ratio"].endswith("x"))
        self.assertTrue(result["rate"].endswith("%"))

    def test_chinese_drop_consecutive_removes_punct_runs(self):
        # Forcing "，" while compressing aggressively can leave runs of
        # orphaned punctuation (e.g. "，，，"). drop_consecutive=True must
        # collapse such runs so the compressed Chinese text stays readable.
        force = ["\n", "。", "，", "：", "、"]

        without = self.llmlingua.compress_prompt(
            self.ZH_PROMPT,
            rate=0.4,
            force_tokens=force,
            drop_consecutive=False,
        )["compressed_prompt"]
        with_dc = self.llmlingua.compress_prompt(
            self.ZH_PROMPT,
            rate=0.4,
            force_tokens=force,
            drop_consecutive=True,
        )["compressed_prompt"]

        # Longest run of consecutive Chinese punctuation in each output.
        punct = r"[，,。：:、]"
        run_re = re.compile(punct + r"\s*(?:" + punct + r"\s*)+")

        def longest_run(text):
            spans = [len(re.sub(r"\s+", "", m.group())) for m in run_re.finditer(text)]
            return max(spans) if spans else 0

        # Assert: enabling drop_consecutive must not increase punctuation runs,
        # and must keep the longest run short (<= 2, i.e. no "，，，" garbage).
        self.assertLessEqual(
            longest_run(with_dc),
            longest_run(without),
            f"drop_consecutive worsened punct runs: "
            f"with={with_dc!r} without={without!r}",
        )
        self.assertLessEqual(
            longest_run(with_dc),
            2,
            f"drop_consecutive left a long punctuation run: {with_dc!r}",
        )

    def test_chinese_no_extra_spaces_between_cjk(self):
        # Regression for microsoft/LLMLingua#131:
        # "When I use Chinese prompt, the compressed prompt has extra spaces."
        # LLMLingua-2 rejoins kept tokens with a single ASCII space (English
        # convention). For Chinese there is no word delimiter, so a space
        # inserted between two CJK characters (e.g. "长 短") is a spurious
        # artifact and should not appear in the compressed output.
        result = self.llmlingua.compress_prompt(
            self.ZH_PROMPT,
            rate=0.5,
            force_tokens=["\n", "。", "，", "：", "、"],
            drop_consecutive=True,
        )
        out = result["compressed_prompt"]
        cjk = "[\u4e00-\u9fff]"
        bad = re.findall(cjk + " " + cjk, out)
        self.assertEqual(
            bad,
            [],
            f"Extra ASCII space inserted between CJK characters "
            f"(LLMLingua#131). offending={bad} output={out!r}",
        )


class LLMLingua131RegressionTester(unittest.TestCase):
    """
    Reproduction test for microsoft/LLMLingua#131 using only the BERT
    multilingual tokenizer (no model weights, small download).

    Issue #131: "When I use Chinese prompt, the compressed prompt has extra
    spaces." The reporter used a bert-base-multilingual-cased based model.

    Root cause: BERT's tokenizer splits every CJK character into its own
    token with no "##" prefix, so __merge_token_to_word treats each character
    as a separate word, and convert_tokens_to_string(keep_words) rejoins the
    single-character words with ASCII spaces (English word convention),
    yielding "持 续 的 寒 冷 ...". XLM-RoBERTa (SentencePiece) does not exhibit
    this, which is why the model-backed tests above do not catch it.

    This test mirrors the token -> word -> convert_tokens_to_string path of
    PromptCompressor.__compress and asserts the bug is present, pinning the
    defect so a fix can be validated against it.
    """

    MODEL_NAME = "bert-base-multilingual-cased"

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.MODEL_NAME)

    def _merge_tokens_to_words(self, tokens):
        # Mirror PromptCompressor.__merge_token_to_word (no probs, no force).
        words = []
        for token in tokens:
            if is_begin_of_new_word(token, self.MODEL_NAME, [], {}):
                words.append(get_pure_token(token, self.MODEL_NAME))
            else:
                words[-1] += get_pure_token(token, self.MODEL_NAME)
        return words

    def test_bert_reproduces_cjk_extra_spaces(self):
        # Arrange: a Chinese sentence with only CJK characters + punctuation.
        text = "持续的寒冷气候是心脑血管疾病的杀手"
        encoded = self.tokenizer(text, add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"])

        # Act: reproduce the __compress rejoin path.
        words = self._merge_tokens_to_words(tokens)
        keep_str = self.tokenizer.convert_tokens_to_string(words)

        # Assert: the bug is reproduced -- ASCII spaces appear between CJK
        # characters. Each CJK char is its own word, so the joined string
        # contains "字 字" pairs that were absent from the input.
        cjk = "[\u4e00-\u9fff]"
        bad = re.findall(cjk + " " + cjk, keep_str)
        self.assertTrue(
            bad,
            f"Expected to reproduce LLMLingua#131 (extra spaces between CJK "
            f"characters) but found none. output={keep_str!r}",
        )
        # And the corrupted output must differ from the original text.
        self.assertNotEqual(
            keep_str,
            text,
            f"Expected corrupted output to differ from input. "
            f"output={keep_str!r}",
        )

    def test_fix_removes_cjk_extra_spaces(self):
        # Arrange: reproduce the corrupted rejoin, then apply the fix.
        text = "持续的寒冷气候是心脑血管疾病的杀手"
        encoded = self.tokenizer(text, add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        words = self._merge_tokens_to_words(tokens)
        keep_str = self.tokenizer.convert_tokens_to_string(words)

        # Act: the fix strips spaces inserted between CJK characters.
        fixed = remove_space_between_cjk(keep_str)

        # Assert: no CJK-space pairs remain and the text round-trips.
        cjk = "[\u4e00-\u9fff]"
        self.assertEqual(re.findall(cjk + " " + cjk, fixed), [])
        self.assertEqual(fixed, text)

    def test_fix_removes_spaces_around_cjk_punctuation(self):
        # Arrange: BERT splits fullwidth punctuation into its own word too, so
        # the corrupted rejoin also spaces out "字 ， 字" / "字 。 字".
        corrupted = "你 好 ， 世 界 。 测 试"
        # Act
        fixed = remove_space_between_cjk(corrupted)
        # Assert: spaces around CJK punctuation are removed as well.
        self.assertEqual(fixed, "你好，世界。测试")

    def test_fix_preserves_spaces_around_latin(self):
        # Arrange: mixed CJK/Latin text; spaces bordering Latin must survive.
        text = "使用 GPT 模型 with prompt compression 效果 good"
        # Act
        fixed = remove_space_between_cjk(text)
        # Assert: only CJK-CJK spaces are removed; Latin spacing is untouched.
        self.assertIn("GPT 模型", fixed)
        self.assertIn("prompt compression", fixed)


if __name__ == "__main__":
    unittest.main()
