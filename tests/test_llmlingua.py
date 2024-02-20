import unittest

from llmlingua import PromptCompressor


class LLMLinguaTester(unittest.TestCase):
    """
    End2end Test for LLMLingua
    """

    GSM8K_PROMPT = "Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\nAngelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n\nQuestion: You can buy 4 apples or 1 watermelon for the same price. You bought 36 fruits evenly split between oranges, apples and watermelons, and the price of 1 orange is $0.50. How much does 1 apple cost if your total bill was $66?\nLet's think step by step\nIf 36 fruits were evenly split between 3 types of fruits, then I bought 36/3 = 12 units of each fruit\nIf 1 orange costs $0.50 then 12 oranges will cost $0.50 * 12 = $6\nIf my total bill was $66 and I spent $6 on oranges then I spent $66 - $6 = $60 on the other 2 fruit types.\nAssuming the price of watermelon is W, and knowing that you can buy 4 apples for the same price and that the price of one apple is A, then 1W=4A\nIf we know we bought 12 watermelons and 12 apples for $60, then we know that $60 = 12W + 12A\nKnowing that 1W=4A, then we can convert the above to $60 = 12(4A) + 12A\n$60 = 48A + 12A\n$60 = 60A\nThen we know the price of one apple (A) is $60/60= $1\nThe answer is 1"
    GSM8K_150TOKENS_COMPRESSED_SINGLE_CONTEXT_PROMPT = "Question: Angelo and Melanie to plan how many hours they should together their test have 2 their textbook and 4 to They out should and 1 hours. they study, many they study total week they a break every hour, include 3minute and lunch day\n's think step\n Melanie should the chapters hours 2 = hours\n the to dedicate x\n Melanie to with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4"
    GSM8K_150TOKENS_COMPRESSED_MULTIPLE_CONTEXT_PROMPT = "Question: You can buy 4 apples or 1 for. You bought 36 fruits evenly split between, waterons and of 1 orange $.. much does cost if your total bill $\n's think step\nIf were between 3 of, then I 36/3 = 12 of fruitIf 1 orange50 then oranges50 * $If66 I $ oranges I $66 $60 on the other 2 fruit\nAssuming the of is W, and that you price and of is then 1W=4AIf we know we bought 12 and, then we know that $60 = 12W + 12A\nKnowing that 1W=4A, then we can convert the above to $60 = 12(4A) + 12A\n$60 = 48A + 12A\n$60 = 60A\nThen we know the price of one apple (A) is $60/60= $1\nThe answer is 1"

    def __init__(self, *args, **kwargs):
        super(LLMLinguaTester, self).__init__(*args, **kwargs)
        self.llmlingua = PromptCompressor("lgaalves/gpt2-dolly", device_map="cpu")

    def test_general_compress_prompt(self):
        # Single Context
        compressed_prompt = self.llmlingua.compress_prompt(
            self.GSM8K_PROMPT.split("\n\n")[0], target_token=150
        )
        self.assertEqual(
            compressed_prompt["compressed_prompt"],
            self.GSM8K_150TOKENS_COMPRESSED_SINGLE_CONTEXT_PROMPT,
        )
        self.assertEqual(compressed_prompt["origin_tokens"], 422)
        self.assertEqual(compressed_prompt["compressed_tokens"], 293)
        self.assertEqual(compressed_prompt["ratio"], "1.4x")
        self.assertEqual(compressed_prompt["rate"], "69.4%")

        # Multiple Context
        compressed_prompt = self.llmlingua.compress_prompt(
            self.GSM8K_PROMPT.split("\n\n"), target_token=150
        )
        self.assertEqual(
            compressed_prompt["compressed_prompt"],
            self.GSM8K_150TOKENS_COMPRESSED_MULTIPLE_CONTEXT_PROMPT,
        )
        self.assertEqual(compressed_prompt["origin_tokens"], 727)
        self.assertEqual(compressed_prompt["compressed_tokens"], 206)
        self.assertEqual(compressed_prompt["ratio"], "3.5x")
        self.assertEqual(compressed_prompt["rate"], "28.3%")
