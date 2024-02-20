import unittest
import unittest.mock as mock

from llmlingua import PromptCompressor


class LongLLMLinguaTester(unittest.TestCase):
    """
    End2end Test for LongLLMLingua
    """

    GSM8K_PROMPT = "Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\nAngelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n\nQuestion: You can buy 4 apples or 1 watermelon for the same price. You bought 36 fruits evenly split between oranges, apples and watermelons, and the price of 1 orange is $0.50. How much does 1 apple cost if your total bill was $66?\nLet's think step by step\nIf 36 fruits were evenly split between 3 types of fruits, then I bought 36/3 = 12 units of each fruit\nIf 1 orange costs $0.50 then 12 oranges will cost $0.50 * 12 = $6\nIf my total bill was $66 and I spent $6 on oranges then I spent $66 - $6 = $60 on the other 2 fruit types.\nAssuming the price of watermelon is W, and knowing that you can buy 4 apples for the same price and that the price of one apple is A, then 1W=4A\nIf we know we bought 12 watermelons and 12 apples for $60, then we know that $60 = 12W + 12A\nKnowing that 1W=4A, then we can convert the above to $60 = 12(4A) + 12A\n$60 = 48A + 12A\n$60 = 60A\nThen we know the price of one apple (A) is $60/60= $1\nThe answer is 1\n\nQuestion: Susy goes to a large school with 800 students, while Sarah goes to a smaller school with only 300 students.  At the start of the school year, Susy had 100 social media followers.  She gained 40 new followers in the first week of the school year, half that in the second week, and half of that in the third week.  Sarah only had 50 social media followers at the start of the year, but she gained 90 new followers the first week, a third of that in the second week, and a third of that in the third week.  After three weeks, how many social media followers did the girl with the most total followers have?\nLet's think step by step\nAfter one week, Susy has 100+40 = 140 followers.\nIn the second week, Susy gains 40/2 = 20 new followers.\nIn the third week, Susy gains 20/2 = 10 new followers.\nIn total, Susy finishes the three weeks with 140+20+10 = 170 total followers.\nAfter one week, Sarah has 50+90 = 140 followers.\nAfter the second week, Sarah gains 90/3 = 30 followers.\nAfter the third week, Sarah gains 30/3 = 10 followers.\nSo, Sarah finishes the three weeks with 140+30+10 = 180 total followers.\nThus, Sarah is the girl with the most total followers with a total of 180.\nThe answer is 180"
    GSM8K_250TOKENS_COMPRESSED_SINGLE_CONTEXT_PROMPT = "Question: Angelo Melanie want plan many over the together for their next week chapters of to study worksheets memorize They out they should hours each chapter 1. hours each worksheet plan study no each day, many plan to total take a 10minute break every hour 10- snack breaks each day, 30 minutes each day?Let think step by step\nAngelo and Melanie think they should dedicate 3 hours to each of 2 chapters, x chapters hours total.For worksheets plan to 1.5 for works,.5 hours 4 worksheets hours total and Melanie need with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n\nQuestion: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?"
    GSM8K_250TOKENS_COMPRESSED_MULTIPLE_CONTEXT_PROMPT = "Question: You or 1 watermelon the price You bought 36 evenly split between oranges, watermelons, and price 1 orange is $0.50 How apple if total bill was $?Let's think step by stepIf were evenly split 3 types of fruits, 36/ = 12 units of fruit\nIf 1 orange $0. then will cost $0.50 * 126\nIf my total bill was $ and I spent $6 on oranges then I spent $66 - $6 = $ the other 2 fruit types.\nAssuming price is W knowing you 4 apples the same price that price one is A, 1=4A\n we we bought watermelons and apples for $60 we know that $ 12W + 12A\nKnowing that 1W=4A, we can convert the above to $ 12(4A) + 12A$60 = 48A + 12A\n$60 = 60\nThen know price of one apple (A) is $60/60= $\nThe answer is 1\n\nQuestiony while.  After three weeks, how many social media followers did the girl with the most total followers have?\nLet's think step by step\nAfter one week, Susy has 100+40 = 140 followers.\nIn the second week, Susy gains 40/2 = 20 new followers.\nIn the third week, Susy gains 20/2 = 10 new followers.\nIn total, Susy finishes the three weeks with 140+20+10 = 170 total followers.\nAfter one week, Sarah has 50+90 = 140 followers.\nAfter the second week, Sarah gains 90/3 = 30 followers.\nAfter the third week, Sarah gains 30/3 = 10 followers.\nSo, Sarah finishes the three weeks with 140+30+10 = 180 total followers.\nThus, Sarah is the girl with the most total followers with a total of 180.\nThe answer is 180\n\nQuestion: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?"
    QUESTION = "Question: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?"

    def __init__(self, *args, **kwargs):
        super(LongLLMLinguaTester, self).__init__(*args, **kwargs)
        self.llmlingua = PromptCompressor("lgaalves/gpt2-dolly", device_map="cpu")

    def test_general_compress_prompt(self):
        # Single Context
        compressed_prompt = self.llmlingua.compress_prompt(
            self.GSM8K_PROMPT.split("\n\n")[0],
            question=self.QUESTION,
            target_token=250,
            condition_in_question="after_condition",
            reorder_context="sort",
            dynamic_context_compression_ratio=0.4,
            condition_compare=True,
            context_budget="+100",
            rank_method="longllmlingua",
        )
        self.assertEqual(
            compressed_prompt["compressed_prompt"],
            self.GSM8K_250TOKENS_COMPRESSED_SINGLE_CONTEXT_PROMPT,
        )
        self.assertEqual(compressed_prompt["origin_tokens"], 474)
        self.assertEqual(compressed_prompt["compressed_tokens"], 385)
        self.assertEqual(compressed_prompt["ratio"], "1.2x")
        self.assertEqual(compressed_prompt["rate"], "81.2%")

        # Multiple Context
        compressed_prompt = self.llmlingua.compress_prompt(
            self.GSM8K_PROMPT.split("\n\n"),
            question=self.QUESTION,
            target_token=250,
            condition_in_question="after_condition",
            reorder_context="sort",
            dynamic_context_compression_ratio=0.4,
            condition_compare=True,
            context_budget="+100",
            rank_method="longllmlingua",
        )
        self.assertEqual(
            compressed_prompt["compressed_prompt"],
            self.GSM8K_250TOKENS_COMPRESSED_MULTIPLE_CONTEXT_PROMPT,
        )
        self.assertEqual(compressed_prompt["origin_tokens"], 1094)
        self.assertEqual(compressed_prompt["compressed_tokens"], 474)
        self.assertEqual(compressed_prompt["ratio"], "2.3x")
        self.assertEqual(compressed_prompt["rate"], "43.3%")
