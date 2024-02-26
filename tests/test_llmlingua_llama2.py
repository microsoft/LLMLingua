import unittest

from llmlingua import PromptCompressor


class LLMLinguaTester(unittest.TestCase):
    """
    End2end Test for LLMLingua
    """

    GSM8K_PROMPT = "Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\nAngelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n\nQuestion: You can buy 4 apples or 1 watermelon for the same price. You bought 36 fruits evenly split between oranges, apples and watermelons, and the price of 1 orange is $0.50. How much does 1 apple cost if your total bill was $66?\nLet's think step by step\nIf 36 fruits were evenly split between 3 types of fruits, then I bought 36/3 = 12 units of each fruit\nIf 1 orange costs $0.50 then 12 oranges will cost $0.50 * 12 = $6\nIf my total bill was $66 and I spent $6 on oranges then I spent $66 - $6 = $60 on the other 2 fruit types.\nAssuming the price of watermelon is W, and knowing that you can buy 4 apples for the same price and that the price of one apple is A, then 1W=4A\nIf we know we bought 12 watermelons and 12 apples for $60, then we know that $60 = 12W + 12A\nKnowing that 1W=4A, then we can convert the above to $60 = 12(4A) + 12A\n$60 = 48A + 12A\n$60 = 60A\nThen we know the price of one apple (A) is $60/60= $1\nThe answer is 1"
    GSM8K_150TOKENS_COMPRESSED_SINGLE_CONTEXT_PROMPT = "Question: Angelo and Melanie to plan how many hours the next they should study together their test They have 2ters of study and 4 to. They figure they should  to chapter1 hours. they study  hours they study total week they take a1 break hour, include 3ute breaks and 3\nLets think step\nanie should eachters,  hours2 hours total\nets they1 hours each hours hours\nanie to start with planning1 hours study, 2\n to include and. they to2 hours03ute,\nunch for minutes  60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4"
    GSM8K_150TOKENS_COMPRESSED_MULTIPLE_CONTEXT_PROMPT = "Question: You can buy 4les 1on for. You bought 36ruits evenly betweenangeslesons, and the price of1 is $.0. much does1 cost if your total bill $6\nLets think step\nIf ruits were between3, then I 632 each fruit\nIf0 then12anges $0\nIf6 Ianges I on other2 types\numing the of is W, and knowing that you can buy 4 apples for the same price and that the price of one apple is A, then 1W=4A\nIf we know we bought 12 watermelons and 12 apples for $60, then we know that $60 = 12W + 12A\nKnowing that 1W=4A, then we can convert the above to $60 = 12(4A) + 12A\n$60 = 48A + 12A\n$60 = 60A\nThen we know the price of one apple (A) is $60/60= $1\nThe answer is 1"
    JSON_PROMPT = """<llmlingua, compress=False>
    {
        "id": "</llmlingua><llmlingua, rate=0.8>987654</llmlingua><llmlingua, compress=False>",
        "name": "</llmlingua><llmlingua, rate=0.8>John Doe</llmlingua><llmlingua, compress=False>",
        "isActive": "</llmlingua><llmlingua, rate=0.8>true</llmlingua><llmlingua, compress=False>",
        "biography": "</llmlingua><llmlingua, rate=0.4>John Doe, born in New York in 1985, is a renowned software engineer with over 10 years of experience in the field. John graduated from MIT with a degree in Computer Science and has since worked with several Fortune 500 companies. He has a passion for developing innovative software solutions and has contributed to numerous open source projects. John is also an avid writer and speaker at tech conferences, sharing his insights on emerging technologies and their impact on the business world. In his free time, John enjoys hiking, reading science fiction novels, and playing the piano.</llmlingua><llmlingua, compress=False>",
        "employmentHistory": [
            {
            "company": "TechCorp",
            "role": "</llmlingua><llmlingua, rate=0.5>Senior Software Engineer</llmlingua><llmlingua, compress=False>",
            "description": "</llmlingua><llmlingua, rate=0.4>At TechCorp, John was responsible for leading a team of software engineers and overseeing the development of scalable web applications. He played a key role in driving the adoption of cloud technologies within the company, significantly enhancing the efficiency of their digital operations.</llmlingua><llmlingua, compress=False>"
            },
            {
            "company": "Innovatech",
            "role": "</llmlingua><llmlingua, rate=0.5>Lead Developer</llmlingua><llmlingua, compress=False>",
            "description": "</llmlingua><llmlingua, rate=0.4>In his role at Innovatech, John focused on developing cutting-edge AI algorithms and implementing machine learning solutions for various business applications. He was instrumental in developing a predictive analytics tool that transformed the company's approach to data-driven decision making.</llmlingua><llmlingua, compress=False>"
            }
        ],
        "skills": "</llmlingua><llmlingua, rate=0.4>Java, Python, Machine Learning, Cloud Computing, AI Development</llmlingua><llmlingua, compress=False>"
    }</llmlingua>"""
    JSON_COMPRESSED_PROMPT = """
    {
        "id": "987654",
        "name": "Johne",
        "isActive": "true",
        "biography": "e, born in York98 is aowned software engineer with experience in the.John fromIT a and has since worked with several companies.He has a for developingative and has to open isid and speaker atferences hisights on and the.ys, reading playing the piano.",
        "employmentHistory": [
            {
            "company": "TechCorp",
            "role": "Senior Software Engineer",
            "description": "At TechCorp, John was responsible for leading a team of software engineers and overseeing the development of scalable web applications.He played a key role in driving the adoption of cloud technologies within the company, significantly enhancing the efficiency of their digital operations."},
            {
            "company": "Innovatech",
            "role": "Lead Developer",
            "description": "In his role at Innovatech, John focused on developing cutting-edge AI algorithms and implementing machine learning solutions for various business applications.He was instrumental in developing a predictive analytics tool that transformed the company's approach to data-driven decision making."}
        ],
        "skills": "Java, Python, Machine Learning, Cloud Computing, AI Development"
    }"""
    MEETINGBANK_TRANSCRIPT_0_PROMPT = "<llmlingua, compress=False>Speaker 4:</llmlingua><llmlingua, rate=0.4> Thank you. And can we do the functions for content? Items I believe are 11, three, 14, 16 and 28, I believe.</llmlingua><llmlingua, compress=False>\nSpeaker 0:</llmlingua><llmlingua, rate=0.2> Item 11 is a communication from Council on Price recommendation to increase appropriation in the general fund group in the City Manager Department by $200 to provide a contribution to the Friends of the Long Beach Public Library. Item 12 is communication from Councilman Super Now. Recommendation to increase appropriation in the special advertising and promotion fund group and the city manager's department by $10,000 to provide support for the end of summer celebration. Item 13 is a communication from Councilman Austin. Recommendation to increase appropriation in the general fund group in the city manager department by $500 to provide a donation to the Jazz Angels . Item 14 is a communication from Councilman Austin. Recommendation to increase appropriation in the general fund group in the City Manager department by $300 to provide a donation to the Little Lion Foundation. Item 16 is a communication from Councilman Allen recommendation to increase appropriation in the general fund group in the city manager department by $1,020 to provide contribution to Casa Korero, Sew Feria Business Association, Friends of Long Beach Public Library and Dave Van Patten. Item 28 is a communication. Communication from Vice Mayor Richardson and Council Member Muranga. Recommendation to increase appropriation in the general fund group in the City Manager Department by $1,000 to provide a donation to Ron Palmer Summit. Basketball and Academic Camp.</llmlingua><llmlingua, compress=False>\nSpeaker 4:</llmlingua><llmlingua, rate=0.6> We have a promotion and a second time as councilman served Councilman Ringa and customers and they have any comments.</llmlingua><llmlingua, compress=False>\nSpeaker 2:</llmlingua><llmlingua, rate=0.6> Now. I had queued up to motion, but.</llmlingua><llmlingua, compress=False>\nSpeaker 4:</llmlingua><llmlingua, rate=0.6> Great that we have any public comment on this.</llmlingua><llmlingua, compress=False>\nSpeaker 5:</llmlingua><llmlingua, rate=0.6> If there are any members of the public that would like to speak on items 11, 12, 13, 14, 16 and 28 in person, please sign up at the podium in Zoom. Please use the raise hand feature or dial star nine now. Seen on the concludes public comment.</llmlingua><llmlingua, compress=False>\nSpeaker 4:</llmlingua><llmlingua, rate=0.3> Thank you. Please to a roll call vote, please.</llmlingua><llmlingua, compress=False>\nSpeaker 0:</llmlingua><llmlingua, rate=0.3> Councilwoman Sanchez.</llmlingua><llmlingua, compress=False>\nSpeaker 2:</llmlingua><llmlingua, rate=0.3> I am.</llmlingua><llmlingua, compress=False>\nSpeaker 0:</llmlingua><llmlingua, rate=0.3> Councilwoman Allen. I. Councilwoman Price.</llmlingua><llmlingua, compress=False>\nSpeaker 2:</llmlingua><llmlingua, rate=0.3> I.</llmlingua><llmlingua, compress=False>\nSpeaker 0:</llmlingua><llmlingua, rate=0.3> Councilman Spooner, i. Councilwoman Mongo i. Councilwoman Sarah I. Councilmember Waronker I. Councilman Alston.</llmlingua><llmlingua, compress=False>\nSpeaker 1:</llmlingua><llmlingua, rate=0.3> I.</llmlingua><llmlingua, compress=False>\nSpeaker 0:</llmlingua><llmlingua, rate=0.3> Vice Mayor Richardson.</llmlingua><llmlingua, compress=False>\nSpeaker 3:</llmlingua><llmlingua, rate=0.3> I.</llmlingua><llmlingua, compress=False>\nSpeaker 0:</llmlingua><llmlingua, rate=0.3> The motion is carried nine zero.</llmlingua><llmlingua, compress=False>\nSpeaker 4:</llmlingua><llmlingua, rate=0.05> Thank you. That concludes the consent. Just a couple announcements for the regular agenda. So we do have a very long and full agenda today. We have the budget hearing, which will happen first and then right after the budget hearing. We have a variety of other hearings as it relate to the our local control program and sales tax agreement. And then we have we're going to go right into some issues around and bonds around the aquarium and also the second reading of the health care worker ordinance, which we're going to try to do all of that towards the beginning of the agenda. And then we have a long agenda for the rest of of the council. So I just want to warn folks that we do have a we do have a long meeting. We're going to go right into the budget hearings. That's the first thing on the agenda. And they're going to try to move through that, through the council as expeditiously as possible. And so with that, let's continue the budget hearing, which we are doing for fire, police and parks. We're going to hear all of the presentations at once. And then after we go through all the presentations, we'll do all the all of the questions at once and then any any public comment, and we'll go from there.</llmlingua>"
    COMPRESSED_MULTIPLE_STRUCTURED_CONTEXT_PROMPT = '\n    {\n        "id": "987654",\n        "name": "Johne",\n        "isActive": "true",\n        "biography": "e, in York98 aowned engineer with. John from worked with several companies He has for and has toid and speaker atferences on and the.ys, reading playing the piano.",\n        "employmentHistory": [\n            {\n            "company": "TechCorp",\n            "role": "ior",\n            "description": " forers and ofable He in the of cloudologies company, significantly of their."\n            },\n            {\n            "company": "Innovatech",\n            "role": "ead",\n            "description": " John developingedgeI and implementing learning various business in aics thats."\n            }\n        ],\n        "skills": ",,,I"\n    }\n\nSpeaker 4: you. And we do the functions for content? Items I believe are11,11628,\nSpeaker 0: a communication from on Price to increase the fund group the0 a the the Councilman Super. the special group theman the Jazzelsman. increase the a the Littleman Allen0 provide to Casa, Sewia, of and Dave Van communication. from Member Mur to Ronmer. and\nSpeaker 4: We have a promotion and a second time as councilman servedman Ringa and customers and they have any comments.\nSpeaker 2: Now. I had que up to motion, but.\nSpeaker 4: Great that we have any public comment on this\nSpeaker 5: If there are any public that to speak on items116 8 in, please up theium in. Please hand feature orial star nine now. Seen on the concludes\nSpeaker 4:. Please to a\nSpeaker 0: Counciloman\nSpeaker 2:\nSpeaker 0:..oman\nSpeaker 2:\nSpeaker 0:man,oman Mongooman Sarah Councilmember Waronkerman Al\nSpeaker 1:\nSpeaker 0: Mayor\nSpeaker 3:\nSpeaker 0: The carried\nSpeaker 4:. So I just want to warn folks that we do have a we do have a long meeting. We\'re going to go right into the budget hearings. That\'s the first thing on the agenda. And they\'re going to try to move through that, through the council as expeditiously as possible. And so with that, let\'s continue the budget hearing, which we are doing for fire, police and parks. We\'re going to hear all of the presentations at once. And then after we go through all the presentations, we\'ll do all the all of the questions at once and then any any public comment, and we\'ll go from there.'

    def __init__(self, *args, **kwargs):
        super(LLMLinguaTester, self).__init__(*args, **kwargs)
        try:
            import nltk
            nltk.download('punkt')
        except:
            print('nltk_data exits.')
        self.llmlingua = PromptCompressor("NousResearch/Llama-2-7b-hf", device_map="cpu")

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
        self.assertEqual(compressed_prompt["compressed_tokens"], 201)
        self.assertEqual(compressed_prompt["ratio"], "2.1x")
        self.assertEqual(compressed_prompt["rate"], "47.6%")

        # Multiple Context
        compressed_prompt = self.llmlingua.compress_prompt(
            self.GSM8K_PROMPT.split("\n\n"), target_token=150
        )
        self.assertEqual(
            compressed_prompt["compressed_prompt"],
            self.GSM8K_150TOKENS_COMPRESSED_MULTIPLE_CONTEXT_PROMPT,
        )
        self.assertEqual(compressed_prompt["origin_tokens"], 727)
        self.assertEqual(compressed_prompt["compressed_tokens"], 223)
        self.assertEqual(compressed_prompt["ratio"], "3.3x")
        self.assertEqual(compressed_prompt["rate"], "30.7%")

    def test_general_structured_compress_prompt(self):
        # Single Stuctured Context
        import json

        context, _, _, _ = self.llmlingua.segment_structured_context(
            [self.JSON_PROMPT], 0.5
        )
        _ = json.loads(context[0])
        compressed_prompt = self.llmlingua.structured_compress_prompt(
            [self.JSON_PROMPT],
            rate=0.5,
            use_sentence_level_filter=True,
            use_token_level_filter=True,
        )
        _ = json.loads(compressed_prompt["compressed_prompt"])
        self.assertEqual(
            compressed_prompt["compressed_prompt"],
            self.JSON_COMPRESSED_PROMPT,
        )
        self.assertEqual(compressed_prompt["origin_tokens"], 318)
        self.assertEqual(compressed_prompt["compressed_tokens"], 260)
        self.assertEqual(compressed_prompt["ratio"], "1.2x")
        self.assertEqual(compressed_prompt["rate"], "81.8%")

        # Multiple Stuctured Context
        compressed_prompt = self.llmlingua.structured_compress_prompt(
            [self.JSON_PROMPT, self.MEETINGBANK_TRANSCRIPT_0_PROMPT],
            rate=0.5,
            use_sentence_level_filter=False,
            use_token_level_filter=True,
        )
        self.assertEqual(
            compressed_prompt["compressed_prompt"],
            self.COMPRESSED_MULTIPLE_STRUCTURED_CONTEXT_PROMPT,
        )
        self.assertEqual(compressed_prompt["origin_tokens"], 1130)
        self.assertEqual(compressed_prompt["compressed_tokens"], 555)
        self.assertEqual(compressed_prompt["ratio"], "2.0x")
        self.assertEqual(compressed_prompt["rate"], "49.1%")
