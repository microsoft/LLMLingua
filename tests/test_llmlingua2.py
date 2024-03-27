# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import unittest

from llmlingua import PromptCompressor


class LLMLingua2Tester(unittest.TestCase):
    """
    End2end Test for LLMLingua-2
    """

    PROMPT = "John: So, um, I've been thinking about the project, you know, and I believe we need to, uh, make some changes. I mean, we want the project to succeed, right? So, like, I think we should consider maybe revising the timeline.\n\nSarah: I totally agree, John. I mean, we have to be realistic, you know. The timeline is, like, too tight. You know what I mean? We should definitely extend it."
    COMPRESSED_SINGLE_CONTEXT_PROMPT = "John: thinking project believe need make changes. want project succeed? consider revising timeline.\n\n Sarah agree. be realistic. timeline too tight.? extend."
    COMPRESSED_MULTIPLE_CONTEXT_PROMPT = "John: So, I've been thinking about project believe we need to make changes. we want project to succeed, right? think we should consider maybe revising timeline."

    GSM8K_PROMPT = "Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\nAngelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n\nQuestion: You can buy 4 apples or 1 watermelon for the same price. You bought 36 fruits evenly split between oranges, apples and watermelons, and the price of 1 orange is $0.50. How much does 1 apple cost if your total bill was $66?\nLet's think step by step\nIf 36 fruits were evenly split between 3 types of fruits, then I bought 36/3 = 12 units of each fruit\nIf 1 orange costs $0.50 then 12 oranges will cost $0.50 * 12 = $6\nIf my total bill was $66 and I spent $6 on oranges then I spent $66 - $6 = $60 on the other 2 fruit types.\nAssuming the price of watermelon is W, and knowing that you can buy 4 apples for the same price and that the price of one apple is A, then 1W=4A\nIf we know we bought 12 watermelons and 12 apples for $60, then we know that $60 = 12W + 12A\nKnowing that 1W=4A, then we can convert the above to $60 = 12(4A) + 12A\n$60 = 48A + 12A\n$60 = 60A\nThen we know the price of one apple (A) is $60/60= $1\nThe answer is 1"
    GSM8K_150TOKENS_COMPRESSED_SINGLE_CONTEXT_PROMPT = "Question: Angelo Melanie plan test 2 chapters 4 worksheets 3 hours each chapter 1.5 hours each worksheet study 4 hours day how days 10-minute break every 3 10-minute snack breaks 30 minutes lunch\n\n dedicate 3 hours 2 chapters 3 2 = 6 hours total\n worksheets 1.5 hours each worksheet 1.5 4 = 6 hours total\n 12 hours study 4 hours a day 12 / 4 = 3 days\n breaks lunch 10-minute break 12 hours 10 = 120 minutes\n 3 10-minute snack breaks 3 10 = 30 minutes\n 30 minutes lunch 120 + 30 + 30 = 180 minutes 180 / 60 = 3 extra hours\n 12 hours study + 3 hours breaks = 15 hours total\n 4 hours each day 15 / 4 = 3.75\n 4 days\nThe answer is 4"
    GSM8K_150TOKENS_COMPRESSED_MULTIPLE_CONTEXT_PROMPT = "4 apples 1 watermelon 36 fruits oranges watermelons 1 orange $0.50 1 apple bill $66\n\n 36 fruits 3 36/3 = 12 units\n 1 orange $0.50 12 oranges $0.50 * 12 = $6\n total bill $66 spent $6 oranges $66 - $6 = $60 other 2\n watermelon W 4 apples one apple A 1W=4A\n 12 watermelons 12 apples $60 $60 = 12W + 12A\n $60 = 12(4A + 12A\n = 48A + 12A\n = 60A\n one apple $60/60= $1\nThe answer is 1"

    MEETINGBANK_PROMPT = "Item 28 Report from Development. Services Recommendation to declare ordinance amending the Land Use District Map from institutional to IRP 13 read and adopted as read District eight. Councilman Austin. So moved. Wonderful. And I want to ask Councilman Andrews so any member of the public that wishes to address item 28 saying none, members, cast your vote. Oh, I'm sorry, sir. I did not see you. Can we? I know this sounds picky and stupid. But this is an illogical motion because you haven't yet created ARP 13. By the way, unlike some other speakers, I will furnish you my name. I'm Joe Weinstein. I did speak last week. I do not like to come down here again to talk on the same subjects. But. There is a minor little matter. As to whether a. The proposed zoning is a good idea. And B, whether. The project, which it is intended. To permit. In fact. Meets the specifications of the zoning. I have not check that out, but someone else did raise that question and there may be some question as to whether all of the conditions of that zoning have, in fact, been met by the details of this project. This particular zoning, perhaps in the abstract, need not be a bad idea, but the way you see it realized in the project. Is not a very good idea. You could have the same density and more without destroying the usability, the usable green space that this design does. Because really, although it looks impressive from a top down view, it looks like you see plenty of green space between the buildings, that that space is pretty well wasted and useless because the buildings are high enough to pretty well shade and dominate the green space that's in that project. So I'm not saying that the density that you're going for is a bad thing. But doing it in this way doesn't work, and any zoning that just permits this without further control is not a good idea. Thank you. Okay. Thank you, sir. Members, please cast your vote. Councilman Andrew's motion carries. Next time, please. Report from Development Services recommendation to declare ordinance amending the Land Use District Map from institutional to park red and adopted as Red District eight."
    MEETINGBANK_150TOKENS_COMPRESSED_SINGLE_CONTEXT_PROMPT = "Item 28 Report Development. Services Recommendation declare ordinance amending Land Use District Map institutional IRP 13 adopted District eight. Councilman Austin. ask Councilman Andrews public address item 28 cast vote. see?. illogical motion created ARP 13. Joe Weinstein. last week. same subjects. minor matter. proposed zoning good idea. project intended. permit Meets specifications zoning. question conditions zoning met details project. zoning not bad project. not good. same density more without destroying usability green space. green space between buildings wasted useless buildings high shade dominate green space. not density bad. doesn't work zoning permits without control not good idea. Thank you. cast vote. Councilman Andrew's motion carries. Next time.Development Services ordinance Land District Map park District."

    LONGBENCH_PROMPT_LIST = [
        "新闻内容：\n（服务·健康）专家提醒：寒冷气候易诱发心脑血管疾病\n新华社海口２月９日专电（张苏民、李建国）海口市疾病预防控制中心专家介绍，持续的寒冷气候是心脑血管疾病的杀手，尤其患有高血压或高血脂疾病的老人更应做好防范，防止脑中风发生。\n　　在寒冷的气候环境当中要注意保暖，增添衣服，饮食以清淡为主，多食用蔬菜，忌暴食荤类。尤其过年时，切忌熬夜，平时要加强身体锻炼，劳逸结合。除此之外，冬季还是呼吸道传染病暴发和流行的季节，应该注意预防流感、麻疹、流脑、水痘等呼吸道传染病的发生。\n　　专家还指出，由于寒冷气候影响，人们习惯门窗紧闭，空气不对流，一旦有传染源传入，极容易造成疾病的暴发。春节期间，一些商场或公共娱乐场所人群密集，有关单位应加强通风。（完）\n类别：医药、卫生",
        "\n\n新闻内容：\n李明波在恩施调研时强调 大力推进基层党内民主建设\n本报讯　（记者吴畏、通讯员曾言、周恩祖）11日至13日，省委常委、秘书长李明波到恩施州调研基层党建工作时强调，要以增强党的创新活力、巩固党的团结统一为目标，以改革创新精神大力抓好基层党内民主建设。\n　　李明波视察了非公有制企业党建、党代表常任制、基层党务公开、以党内和谐推进社区和谐等党建工作现场，与基层党务工作者座谈。李明波强调，在新形势下，要把握好民主进程与经济社会发展、尊重党员主体地位与提高党员民主素质、履行党员民主权利与保证党的统一意志、发挥党员民主监督作用与加强党纪教育管理等的关系，进一步深入探索，在丰富形式、拓宽渠道、完善机制等方面取得更大成绩。\n类别：政治",
        "\n\n新闻内容：\n第38届世界贸易中心年会及经贸洽谈会\n第38届世界贸易中心年会将于2007年10月21至24日在美国路易斯\n安那州首府新奥尔良召开。该会由美国纽约世界贸易中心总部和美国贸\n易服务管理总局、新奥尔良世贸中心共同举办，届时将有来自60多个国\n家和地区的经贸代表团约600余人与会。天津贸促会与天津世贸中心协\n会将共同组织天津经贸代表团赴美国参加“世贸中心2007年年会及经贸\n洽谈会”。\n　　联系人：王岭　刘鹏\n　　电话：022－2520231725202123\n　　传真：022－25201975\n　　地址：天津经济技术开发区宏达街19号A区2楼\n类别：商业、外贸、海关",
        "\n\n新闻内容：\n（全运会）第十一届全运会开闭幕时间确定\n新华社济南６月５日体育专电（记者赵仁伟）第十一届全国运动会组委会５日在济南宣布，十一运会将于今年１０月１６日在济南奥体中心开幕，闭幕时间为１０月２８日。\n　　十一运会组委会常务副秘书长、山东省体育局局长张洪涛介绍，十一运会的比赛项目共设３３个大项、４３个分项、３６２个小项，其中包括２８个夏季奥运会项目、４个冬季项目以及武术项目。与２００５年十运会相比，大项增加了１个，即自由式滑雪；小项增加了５个，分别是自由式滑雪男子个人、女子个人，女子水球项目，足球男子１６岁以下组和女子１８岁以下组。\n　　在十一运会全部３６２个小项中，马拉松男、女２个小项的比赛在北京举办，速度滑冰４个小项、自由式滑雪２个小项的比赛分别在沈阳和长春举办，其余３５４个小项的比赛在山东省１７个赛区举行。其中，济南赛区共举办小项２１２个，青岛４８个，日照４０个，滨州２８个，枣庄８个，菏泽７个，威海５个，烟台、德州各３个；淄博、东营、潍坊、济宁、泰安、莱芜、临沂、聊城８个赛区只举办小组赛和第四名以后的比赛，不产生金牌。\n　　张洪涛介绍，十一运会冰雪项目已于１月至４月举行，占全部小项的４.４％。因部分夏季项目的世界锦标赛或国际重要赛事的时间与十一运会比赛时间冲突或相距较近，国家体育总局确定把这些项目的比赛安排在开幕式前举行，共有１５个项目、８０个小项，占全部小项的２２.１％。（完）\n类别：体育",
        "\n\n新闻内容：\n（教育）河北整顿公办初中、小学招收择校生\n（教育）河北整顿公办初中、小学招收择校生\n　　新华社石家庄３月１２日电（冯月静）记者从河北省教育纪检监察审计工作会议上了解到，从今年起，河北省不再审批新的改制学校。对已审批的改制学校进行一次全面整顿和规范，重点解决公办初中、小学以改制为名或以民办为名举办“校中校”“校中班”高收费问题。\n　　据了解，河北省规定达不到要求的，要限期整改；年内仍达不到标准要求的，一律停止招生。公办学校一律不准搞“一校两制”，更不准以改制为名高收费。\n　　同时，今年秋季新学年开始，设区市市区的公办省级示范性普通高中（含在县镇办学的市直属省级示范性高中）择校生比例最高限额由原定的４０％一律下调为３０％。严禁学校擅自扩大择校生招生比例、降低录取分数线、提高收费标准或在限定金额外加收任何其他费用。（完）\n类别：教育",
        "\n\n新闻内容：\n（服务·关注“过劳死”） “过劳死”青睐什么人？\n人？\n    新华社郑州３月１６日专电(记者李丽静)  有关专家\n研究表明：受教育程度高、中青年、女性是“过劳死”这\n一疾病的危险人群。这是因为这些人事业上强力拼搏，生\n活负荷过重，自身经常处于紧张状态之中，过度疲劳难以\n避免。\n    随着社会竞争的日趋激烈，该病也越来越多地困扰着\n我国的都市人。据一项在上海、无锡、深圳等地对\n１１９７位中年人健康状况调查显示：其中６６％的人有\n失眠、多梦、不易入睡等现象；６２％的人经常腰酸背痛；\n５８％的人一干活就累；５７％的人爬楼时感到吃力或记\n忆力明显减退；４８％的人皮肤干燥、瘙痒、面色晦暗、\n脾气暴躁、焦虑。据国家有关部门的一项调查结果表明，\n慢性疲劳综合征在城市新兴行业人群中的发病率为１０％\n至２０％，在科技、新闻、广告、公务人员、演艺、出租\n车司机等行业中发病率则更高。\n    有关专家通过统计认为，“过劳死”特别“青睐”三\n种人：\n    第一种是有钱但不知保养的人。这部分人“富裕”的\n背后，往往有一条铺满辛酸的路。由于对贫穷的恐惧，使\n他们对财富永远不满足。为了追逐更多的财富，即使赴汤\n蹈火也在所不辞，而对他们最初惟一的资本———身体，\n则很不在乎。 \n    第二种是有事业心，特别是称得上“工作狂”的人。\n主要以从事科研、教学、新型高科技，如网络等职业者居\n多。\n    第三种是有家族遗传背景者。如果父母亲、爷爷奶奶\n等直系亲属中有心绞痛、心肌梗死、脑中风的患者，就要\n特别小心了，千万别让自己累着，否则很有可能在年轻时\n就诱发疾病。\n    而在对“过劳死”人群深入研究中发现，猝死直接死\n因的前5位是冠状动脉疾病、主动脉瘤、心瓣膜病、心肌\n病和脑出血。一些无症状冠心病，特别是无症状心肌梗死\n是首要的危险因素，一般的体检和心电图不易发现隐性冠\n心病。一旦发作，措手不及。此外，高血压也是一个潜在\n的危险因素。在遇到某些诱因时，便会引发高血压、脑中\n风等。（完）\n类别：医药、卫生",
        "\n\n新闻内容：\n五项措施应对技术性贸易壁垒\n调查结果显示,2006年我国有31\n    .4%的出口企业受到国外技术性贸易措施不同程度的影响,比2005年增长6.3个百分点;全年出口贸易直接损失359.20亿美元,占同期出口额的3.71%,企业新增成本191.55亿美元。\n    会议通报的情况显示,对中国企业出口影响较大的技术性贸易措施类型集中在认证要求、技术标准要求、有毒有害物质限量要求、包装及材料的要求和环保要求(包括节能及产品回收),食品中农兽药残留要求、重金属等有害物质限量要求、细菌等卫生指标要求、食品标签要求和食品接触材料的要求等方面。受国外技术性贸易措施影响较大的行业排在前五位的是机电、农食产品、化矿、塑料皮革和纺织鞋帽。\n    会议提出了加强应对的5点意见。一是要强化进出口质量监管措施,在“严”字上下功夫,重点从源头上抓好农兽药残留、有毒化学物质残留、微生物等问题,同时要完善监管机制,提高检测能力,要检得出,检得快,检得准。二是要加快实施技术标准战略,在“高”字上下功夫,不断提高采标率,加快标准的制修订步伐。三是要加大信息共享力度,在“准”字上下功夫,各部门要密切配合,建立沟通机制,做到信息资源的充分利用。四是要果断迅速应对突发事件,在“快”字上下功夫。五是要加强技术性贸易措施的积极应对,在“实”字上下功夫,协调配合、相互支持。\n类别：商业、外贸、海关",
        "\n\n新闻内容：\n（新华时评·奥运会倒计时一百天）让我们共同守护奥林匹克精神\n新华社北京４月３０日电　题：让我们共同守护奥林匹克精神\n    新华社记者张旭\n    在北京奥运会倒计时一百天之际，奥运圣火结束在其他国家的传递进入中国香港。在这两个重要时间节点重合之时，让我们以奥林匹克精神为依归，回味今年以来围绕北京奥运的风风雨雨，并以百倍的努力在接下来的日子里守护这一美好理想。\n    奥林匹克运动会是古希腊人的体育盛会，许多比赛项目源于古希腊文化。顾拜旦说：“古希腊人之所以组织竞赛活动，不仅仅只是为了锻炼体格和显示一种廉价的壮观场面，更是为了教育人”。更高更快更强并不是现代奥林匹克运动的全部价值诉求。现代奥林匹克运动经过了一百年的历史变迁，向世界传达的精神与主题始终如一，那就是在共同创造、共同分享、平等友爱的旗帜下，展现人类最美好的情感。奥林匹克是迄今为止人类社会不同种族、地域乃至不同意识形态间最大的交集。\n　　２００１年７月１３日，时任国际奥委会主席的萨马兰奇宣布北京取得２００８年奥运会主办权，现代奥林匹克运动从奥林匹亚来到万里长城。７年后的春天，当奥运圣火开始在中国境外传递时，妖魔化中国的舆论攻势和扰乱奥运火炬传递的暴力举动让海内外目光聚焦中国。我们可以肯定地说，这些人在为一己之私对奥林匹克精神进行亵渎。\n   北京奥运圣火一路走来，虽然遇到了噪音和干扰，但更多面对的还是像火一样热情的世界人民和对奥林匹克精神充分尊重的各国人士。他们因为懂得尊重奥林匹克精神，因此也能够享受奥林匹克带来的快乐。\n    ２００８年４月３０日，“北京欢迎你”的歌声回荡在有着近６００年历史的紫禁城太庙上空。８月８日，中国人民将第一次以东道主的身份在北京承办举世瞩目的奥林匹克运动会。北京奥运会对中国来说不仅仅是一次体育盛会，更是一次与世界各国开展文化交流的机会。如同当年奥林匹亚为神圣的无战争区域一样，体育竞技的目标是为了全世界的和平与发展。北京奥运会也完全可以成为世界各种文明一个共同的精神家园，通过沟通交流，达到良性互动。\n   奥运会的脚步声离我们越来越近的时候，奥林匹克运动正在为１３亿中国人民所熟悉，奥林匹克精神也继续在世界范围内承载起人类追求幸福生活的梦想。中国人民真诚地邀请各国运动员、教练员和朋友们参与２００８年北京奥运会。中国人民同时真诚地邀请全世界热爱奥林匹克精神和奥林匹克运动的人们一起，共同守护这一人类美好理想，让它在北京奥运会上开放出更加美丽的花朵。（完）\n类别：体育",
        "\n\n新闻内容：\n海口“接管”省 特殊教育 学校\n创建于1989年的海南省特殊教育学校原属省教育厅直属正处级事业单位，为海南省惟一一所全日寄宿的公立特殊教育学校。\n    我市“接管”省特殊教育学校之后，将继续面向全省招收视障、听障两类适龄儿童，优化教育布局调整，促进特殊教育又好又快发展。\n类别：教育",
        "\n\n新闻内容：\n９月７日特稿（加１）（美国－大学流感）\n美一大学两千学生恐染流感\n　　　　马震\n　　美国华盛顿州立大学大约２０００名学生报告甲型Ｈ１Ｎ１流感症状。校方和医护人员说，这可能是最严重的一起大学生感染新型流感事件。\n　　（小标题）人数众多\n　　这所大学位于华盛顿州普尔曼，主校区大约有１.９万名学生。据美国《纽约时报》网络版６日报道，华盛顿州注册护士萨莉·雷德曼证实了大约２０００名华盛顿州立大学学生报告流感症状一事。\n　　雷德曼在华盛顿州立大学学生医疗部门工作。她说，流感暴发情况出现在８月２１日，那时学校还没开学。但如今为学生提供医疗服务的部门总是门庭若市。有一天，大约有２００名学生就诊或给医疗机构打电话报告喉咙疼、发烧、咳嗽等症状。\n　　华盛顿州立大学所在惠特曼县的卫生部门官员说，州实验室上周的检测结果显示，这所大学的疫情确实是因甲型Ｈ１Ｎ１流感病毒引起。\n　　学校现已开学。法新社本月６日报道，学校上周开了关于流感疫情的博客，博客上最新的信息说：“秋季学期的前１０天，我们估计已与大约２０００名有流感症状的人联络。”\n　　校方管理人员说，一些学生可能到社区医院就诊，一些学生可能居家自我治疗。校方无法掌握这些人的人数，已要求当地卫生部门提供相关数据，以便校方更好了解疫情情况。\n　　（小标题）无一死亡\n　　华盛顿州立大学已根据国家疾病控制和预防中心的防流感指南向学生提供咨询服务，以避免疫情进一步加重。学校还向学生发放了一些防流感的药品和护具等。\n　　为防止甲型流感传播，美国的一些大学已建立起隔离机制，但华盛顿州立大学没有类似机制。雷德曼说，在华盛顿州立大学上报的大部分流感疫情案例中，疑似染病的学生被要求待在居所内休息并吃退烧药。如果这些人在不吃退烧药２４小时后体温仍旧正常，就可以正常来上课。\n　　美国已有５９３例与甲型流感有关的死亡病例，但华盛顿州立大学尚未发现一起死亡病例。到目前为止，学生的流感症状相对温和，只有两个不是学生的患者入院治疗。\n　　校方在声明中说：“我校患者中的绝大部分症状温和，通常３到５天就能见强。”\n　　（小标题）担心传播\n　　华盛顿州立大学大规模流感疫情出现前，美国大学健康协会于８月２８日对１６５所大学实施了流感疫情调查。调查结果显示，全国超过２０００名学生报告说有甲型流感症状。\n　　惠特曼县公共卫生部门负责人蒂莫西·穆迪认为本月晚些时候开学的其他大学可能会遭遇类似华盛顿州立大学的情况，而地方医疗机构会担心疫情可能向校外蔓延。\n　　国家疾病控制和预防中心主任托马斯·弗里登６日接受美国有线电视新闻网采访时说，学校医务人员本学年报告的流感数字不同寻常。疾病控制和预防中心此前未遭遇过８月和９月数字增长这么快的情况。\n　　国家疾病控制和预防中心现在特别重视流感疫情。弗里登说：“如果它的致命性增加，可能会造成特别严重的情形，可能会给上学和上班的人带来特别多麻烦。”（完）（新华社供本报特稿）\n　　关键词：华盛顿州立大学(Washington State University)\n类别：医药、卫生",
        "\n\n新闻内容：\n在国防教育的落实上下功夫\n在国防教育的落实上下功夫 赵荣\n    加强全民国防教育是增强国防观念和忧患意识、促进国防和军队建设的基础性工程。鉴此，在今后的实践中，要坚持以科学发展观为指导，科学谋划、创新形式、狠抓落实，使全民国防教育深入人心，扎实有效地开展下去。\n    抓好责任落实。《国防教育法》第三章第十八条规定：各地区各部门的领导人员应当依法履行组织、领导本地区、本部门开展国防教育的职责。因而，要使全民国防教育扎实有效地开展下去，各级领导和职能部门要依法负起抓好全民国防教育的责任，对本地区、本单位、本行业的国防教育，从计划安排到组织实施都要认真负责地抓好落实。\n    抓好人员落实。国防教育是面向全民的教育，它的开展必须面向全社会，而不能只针对个别地区、个别单位和个别人员。因而，各地要对一切有接受能力的公民实施国防教育，以提高全民的政治、思想和道德素质，使全体公民积极争当热爱祖国、热爱国防的好公民。\n    抓好效果落实。国防教育的开展，效果的落实极为重要。为此，教育中应着重抓好国防理论、国防精神、国防知识、国防历史、国防技能、国防法制的教育，以强化爱国精神、增长国防知识、强化国防观念。通过教育，使全体公民进一步了解我国安全面临的新形势、世界军事变革的新发展、我国国防和军队建设面临的新挑战、以及在对国防建设中应承担的义务和责任等，不断提高他们支持和关心国防建设的积极性和自觉性。\n    (来源：中国国防报 发布时间： 2007-11-22 08:19)\n类别：军事",
        "\n\n新闻内容：\n中国又一学者当选瑞典皇家工程科学院外籍院士\n新华社北京８月２０日电　北京航空航天大学中国循环经济研究中心主任、北京循环经济促进会会长吴季松教授，日前被瑞典皇家工程科学院全体大会选为该院外籍院士。\n　　作为改革开放后首批出国访问学者之一，吴季松曾在欧洲原子能联营法国原子能委员会研究受控热核聚变，还曾任中国常驻联合国教科文组织代表团参赞衔副代表、联合国教科文组织科技部门高技术与环境顾问。１９８５至１９８６年，主持联合国教科文组织“多学科综合研究应用于经济发展”专题研究，并由联合国教科文组织发表项目研究报告创意知识经济。\n    他在中国科技和产业领域作出了多项贡献，主要包括：创意“知识经济”并将科技园区的实践介绍到中国、提出修复生态系统理论并主持制定水资源规划、创立新循环经济学等。\n　　瑞典皇家工程科学院创建于１９１９年，是世界上第一个工程院，现有机械工程、电机工程等学部。该院参与相关诺贝尔奖项的提名和评审工作。目前共有院士（含外籍院士）近１１００人，来自中国的外籍院士包括宋健、徐冠华等。（完）\n类别：科学技术",
    ]
    LONGBENCH_1000TOKENS_COMPRESSED_MULTIPLE_CONTEXT_PROMPT = "\n 新闻内容 第38届世界贸易中心年会及经贸洽谈会\n 安那州首府新奥尔良召开。\n 易服务管理总局、新奥尔良世贸中心共同举办\n 家和地区的经贸代表团约600余人与会。 天津贸促会与天津世贸中心协\n 会将共同组织天津经贸代表团赴美国参加“世贸中心2007年年会及经贸\n 洽谈会”。\n 联系人:王岭 刘鹏\n 电话:022-2520231725202123\n 传真:022-25201975\n 地址:天津经济 技术开发区宏达街19号A区2楼\n类别：商业、外贸、海关\n\n\n 新闻内容\n 海口“接管”省 特殊教育 学校\n 创建于1989年的海南省特殊教育 学校原属省教育 厅直属正处级事业单位,为海南省惟一一所全日寄宿的公立特殊教育 学校。\n教育 学校之后,将继续面向全省招收视障、听障两类适龄儿童教育 布局调整教育。\n类别：教育\n\n\n 中国又一学者当选瑞典皇家工程科学院外籍院士\n 新华社北京8月20日电 北京航空航天大学中国循环经济 研究中心主任、北京循环经济 促进会会长吴季松教授,日前被瑞典皇家工程科学院全体大会选为该院外籍院士。\n 作为改革开放后首批出国访问学者之一,吴季松曾在欧洲原子能联营法国原子能委员会研究受控热核聚变,还曾任中国常驻联合国教科文组织代表团参赞衔副代表、联合国教科文组织科技部门高技术与环境顾问。 1985至1986年,主持联合国教科文组织“多学科综合研究应用于经济 发展”专题研究经济。\n:创意“知识经济 ”并将科技园区的实践介绍到中国、提出修复生态系统理论并主持制定水资源规划、创立新循环经济 学等。\n 瑞典皇家工程科学院创建于1919年,是世界上第一个工程院,现有机械工程、电机工程等学部。 目前共有院士(含外籍院士)近1100人,来自中国的外籍院士包括宋健、徐冠华等。\n类别：科学技术"

    def __init__(self, *args, **kwargs):
        super(LLMLingua2Tester, self).__init__(*args, **kwargs)
        self.llmlingua = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            device_map="cpu",
            use_llmlingua2=True,
        )

    def test_general_compress_prompt(self):
        compressed_prompt = self.llmlingua.compress_prompt(
            self.PROMPT,
            rate=0.33,
            force_tokens=["\n", ".", "!", "?"],
            drop_consecutive=False,
            force_reserve_digit=False,
        )
        self.assertEqual(
            compressed_prompt["compressed_prompt"],
            self.COMPRESSED_SINGLE_CONTEXT_PROMPT,
        )
        self.assertEqual(compressed_prompt["origin_tokens"], 98)
        self.assertEqual(compressed_prompt["compressed_tokens"], 30)
        self.assertEqual(compressed_prompt["ratio"], "3.3x")
        self.assertEqual(compressed_prompt["rate"], "30.6%")

        compressed_prompt = self.llmlingua.compress_prompt(
            self.PROMPT.split("\n\n"),
            target_token=40,
            use_context_level_filter=True,
            force_tokens=["\n", ".", "!", "?"],
            drop_consecutive=False,
            force_reserve_digit=False,
        )
        self.assertEqual(
            compressed_prompt["compressed_prompt"],
            self.COMPRESSED_MULTIPLE_CONTEXT_PROMPT,
        )
        self.assertEqual(compressed_prompt["origin_tokens"], 98)
        self.assertEqual(compressed_prompt["compressed_tokens"], 34)
        self.assertEqual(compressed_prompt["ratio"], "2.9x")
        self.assertEqual(compressed_prompt["rate"], "34.7%")

        # Single Context
        compressed_prompt = self.llmlingua.compress_prompt(
            self.GSM8K_PROMPT.split("\n\n")[0],
            target_token=170,
            force_tokens=[
                "+",
                "-",
                "*",
                "×",
                "/",
                "÷",
                "=",
                "The answer is",
                "\n",
                "Question:",
            ],
            drop_consecutive=False,
            force_reserve_digit=True,
        )
        self.assertEqual(
            compressed_prompt["compressed_prompt"],
            self.GSM8K_150TOKENS_COMPRESSED_SINGLE_CONTEXT_PROMPT,
        )
        self.assertEqual(compressed_prompt["origin_tokens"], 422)
        self.assertEqual(compressed_prompt["compressed_tokens"], 203)
        self.assertEqual(compressed_prompt["ratio"], "2.1x")
        self.assertEqual(compressed_prompt["rate"], "48.1%")

        # Single Context
        compressed_prompt = self.llmlingua.compress_prompt(
            self.MEETINGBANK_PROMPT.split("\n\n")[0],
            target_token=150,
            force_tokens=["\n", ".", "?", "!"],
            drop_consecutive=True,
            force_reserve_digit=False,
        )
        self.assertEqual(
            compressed_prompt["compressed_prompt"],
            self.MEETINGBANK_150TOKENS_COMPRESSED_SINGLE_CONTEXT_PROMPT,
        )
        self.assertEqual(compressed_prompt["origin_tokens"], 464)
        self.assertEqual(compressed_prompt["compressed_tokens"], 154)
        self.assertEqual(compressed_prompt["ratio"], "3.0x")
        self.assertEqual(compressed_prompt["rate"], "33.2%")

        # Multiple Context
        compressed_prompt = self.llmlingua.compress_prompt(
            self.GSM8K_PROMPT.split("\n\n"),
            target_token=150,
            use_context_level_filter=True,
            force_tokens=["+", "-", "*", "×", "/", "÷", "=", "The answer is", "\n"],
            drop_consecutive=False,
            force_reserve_digit=True,
        )
        self.assertEqual(
            compressed_prompt["compressed_prompt"],
            self.GSM8K_150TOKENS_COMPRESSED_MULTIPLE_CONTEXT_PROMPT,
        )
        self.assertEqual(compressed_prompt["origin_tokens"], 726)
        self.assertEqual(compressed_prompt["compressed_tokens"], 161)
        self.assertEqual(compressed_prompt["ratio"], "4.5x")
        self.assertEqual(compressed_prompt["rate"], "22.2%")

        # Multiple Context
        compressed_prompt = self.llmlingua.compress_prompt(
            self.LONGBENCH_PROMPT_LIST,
            target_token=1000,
            use_context_level_filter=True,
            force_tokens=[
                "\n",
                "。",
                "：",
                "？",
                "类别：",
                "农业、农村",
                "军事",
                "文学、艺术",
                "体育",
                "传媒业",
                "电子信息产业",
                "文化、休闲娱乐",
                "社会、劳动",
                "经济",
                "服务业、旅游业",
                "环境、气象",
                "能源、水务、水利",
                "财政、金融",
                "教育",
                "科学技术",
                "对外关系、国际关系",
                "矿业、工业",
                "政治",
                "交通运输、邮政、物流",
                "灾难、事故",
                "基本建设、建筑业、房地产",
                "医药、卫生",
                "法律、司法",
                "商业、外贸、海关",
            ],
            drop_consecutive=True,
            force_reserve_digit=False,
        )
        self.assertEqual(
            compressed_prompt["compressed_prompt"],
            self.LONGBENCH_1000TOKENS_COMPRESSED_MULTIPLE_CONTEXT_PROMPT,
        )
        self.assertEqual(compressed_prompt["origin_tokens"], 8389)
        self.assertEqual(compressed_prompt["compressed_tokens"], 870)
        self.assertEqual(compressed_prompt["ratio"], "9.6x")
        self.assertEqual(compressed_prompt["rate"], "10.4%")
