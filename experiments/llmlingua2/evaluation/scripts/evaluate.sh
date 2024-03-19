# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

python eval_meetingbank_qa.py --load_prompt_from ../../../results/meetingbank_short/llmlingua2/compression_ratio33_meetingbank_test_3qa_pairs_summary_formated.json \
    --load_key compressed_prompt \
    --model_name_or_path gpt-35-turbo-instruct \
    --save_path ../../../results/meetingbank_short/llmlingua2/gpt35_answer/answer_ratio33_meetingbank_test_3qa_pairs_summary_formated.json

python eval_longbench.py --load_prompt_from ../../../results/longbench/llmlingua2/compression_target2000_longbench_test_single_doc_qa_formated.json \
    --load_key compressed_prompt \
    --model_name_or_path gpt-35-turbo-instruct \
    --save_path ../../../results/longbench/llmlingua2/gpt35_answer/answer_target2000_longbench_test_single_doc_qa_formated.json

python eval_zero_scrolls.py --load_prompt_from ../../../results/zero_scrolls/llmlingua2/compression_target2000_zero_scrolls_validation.json \
    --load_key compressed_prompt \
    --model_name_or_path gpt-35-turbo-instruct \
    --save_path ../../../results/zero_scrolls/llmlingua2/gpt35_answer/answer_target2000_zero_scrolls_validation.json

python eval_gsm8k.py --load_prompt_from ../../../results/gsm8k/llmlingua2/compression_target200_gsm8k_cot_example_all_in_one.json \
    --load_key compressed_prompt_list \
    --model_name_or_path gpt-35-turbo-instruct \
    --save_path ../../../results/gsm8k/llmlingua2/gpt35_answer/answer_target200_gsm8k_cot_example_all_in_one.json
