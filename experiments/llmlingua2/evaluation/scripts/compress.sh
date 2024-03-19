# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

python compress.py --load_origin_from ../../../results/meetingbank_short/origin/meetingbank_test_3qa_pairs_summary_formated.json \
    --compression_rate 0.33 \
    --force_tokens "\n,?,!,." \
    --save_path ../../../results/meetingbank_short/llmlingua2/compression_ratio33_meetingbank_test_3qa_pairs_summary_formated.json

python compress.py --load_origin_from ../../../results/longbench/origin/longbench_test_single_doc_qa_formated.json \
    --target_token 2000 \
    --force_tokens "\n,?,!,." \
    --save_path ../../../results/longbench/llmlingua2/compression_target2000_longbench_test_single_doc_qa_formated.json

python compress.py --load_origin_from ../../../results/zero_scrolls/origin/zero_scrolls_validation.json \
    --target_token 2000 \
    --force_tokens "\n,?,!,." \
    --save_path ../../../results/zero_scrolls/llmlingua2/compression_target2000_zero_scrolls_validation.json

python compress.py --load_origin_from ../../../results/gsm8k/origin/gsm8k_cot_example_all_in_one.json \
    --load_key prompt_list \
    --target_token 250 \
    --force_tokens "+,-,*,×,/,÷,=,The answer is,\n" \
    --use_context_level_filter \
    --force_reserve_digit \
    --save_path ../../../results/gsm8k/llmlingua2/compression_target250_gsm8k_cot_example_all_in_one.json
