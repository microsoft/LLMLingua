# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# python format_data.py

# python compress.py --load_origin_from ../results/meetingbank/origin/meetingbank_train_formated.json \
#     --compressor gpt4 \
#     --chunk_size 512 \
#     --save_path ../results/meetingbank/gpt-4-32k_comp/compression_cs512_meetingbank_train_formated.json

python label_word.py --load_prompt_from microsoft/MeetingBank-LLMCompressed \
    --window_size 400 \
    --save_path ../results/meetingbank/gpt-4-32k_comp/annotation_cs512_meetingbank_train_formated.json

python filter.py --load_path ../results/meetingbank/gpt-4-32k_comp/annotation_cs512_meetingbank_train_formated.pt \
    --save_path ../results/meetingbank/gpt-4-32k_comp/annotation_kept_cs512_meetingbank_train_formated.pt
