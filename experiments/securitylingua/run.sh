python label_word.py \
    --load_prompt_from liyucheng/jailbreak-pairs \
    --window_size 400 \
    --save_path ../results/security_lingua/jailbreak_pairs_annotated.pt

python filter.py \
    --load_path ../results/security_lingua/jailbreak_pairs_annotated.pt \
    --save_path ../results/security_lingua/jailbreak_pairs_annotated_filtered.pt

python train_roberta.py \
    --data_path ../results/security_lingua/jailbreak_pairs_annotated_filtered.pt \
    --save_path ../results/security_lingua/jailbreak_pairs_annotated_filtered_roberta.pt \
    --model_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
    --num_epoch 5 \
    --run_name meetbank_slingua \
    --wandb_project slingua \
    --wandb_name meetbank_slingua

# Multi-GPU training
# ACCELERATE_LOG_LEVEL="ERROR" accelerate launch --num_processes 4 experiments/llmlingua2/model_training/train_roberta.py \
#     --data_path experiments/llmlingua2/results/security_lingua/v8_filtered.pt  \
#     --save_path experiments/llmlingua2/results/models/xlm_slingua_v8.pth \
#     --num_epoch 5 \
#     --run_name xlm_slingua_v8 \
#     --wandb_project slingua \
#     --wandb_name xlm_slingua_v8 \
#     --batch_size 32

