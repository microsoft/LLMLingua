conda create -n llmlingua python=3.10 -y && conda activate llmlingua
pip install -e .
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install accelerate wandb
pip install openai==0.28

pip install spacy
python -m spacy download en_core_web_sm
pip install scikit-learn
pip install tensorboard
pip install datasets hf_transfer

unset WANDB_RUN_ID WANDB_RUN_GROUP WANDB_PROJECT WANDB_NOTES WANDB_NAME
