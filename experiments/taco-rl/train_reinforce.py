import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModelForTokenClassification
import torch.nn.functional as F
import random
import json
from tqdm.auto import tqdm
import torch.nn as nn
from torch.distributions import Categorical
from llmlingua.taco_rl import PromptCompressorReinforce
from metrics import evaluate_sim2
from utils import query_llm
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as DDPK
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
from csv_logger import CsvLogger
import logging
import regex as re
import hydra
import gc
import os
import warnings
import math


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# turn off logging
logging.disable(logging.CRITICAL)

# kwargs = DDPK(find_unused_parameters=True)
kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
accelerator = Accelerator(kwargs_handlers=[kwargs])

input_dict={}
@hydra.main(config_path="configs", config_name="train_reinforce.yaml", version_base=None)
def run(cfg):
    input_dict["train_data_path"] = cfg.data.train_file_path
    input_dict["model_load_path_hf"] = cfg.model.model_load_path
    input_dict["model_save_path_hf"] = cfg.model.trained_model_output_dir_hf
    input_dict["num_epochs"] = cfg.hyperparams.epochs
    input_dict["compression_rate"] = cfg.hyperparams.compression_rate
    input_dict["policy_lr"] = cfg.hyperparams.policy_lr
    input_dict["train_batch_size"] = cfg.hyperparams.train_batch_size
    input_dict["compression_relaxation_tokens"] = cfg.hyperparams.compression_relaxation_tokens
    input_dict["prompt"] = cfg.task.prompt
    input_dict["gpt_model"] = cfg.task.gpt_model
    input_dict["max_tokens"] = cfg.task.max_tokens
    input_dict["target_token"] = cfg.hyperparams.target_token
    input_dict["use_context_level_filter"] = cfg.hyperparams.use_context_level_filter
    input_dict["use_token_level_filter"] = cfg.hyperparams.use_token_level_filter
    input_dict["target_context"] = cfg.hyperparams.target_context
    input_dict["context_level_compression_rate"] = cfg.hyperparams.context_level_compression_rate
    input_dict["context_level_target_token"] = cfg.hyperparams.context_level_target_token
    input_dict["force_tokens"] = cfg.hyperparams.force_tokens
    input_dict["drop_consecutive"] = cfg.hyperparams.drop_consecutive
    input_dict["force_reserve_digit"] = cfg.hyperparams.force_reserve_digit
    input_dict["max_seq_len"] = cfg.hyperparams.max_seq_len
    
    # Logging settings
    input_dict["log_dir"] = cfg.logging.log_dir
    input_dict["log_interval"] = cfg.logging.log_interval
    input_dict["save_interval"] = cfg.logging.save_interval
    
    # Device settings
    input_dict["use_cuda"] = cfg.device.use_cuda
    input_dict["device_map"] = cfg.device.device_map

    # REINFORCE parameters
    input_dict["entropy_coeff"] = cfg.hyperparams.entropy_coeff

run()

# Create log directory if it doesn't exist
os.makedirs(input_dict["log_dir"], exist_ok=True)

if accelerator.is_local_main_process:
    delimiter = ","
    level_names = ["Train", "Val"]
    reward_level_names = ["TrainR", "ValR"]
    csv_logger = CsvLogger(filename=f"{input_dict['log_dir']}/reinforce_logs_updated.csv",fmt = f'%(levelname)s{delimiter}%(message)s', add_level_names= level_names,header=["Train/Val", "Epoch", "Step", "Policy Loss", "LR"])
    rewards_logger = CsvLogger(filename=f"{input_dict['log_dir']}/rewards_logs_updated.csv",fmt = f'%(levelname)s{delimiter}%(message)s', add_level_names= reward_level_names,header=["Train/Val", "Epoch", "Step", "Rewards", "Comp Ratio"])

def get_model_output(input_prompt_list):
    for i in range(len(input_prompt_list)):
        input_prompt_list[i] = re.sub(r'\n', ' ', input_prompt_list[i])

    compressed_output_dict = compressor.compress_prompt_llmlingua2(
        input_prompt_list,
        rate=compression_rate,
        target_token=target_token,
        use_context_level_filter=use_context_level_filter,
        use_token_level_filter=use_token_level_filter,
        target_context=target_context,
        context_level_rate=context_level_compression_rate,
        context_level_target_token=context_level_target_token,
        force_tokens=force_tokens,
        drop_consecutive=drop_consecutive,
        force_reserve_digit=force_reserve_digit,
    )

    return compressed_output_dict['compressed_prompt'], compressed_output_dict['compressed_prompt_list'], compressed_output_dict['compression_ratios'], compressed_output_dict["actions"], compressed_output_dict["old_log_probs"], compressed_output_dict["old_logits"], compressed_output_dict["compressed_prompt_list_2"], compressed_output_dict["entropy"]


class REINFORCE:
    def __init__(self, policy_network, policy_optimizer, policy_scheduler):
        self.policy_network = policy_network
        self.policy_optimizer = policy_optimizer
        self.policy_scheduler = policy_scheduler

    def update(self, rewards_and_comp_ratios, old_log_probs, entropies, epoch_counter, steps):
        rewards = [reward["reward"] for reward in rewards_and_comp_ratios]
        comp_ratios = [reward["comp_ratio"] for reward in rewards_and_comp_ratios]
        
        rewards = torch.FloatTensor(rewards).to(accelerator.device)
        rewards = rewards.unsqueeze(1).detach()

        comp_ratios = torch.FloatTensor(comp_ratios).to(accelerator.device)
        comp_ratios = comp_ratios.unsqueeze(1).detach()

        entropies = torch.stack(entropies)
        old_log_probs = torch.stack(old_log_probs)

        # Enhanced policy loss with entropy regularization
        policy_loss = -torch.mean(old_log_probs * (rewards)) - input_dict["entropy_coeff"]*torch.mean(entropies)
        
        self.policy_optimizer.zero_grad()

        if steps % input_dict["log_interval"] == 0:
            print(f"Avg. Entropy: {torch.mean(entropies)}")

        if accelerator.is_local_main_process:
            csv_logger.Train([epoch_counter, steps, policy_loss.item(), self.policy_scheduler.get_last_lr()[0]])

        accelerator.backward(policy_loss)
        self.policy_optimizer.step()
        self.policy_scheduler.step()

def get_gpt_output(input_text):
    query = prompt.format(transcript=input_text)
    gpt_output = query_llm(
        prompt=query,
        model_name=input_dict["gpt_model"],
        max_tokens=input_dict["max_tokens"],
    )

    return gpt_output

def calculate_rewards(gpt_output, model_compressed_text, compression_ratio):
    if compression_ratio["compressed"] > 0.1*compression_ratio["original"]:
        org_input_summary = gpt_output
        model_compressed_summary = get_gpt_output(model_compressed_text)
        comp_ratio = compression_ratio["original"] / compression_ratio["compressed"]
    else:
        print(compression_ratio)
        comp_ratio = compression_ratio["original"] / compression_ratio["compressed"]
        return {"comp_ratio": comp_ratio, "reward": -0.4}       

    if org_input_summary is not None and model_compressed_summary is not None:
        eval_scores = evaluate_sim2([model_compressed_summary], [org_input_summary])
        bleu = eval_scores["bleu"]
        comp_coeff = compression_ratio["compressed"] - compression_rate*compression_ratio["original"]
        
        if abs(comp_coeff) > input_dict["compression_relaxation_tokens"]:
            reward = -0.1
        else:
            reward = bleu

        return {"comp_ratio": comp_ratio, "reward": reward}
    else:
        return {"comp_ratio": comp_ratio, "reward": 0.4}

def train(dataloader, num_epochs):
    epoch_counter = 0
    steps = 0
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader)*(num_epochs-epoch))):
            input_prompt_list = batch["org_prompt"]
            gpt_output_list = batch["gpt_output"]
            
            compressed_prompt, compressed_prompt_list, compression_ratio_list, actions, old_log_probs, _, compressed_prompt_list_2, entropies = get_model_output(input_prompt_list)

            rewards_and_comp_ratios = [calculate_rewards(gpt_output_list[i], compressed_prompt_list_2[i], compression_ratio_list[i]) for i in range(len(gpt_output_list))]
            rewards = [reward["reward"] for reward in rewards_and_comp_ratios]
            comp_ratios = [reward["comp_ratio"] for reward in rewards_and_comp_ratios]

            # update the policy network
            reinforce_agent.update(rewards_and_comp_ratios, old_log_probs, entropies, epoch_counter, steps)

            # Enhanced logging with more metrics
            if batch_idx % input_dict["log_interval"] == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Avg. Batch Rewards: {sum(rewards)/len(rewards)}, Avg. Comp Ratio: {sum(comp_ratios)/len(comp_ratios)}")
                if accelerator.is_local_main_process:
                    rewards_logger.TrainR([epoch_counter, batch_idx//input_dict["log_interval"], sum(rewards)/len(rewards), sum(comp_ratios)/len(comp_ratios)])
            
            steps += 1

        epoch_counter += 1

def tokenize_text(text):
    tokenize_text = tokenizer(text, max_length=input_dict["max_seq_len"], padding="max_length", return_tensors="pt", truncation=True)
    return {k: v[0] for k, v in tokenize_text.items()}

class GenericRLDataset(Dataset):
    def __init__(self, data_path):
        self.data = json.load(open(data_path))
        # self.data = dict(list(self.data.items())[:100])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokenized_dataset = tokenize_text(self.data[str(idx)]["original_prompt"])
        return {"input_ids": tokenized_dataset["input_ids"], "attention_mask": tokenized_dataset["attention_mask"], "org_prompt": self.data[str(idx)]["original_prompt"], "gpt_output": self.data[str(idx)]["gpt_output"]}


# Example usage
device = torch.device("cuda" if torch.cuda.is_available() and input_dict["use_cuda"] else "cpu")

input_data_path = input_dict["train_data_path"]
model_name = input_dict["model_load_path_hf"]

compression_rate = input_dict["compression_rate"]
target_token = input_dict["target_token"]
use_context_level_filter = input_dict["use_context_level_filter"]
use_token_level_filter = input_dict["use_token_level_filter"]
target_context = input_dict["target_context"]
context_level_compression_rate = input_dict["context_level_compression_rate"]
context_level_target_token = input_dict["context_level_target_token"]
force_tokens = input_dict["force_tokens"]
drop_consecutive = input_dict["drop_consecutive"]
force_reserve_digit = input_dict["force_reserve_digit"]

# Use the new PromptCompressorReinforce class
compressor = PromptCompressorReinforce(
    model_name=model_name,
    model_config={},
    use_llmlingua2=True,
    device_map=input_dict["device_map"],
) 

tokenizer = compressor.tokenizer

prompt = input_dict["prompt"]

policy_optimizer = optim.AdamW(compressor.model.parameters(), lr=input_dict["policy_lr"])

dataset = GenericRLDataset(input_data_path)
dataloader = DataLoader(dataset, batch_size=input_dict["train_batch_size"], shuffle=True)

num_epochs = input_dict["num_epochs"]

policy_scheduler = optim.lr_scheduler.CosineAnnealingLR(policy_optimizer, T_max=num_epochs*len(dataloader))

compressor.model, policy_optimizer, policy_scheduler, dataloader = accelerator.prepare(
    compressor.model, policy_optimizer, policy_scheduler, dataloader
)

policy_net = compressor.model

reinforce_agent = REINFORCE(policy_net, policy_optimizer, policy_scheduler)

train(dataloader, num_epochs=num_epochs)

# save the model
unwrapped_model = accelerator.unwrap_model(policy_net)
unwrapped_model.save_pretrained(
    input_dict["model_save_path_hf"],
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
) 