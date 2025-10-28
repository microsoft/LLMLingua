# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import os
import random
import time
from typing import List, Tuple, Dict

import torch
from sklearn.metrics import accuracy_score
from torch import cuda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer
from utils import TokenClfDataset
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

# Constants
MAX_LEN = 512
MAX_GRAD_NORM = 10

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="train bert to do compression (by token classification)"
    )
    parser.add_argument(
        "--model_name",
        help="token classification model",
        default="FacebookAI/xlm-roberta-large",
    )
    parser.add_argument(
        "--data_path",
        help="training and validation data path",
        default="../../../results/meetingbank/gpt-4-32k_comp/annotation_kept_cs512_meetingbank_train_formated.pt",
    )
    parser.add_argument(
        "--label_type",
        help="word label or token label",
        default="word_label",
        choices=["word_label", "token_label"],
    )
    parser.add_argument(
        "--save_path",
        help="save path",
        default="../../../results/models/xlm_roberta_large_meetingbank_only.pth",
    )
    parser.add_argument(
        "--run_name",
        help="run name",
        default="xlm_roberta_large_meetingbank_only",
    )
    parser.add_argument("--lr", help="learning rate", default=1e-5, type=float)
    parser.add_argument(
        "--num_epoch", help="number of training epoch", default=10, type=int
    )
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument(
        "--wandb_project",
        help="wandb project name. If not provided, wandb will not be used",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--wandb_name",
        help="wandb run name",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization",
    )
    return parser.parse_args()

def setup_wandb(project: str, name: str, accelerator: Accelerator) -> bool:
    """Setup wandb tracking if project is provided and only on main process"""
    if project is None:
        return False
    
    # Only init wandb on main process
    if accelerator.is_main_process:
        import wandb
        wandb.init(project=project, name=name)
    return True

def load_and_split_data(data_path: str, seed: int = 42) -> Tuple[List[Tuple[str, List]], List[Tuple[str, List]]]:
    """Load and split data into train and validation sets"""
    data = torch.load(data_path, weights_only=False)
    text_label = [(text, label) for text, label in zip(data["origin"], data["labels"])]
    set_seed(seed)
    random.shuffle(text_label)
    
    split_idx = int(len(text_label) * 0.9)
    train_data = text_label[:split_idx]
    val_data = text_label[split_idx:]
    
    return train_data, val_data

def prepare_datasets(
    train_data: List[Tuple[str, List]], 
    val_data: List[Tuple[str, List]], 
    tokenizer, 
    model_name: str
) -> Tuple[TokenClfDataset, TokenClfDataset]:
    """Prepare training and validation datasets"""
    train_text = [text for text, label in train_data]
    train_label = [label for text, label in train_data]
    val_text = [text for text, label in val_data]
    val_label = [label for text, label in val_data]

    train_dataset = TokenClfDataset(
        train_text, train_label, MAX_LEN, tokenizer=tokenizer, model_name=model_name
    )
    val_dataset = TokenClfDataset(
        val_text, val_label, MAX_LEN, tokenizer=tokenizer, model_name=model_name
    )
    
    return train_dataset, val_dataset

def train_epoch(
    model: AutoModelForTokenClassification,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    epoch: int,
    use_wandb: bool = False,
) -> None:
    """Train for one epoch"""
    model.train()
    tr_loss, tr_accuracy = 0, 0
    nb_tr_steps = 0

    # Get num_labels from unwrapped model
    num_labels = accelerator.unwrap_model(model).num_labels

    # Add progress bar
    progress_bar = tqdm(
        train_dataloader,
        desc=f"Training Epoch {epoch}",
        disable=not accelerator.is_local_main_process
    )

    for batch in progress_bar:
        outputs = model(
            input_ids=batch["ids"],
            attention_mask=batch["mask"],
            labels=batch["targets"]
        )
        loss, tr_logits = outputs.loss, outputs.logits
        
        accelerator.backward(loss)

        tr_loss += loss.item()
        nb_tr_steps += 1

        # Calculate accuracy
        flattened_targets = batch["targets"].view(-1)
        active_logits = tr_logits.view(-1, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)
        active_accuracy = batch["mask"].view(-1) == 1
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        tmp_tr_accuracy = accuracy_score(
            targets.cpu().numpy(), predictions.cpu().numpy()
        )
        tr_accuracy += tmp_tr_accuracy

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{tr_loss/nb_tr_steps:.4f}',
            'accuracy': f'{tr_accuracy/nb_tr_steps:.4f}'
        })

        # Log metrics to wandb
        if nb_tr_steps % 100 == 0 and use_wandb and accelerator.is_main_process:
            wandb.log({
                "train/loss": tr_loss / nb_tr_steps,
                "train/accuracy": tr_accuracy / nb_tr_steps,
                "train/step": nb_tr_steps + epoch * len(train_dataloader)
            })

        # Optimize
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )
        optimizer.step()
        optimizer.zero_grad()

    # Print epoch metrics
    tr_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {tr_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")

def evaluate(
    model: AutoModelForTokenClassification,
    eval_dataloader: DataLoader,
    accelerator: Accelerator,
    epoch: int,
    use_wandb: bool = False,
) -> float:
    """Evaluate the model"""
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0

    # Get num_labels from unwrapped model
    num_labels = accelerator.unwrap_model(model).num_labels

    # Add progress bar
    progress_bar = tqdm(
        eval_dataloader,
        desc=f"Evaluating Epoch {epoch}",
        disable=not accelerator.is_local_main_process
    )

    with torch.no_grad():
        for batch in progress_bar:
            outputs = model(
                input_ids=batch["ids"],
                attention_mask=batch["mask"],
                labels=batch["targets"]
            )
            loss, eval_logits = outputs.loss, outputs.logits
            eval_loss += loss.item()
            nb_eval_steps += 1

            # Calculate accuracy
            flattened_targets = batch["targets"].view(-1)
            active_logits = eval_logits.view(-1, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)
            active_accuracy = batch["mask"].view(-1) == 1
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            tmp_eval_accuracy = accuracy_score(
                targets.cpu().numpy(), predictions.cpu().numpy()
            )
            eval_accuracy += tmp_eval_accuracy

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{eval_loss/nb_eval_steps:.4f}',
                'accuracy': f'{eval_accuracy/nb_eval_steps:.4f}'
            })

    # Calculate and log metrics
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    if use_wandb and accelerator.is_main_process:
        wandb.log({
            "eval/loss": eval_loss,
            "eval/accuracy": eval_accuracy,
            "eval/epoch": epoch
        })

    return eval_accuracy

def main():
    args = parse_arguments()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Initialize accelerator first
    accelerator = Accelerator()
    logger = get_logger(__name__)
    
    # Setup wandb after accelerator
    use_wandb = setup_wandb(args.wandb_project, args.wandb_name, accelerator)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=2, ignore_mismatched_sizes=True
    )
    
    # Prepare data
    train_data, val_data = load_and_split_data(args.data_path, args.seed)
    train_dataset, val_dataset = prepare_datasets(
        train_data, val_data, tokenizer, args.model_name
    )
    
    logger.info(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
    
    # Training loop
    best_acc = 0
    for epoch in tqdm(range(args.num_epoch)):
        logger.info(f"Training epoch: {epoch + 1}")
        train_epoch(model, train_dataloader, optimizer, accelerator, epoch, use_wandb)
        acc = evaluate(model, val_dataloader, accelerator, epoch, use_wandb)
        
        if acc > best_acc:
            best_acc = acc
            # Unwrap model before saving
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), args.save_path)

if __name__ == "__main__":
    main()
