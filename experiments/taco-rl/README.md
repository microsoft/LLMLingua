# TACO-RL Experiments: Training and Implementation Guide

This directory contains the experimental implementation of TACO-RL (Task-Aware Prompt Compression Optimization with Reinforcement Learning) for fine-tuning LLMLingua models on new tasks.

## Directory Structure

```
experiments/taco-rl/
├── README.md                 # This file
├── train_reinforce.py        # Main training script
├── utils.py                  # Utility functions and LLM querying
├── metrics.py                # Evaluation metrics
├── configs/                  # Configuration files
│   └── train_reinforce.yaml  # Training configuration
└── logs/                     # Training logs (created during training)
```

## Quick Start

### 1. Prepare Your Data

Create a JSON file with your training data in the following format:

```json
{
  "0": {
    "original_prompt": "Your original prompt text here...",
    "gpt_output": "Expected GPT output for the prompt..."
  },
  "1": {
    "original_prompt": "Another prompt...",
    "gpt_output": "Another expected output..."
  }
}
```

**Data Format Requirements:**
- `original_prompt`: The input text that needs to be compressed
- `gpt_output`: The expected output from the teacher model (e.g., GPT-3.5) for the original prompt

**Task-Specific Examples:**

**Meeting Summarization:**
```json
{
  "0": {
    "original_prompt": "Meeting transcript: John discussed Q1 results showing 15% growth in revenue. Sarah mentioned challenges with supply chain. Mike proposed new marketing strategy...",
    "gpt_output": "Summary: Q1 results were positive with 15% revenue growth. Supply chain challenges were identified. New marketing strategy was proposed."
  }
}
```

**Question Answering:**
```json
{
  "0": {
    "original_prompt": "Context: The Eiffel Tower was built in 1889 by Gustave Eiffel for the World's Fair. It stands 324 meters tall and was originally intended to be temporary. Question: When was the Eiffel Tower built?",
    "gpt_output": "The Eiffel Tower was built in 1889."
  }
}
```

**Code Summarization:**
```json
{
  "0": {
    "original_prompt": "def quicksort(arr): if len(arr) <= 1: return arr; pivot = arr[len(arr) // 2]; left = [x for x in arr if x < pivot]; middle = [x for x in arr if x == pivot]; right = [x for x in arr if x > pivot]; return quicksort(left) + middle + quicksort(right)",
    "gpt_output": "This function implements the quicksort algorithm using a divide-and-conquer approach with pivot selection and recursive sorting."
  }
}
```

### 2. Configure API Settings (Optional)

If you're using Azure OpenAI services, you can configure your API settings in `utils.py`:

```python
# In utils.py, update the DEFAULT_API_CONFIG with your settings:
DEFAULT_API_CONFIG = {
    "scope": "https://cognitiveservices.azure.com/.default",
    "client_id": "YOUR_CLIENT_ID_HERE",  # Replace with your client ID
    "api_details": {
        "primary": {
            "api_base": "YOUR_PRIMARY_API_BASE_HERE",  # Replace with your primary API base
            "api_version": "2024-02-01",
        },
        "secondary": {
            "api_base": "YOUR_SECONDARY_API_BASE_HERE",  # Replace with your secondary API base
            "api_version": "2024-02-01",
        }
    }
}
```

Alternatively, you can initialize with custom configuration in your training script:

```python
from utils import initialize_api_config

custom_config = {
    "client_id": "your-client-id",
    "api_details": {
        "primary": {"api_base": "your-primary-endpoint"},
        "secondary": {"api_base": "your-secondary-endpoint"}
    }
}
initialize_api_config(custom_config)
```

### 3. Configure Training Parameters

Edit `configs/train_reinforce.yaml` to set your desired hyperparameters:

```yaml
data:
  train_file_path: "path/to/your/train_data.json"

model:
  model_load_path: "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
  trained_model_output_dir_hf: "models/reinforce_trained_model"

task:
  prompt: "Summarize the provided text (which may be compressed).\n{transcript}\nSummary:"
  gpt_model: "gpt-35-turbo"
  max_tokens: 1500

hyperparams:
  epochs: 4
  train_batch_size: 8
  policy_lr: 1e-5
  compression_rate: 0.5
  compression_relaxation_tokens: 30
  max_seq_len: 512
  entropy_coeff: 0.01
  
  # Compression settings
  target_token: -1
  use_context_level_filter: false
  use_token_level_filter: true
  target_context: -1
  context_level_compression_rate: 1.0
  context_level_target_token: -1
  force_tokens: ""
  drop_consecutive: true
  force_reserve_digit: false

logging:
  log_dir: "logs_new"
  log_interval: 5
  save_interval: 100
  
device:
  use_cuda: true
  device_map: "auto"
```

### 4. Run Training

```bash
cd LLMLingua/experiments/taco-rl
accelerate launch train_reinforce.py
```

## Configuration Parameters

### Data Configuration
- `train_file_path`: Path to your training data JSON file

### Model Configuration
- `model_load_path`: Pre-trained LLMLingua model to fine-tune
- `trained_model_output_dir_hf`: Directory to save the fine-tuned model

### Task Configuration
- `prompt`: Template prompt for the teacher model (GPT-3.5). Use `{transcript}` as placeholder for the compressed text
- `gpt_model`: Teacher model name for reward computation
- `max_tokens`: Maximum tokens for teacher model responses

**Task-Specific Prompt Examples:**

**Meeting Summarization:**
```yaml
task:
  prompt: "Summarize the provided meeting transcript (which may be compressed).\n{transcript}\nSummary:"
```

**Question Answering:**
```yaml
task:
  prompt: "Answer the following question based on the provided context (which may be compressed).\n{transcript}\nAnswer:"
```

**Code Summarization:**
```yaml
task:
  prompt: "Summarize the following code (which may be compressed).\n{transcript}\nSummary:"
```

### Training Hyperparameters
- `epochs`: Number of training epochs
- `train_batch_size`: Batch size for training
- `policy_lr`: Learning rate for policy optimization
- `compression_rate`: Target compression ratio (0.0-1.0)
- `compression_relaxation_tokens`: Tolerance for compression deviation
- `max_seq_len`: Maximum sequence length for tokenization
- `entropy_coeff`: Entropy regularization coefficient for exploration

### Compression Settings
- `target_token`: Target number of tokens (-1 for rate-based compression)
- `use_context_level_filter`: Whether to use context-level filtering
- `use_token_level_filter`: Whether to use token-level filtering
- `target_context`: Target number of contexts (-1 for rate-based)
- `context_level_compression_rate`: Context-level compression rate
- `context_level_target_token`: Context-level target tokens
- `force_tokens`: Tokens to always preserve
- `drop_consecutive`: Whether to drop consecutive tokens
- `force_reserve_digit`: Whether to preserve digits

### Logging Configuration
- `log_dir`: Directory for log files
- `log_interval`: How often to log metrics
- `save_interval`: How often to save model checkpoints

### Device Configuration
- `use_cuda`: Whether to use CUDA if available
- `device_map`: Device mapping strategy

## Training Process

### 1. Data Loading
The training script uses the `GenericRLDataset` class to load training data:

```python
class GenericRLDataset(Dataset):
    def __init__(self, data_path):
        self.data = json.load(open(data_path))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokenized_dataset = tokenize_text(self.data[str(idx)]["original_prompt"])
        return {
            "input_ids": tokenized_dataset["input_ids"], 
            "attention_mask": tokenized_dataset["attention_mask"], 
            "org_prompt": self.data[str(idx)]["original_prompt"], 
            "gpt_output": self.data[str(idx)]["gpt_output"]
        }
```

### 2. Model Initialization
Creates a `PromptCompressorReinforce` instance with the specified pre-trained model:

```python
compressor = PromptCompressorReinforce(
    model_name=model_name,
    model_config={},
    use_llmlingua2=True,
    device_map=device_map,
)
```

### 3. REINFORCE Training Loop
For each batch:
1. Compresses prompts using the RL-enhanced compressor
2. Calculates rewards based on compression quality using the teacher model
3. Updates the policy network using REINFORCE algorithm
4. Logs training metrics and progress

## Reward Function

The reward function evaluates compression quality by:

1. **Compression Quality**: BLEU score between summaries of original and compressed text
2. **Compression Ratio**: Penalty for not meeting target compression rate
3. **Compression Coefficient**: Penalty for deviating from target compression
4. **Entropy Regularization**: Encourages exploration during training

### Reward Computation Details
The reward is positive when the compression is within the tolerance threshold. It is set to a negative value during over-compression and when it crosses the tolerance threshold.

```python
def calculate_rewards(gpt_output, model_compressed_text, compression_ratio):
    if compression_ratio["compressed"] > 0.1*compression_ratio["original"]:
        org_input_summary = gpt_output
        model_compressed_summary = get_gpt_output(model_compressed_text)
        comp_ratio = compression_ratio["original"] / compression_ratio["compressed"]
    else:
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
```
### Note: This function will need to be modified as per the task. For example, for the QA task, the *reward* should be F1 Score.

## Output

The training script produces:

### 1. Fine-tuned Model
- Saved to the specified output directory
- Can be loaded and used with the main LLMLingua package

### 2. Training Logs
- `reinforce_logs_updated.csv`: Policy loss and learning rate logs
- `rewards_logs_updated.csv`: Reward and compression ratio logs
- Console output with real-time training progress

### 3. Log Format
```
Train/Val, Epoch, Step, Policy Loss, LR
Train, 0, 0, 0.123, 1e-05
Train, 0, 5, 0.098, 1e-05
...
```

## Evaluation

After training your TACO-RL model, you can evaluate its performance using the evaluation scripts and utilities available in the [LLMLingua2 experiments directory](../llmlingua2/evaluation/).


## Integration with Main LLMLingua

After training, you can use your fine-tuned model with the main LLMLingua package:

```python
from llmlingua import PromptCompressor

# Load your fine-tuned model
compressor = PromptCompressor(
    model_name="path/to/your/fine_tuned_model",
    use_llmlingua2=True
)

# Use it for compression
compressed_prompt = compressor.compress_prompt_llmlingua2(
    ["Your prompt here..."],
    rate=0.5
)
```


## Notes

- The training uses the REINFORCE algorithm for policy gradient optimization
- The compressor maintains the same interface as the original LLMLingua but provides additional RL information
- Make sure to adjust batch sizes and learning rates based on your hardware capabilities
- The teacher model (GPT-3.5) is used to evaluate compression quality and provide reward signals
- All hyperparameters are configurable through the YAML file for easy experimentation
- Different tasks may require different evaluation metrics for optimal performance

## Debugging Tips

1. **Check Logs**: Monitor the CSV log files for training progress
2. **Validate Data**: Ensure your training data format is correct
3. **Test Configuration**: Start with a small dataset to test your setup
4. **Monitor Rewards**: Watch for consistent reward improvements over time
5. **Check Compression**: Verify that compression ratios are within expected ranges