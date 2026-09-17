# TACO-RL: Task-Aware Prompt Compression Optimization with Reinforcement Learning

This submodule provides reinforcement learning capabilities for fine-tuning LLMLingua models on new tasks using reward signals from language models like GPT-3.5.

## Overview

TACO-RL (Task-Aware Prompt Compression Optimization with Reinforcement Learning) extends the LLMLingua framework by adding reinforcement learning capabilities that allow you to fine-tune pre-trained LLMLingua models on new tasks. The approach addresses the limitations of existing prompt compression techniques by leveraging task-specific reward signals to guide the learning process.

### Key Innovation

Unlike traditional prompt compression methods that either rely on sub-optimal metrics like information entropy or treat compression as a task-agnostic token classification problem, TACO-RL:

1. **Leverages Bidirectional Context**: Uses existing Transformer encoder-based token classification models for low latency
2. **Task-Specific Optimization**: Guides learning with task-specific reward signals using lightweight REINFORCE algorithm
3. **Performance Improvement**: Achieves 8%-189% improvement across diverse tasks while maintaining compression rates and latency requirements

### Research Foundation

Based on the paper "TACO-RL: Task Aware Prompt Compression Optimization with Reinforcement Learning" ([arXiv:2409.13035](https://arxiv.org/pdf/2409.13035)), this implementation addresses two key research questions:

- **Q1**: How to design a prompt compression model that effectively leverages bidirectional context while providing low inference latency?
- **Q2**: How to efficiently train a model with proper guidance from task-specific reward signals while minimizing computational cost?

The solution builds on LLMLingua-2's task-agnostic encoder-based transformer model and enhances it with task-specific reward signals using on-policy REINFORCE algorithm.

## How TACO-RL Works

### Architecture Overview

1. **Pre-trained Base Model**: Start with a pre-trained LLMLingua model (e.g., `microsoft/llmlingua-2-xlm-roberta-large-meetingbank`)
2. **Task-Specific Data**: Prepare task-specific data with original prompts and expected outputs
3. **Reward Signal Generation**: Use a teacher model (e.g., GPT-3.5) to evaluate compression quality by comparing outputs from original vs. compressed prompts
4. **REINFORCE Training**: Fine-tune the model using policy gradients based on task-specific reward signals
5. **Task-Optimized Model**: Get a model optimized for your specific task while maintaining low inference latency

### Training Process

During the model alignment process:
1. Generate task output from both original and compressed prompts
2. Compute task-specific reward signal using divergence between outputs
3. Update the base encoder model using on-policy REINFORCE algorithm
4. The reward signals guide the model to preserve task-relevant information during compression

## Task-Specific Metrics and Evaluation

TACO-RL has been extensively evaluated on three diverse and challenging tasks with different evaluation metrics:

### 1. Text Summarization (MeetingBank Dataset)

**Metrics Considered:**
- **BLEU**: Measures n-gram overlap between generated and reference summaries
- **ROUGE-1**: Unigram overlap for content coverage
- **ROUGE-2**: Bigram overlap for fluency assessment
- **ROUGE-L**: Longest common subsequence for structure preservation
- **BERTScore F1**: Semantic similarity using BERT embeddings

### 2. Question Answering (SQuAD Dataset)

**Metrics Considered:**
- **Exact Match (EM)**: Percentage of answers that exactly match the ground truth
- **F1 Score**: Harmonic mean of precision and recall for answer span prediction
- **BLEU**: For answer generation quality
- **ROUGE**: For answer completeness and relevance

### 3. Code Summarization (CodeSearchNet Dataset)

**Metrics Considered:**
- **BLEU**: For code summary quality
- **ROUGE-1/2/L**: For summary completeness and relevance
- **BERTScore F1**: For semantic similarity of code descriptions
- **CodeBLEU**: Specialized metric for code-to-text generation

## Key Features

The `PromptCompressorReinforce` class extends the base `PromptCompressor` with:

1. **Action Tracking**: Tracks actions taken during compression for RL training
2. **Log Probability Storage**: Stores log probabilities for policy gradient computation
3. **Entropy Calculation**: Computes entropy for exploration regularization
4. **RL Information Methods**: Provides methods to access and clear RL-specific data
5. **Configurable Training**: All hyperparameters configurable via YAML files
6. **Multi-Metric Support**: Configurable evaluation metrics for different tasks


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
