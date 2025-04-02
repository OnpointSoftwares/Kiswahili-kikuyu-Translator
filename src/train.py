#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training script for Kiswahili to Kikuyu neural machine translation model

This script trains a transformer-based sequence-to-sequence model for translation
using the Hugging Face Transformers library.
"""

import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    MarianConfig,
    MarianMTModel,
    MarianTokenizer,
    EarlyStoppingCallback
)

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import evaluate

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def setup_argparse():
    parser = argparse.ArgumentParser(description='Train a Kiswahili to Kikuyu translation model')
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to the training text file')
    parser.add_argument('--validation_file', type=str, required=True,
                        help='Path to the validation text file')
    parser.add_argument('--source_lang', type=str, default='sw',
                        help='Source language code (default: sw for Kiswahili)')
    parser.add_argument('--target_lang', type=str, default='ki',
                        help='Target language code (default: ki for Kikuyu)')
    parser.add_argument('--pretrained_model', type=str, default='Helsinki-NLP/opus-mt-en-mul',
                        help='Pretrained model to start from (default: Helsinki-NLP/opus-mt-en-mul)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the model and tokenizer')
    parser.add_argument('--max_source_length', type=int, default=128,
                        help='Maximum source sequence length')
    parser.add_argument('--max_target_length', type=int, default=128,
                        help='Maximum target sequence length')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help='Evaluation batch size')
    parser.add_argument('--num_train_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Log every X steps')
    parser.add_argument('--save_total_limit', type=int, default=5,
                        help='Maximum number of checkpoints to keep')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Number of evaluation steps with no improvement to wait before early stopping')
    return parser.parse_args()

def load_datasets(train_file, validation_file, source_lang, target_lang):
    """Load parallel text datasets"""
    # Read lines from text files
    with open(train_file, 'r', encoding='utf-8') as f:
        source_lines = f.readlines()
    with open(validation_file, 'r', encoding='utf-8') as f:
        target_lines = f.readlines()

    # Ensure both files have the same number of lines
    assert len(source_lines) == len(target_lines), "Source and target files must have the same number of lines"

    # Create a dataset from the lines
    data = {source_lang: source_lines, target_lang: target_lines}
    dataset = Dataset.from_dict(data)

    # Split into train and validation datasets
    dataset_dict = DatasetDict({
        'train': dataset.select(range(int(len(dataset) * 0.8))),
        'validation': dataset.select(range(int(len(dataset) * 0.8), len(dataset)))
    })

    logger.info("Loaded training set: {} examples".format(len(dataset_dict['train'])))
    logger.info("Loaded validation set: {} examples".format(len(dataset_dict['validation'])))

    return dataset_dict

def preprocess_function(examples, tokenizer, source_lang, target_lang, max_source_length, max_target_length):
    """Tokenize and prepare inputs and targets for model training"""
    inputs = examples[source_lang]
    targets = examples[target_lang]
    
    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding='max_length', truncation=True)
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding='max_length', truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    
    # Replace padding token id's in the labels with -100 so they're ignored in the loss
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
    ]
    
    return model_inputs

def compute_metrics(eval_preds, tokenizer, metric):
    """Compute BLEU score for evaluation"""
    preds, labels = eval_preds
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and references
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # BLEU expects a list of references for each prediction
    result = metric.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels])
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v * 100, 4) for k, v in result.items()}

def main():
    args = setup_argparse()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    datasets = load_datasets(args.train_file, args.validation_file, args.source_lang, args.target_lang)
    
    # Initialize tokenizer and model
    try:
        # First attempt to load a MarianMT model
        #logger.info(f"Attempting to load pretrained model: {args.pretrained_model}")
        tokenizer = MarianTokenizer.from_pretrained(args.pretrained_model)
        model = MarianMTModel.from_pretrained(args.pretrained_model)
        logger.info("Successfully loaded MarianMT model and tokenizer")
    except Exception as e:
        #logger.warning(f"Could not load MarianMT model. Error: {e}")
        logger.info("Falling back to generic sequence-to-sequence model")
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model)
        logger.info("Successfully loaded generic sequence-to-sequence model and tokenizer")
    
    # For MarianMT models, set the source and target languages
    if hasattr(tokenizer, 'source_lang') and hasattr(tokenizer, 'target_lang'):
        tokenizer.source_lang = args.source_lang
        tokenizer.target_lang = args.target_lang
        
        # Update the model configuration with the new language pair
        if hasattr(model.config, 'src_vocab_size'):
            model.config.src_vocab_size = len(tokenizer)
        if hasattr(model.config, 'tgt_vocab_size'):
            model.config.tgt_vocab_size = len(tokenizer)
    
    # Define preprocessing function with fixed parameters
    def preprocess_data(examples):
        return preprocess_function(
            examples, tokenizer, args.source_lang, args.target_lang, 
            args.max_source_length, args.max_target_length
        )
    
    # Preprocess datasets
    tokenized_datasets = datasets.map(
        preprocess_data,
        batched=True,
        remove_columns=datasets["train"].column_names,
        desc="Tokenizing datasets",
    )
    
    # Load BLEU metric for evaluation
    metric = evaluate.load("sacrebleu")
    
    # Create a data collator for sequence-to-sequence task
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # Define the compute_metrics function with fixed tokenizer and metric
    def compute_metrics_fixed(eval_preds):
        return compute_metrics(eval_preds, tokenizer, metric)
    
    # Define training arguments with memory optimizations
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        eval_steps=500,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,

        # Mixed Precision Training
        bf16=torch.cuda.is_bf16_supported(),  # Use BF16 if available
        fp16=not torch.cuda.is_bf16_supported(),  # Fallback to FP16 if BF16 is unavailable

        # Model Optimization
        optim="adamw_torch",  # AdamW is more stable than Adafactor
        gradient_checkpointing=True,  # Save memory by checkpointing activations
        max_grad_norm=1.0,  # Clip gradients to prevent memory spikes
        warmup_ratio=0.1,  # Use warmup ratio instead of steps for better memory management
        group_by_length=True,  # Reduce padding in batches to save memory
        dataloader_num_workers=4,  # Multi-threaded data loading
        dataloader_pin_memory=True,  # Speed up GPU memory transfer
        dataloader_drop_last=True,  # Drop last batch for consistency

        # Evaluation and Saving
        eval_accumulation_steps=8,  # Accumulate evaluation steps to reduce memory usage
        save_strategy="epoch" if args.batch_size > 8 else "steps",
        save_steps=500 if args.batch_size > 8 else 100,
        max_steps=-1,  # Train for the full number of epochs
        disable_tqdm=True,  # Disable progress bar to save memory

        # Early Stopping Fix
        load_best_model_at_end=True,  # Ensure best model is restored
        metric_for_best_model="eval_loss",  # Define a metric to monitor
        greater_is_better=False,  # Lower loss is better
        save_total_limit=2,  # Keep only the 2 best checkpoints

        # Multi-GPU and Distributed Training
        fsdp="full_shard auto_wrap" if torch.cuda.device_count() > 1 else "",  # Ensure fsdp is always a string
    )
    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fixed,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )
    
    # Train the model
    #logger.info("Starting training...")
    trainer.train()
    
    # Save the final model and tokenizer
    #logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    #logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
