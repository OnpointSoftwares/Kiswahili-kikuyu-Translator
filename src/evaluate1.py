#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation script for Kiswahili to Kikuyu translation model

This script evaluates a trained model on a test dataset using BLEU, ROUGE, and other metrics.
"""

import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, MarianTokenizer
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
    parser = argparse.ArgumentParser(description='Evaluate a Kiswahili to Kikuyu translation model')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing the trained model and tokenizer')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to test file (CSV format with sw and ki columns)')
    parser.add_argument('--source_lang', type=str, default='sw',
                        help='Source language code (default: sw for Kiswahili)')
    parser.add_argument('--target_lang', type=str, default='ki',
                        help='Target language code (default: ki for Kikuyu)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum length for generated translations')
    parser.add_argument('--num_beams', type=int, default=5,
                        help='Number of beams for beam search')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computation (cuda or cpu)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save the translations (optional)')
    return parser.parse_args()

def load_model_and_tokenizer(model_dir, device):
    """Load model and tokenizer from the specified directory"""
    try:
        # First try loading as a MarianMT model
        logger.info("Attempting to load MarianMT model and tokenizer")
        tokenizer = MarianTokenizer.from_pretrained(model_dir)
        model = MarianMTModel.from_pretrained(model_dir).to(device)
        logger.info("Successfully loaded MarianMT model and tokenizer")
    except Exception as e:
        logger.warning(f"Could not load as MarianMT model: {e}")
        logger.info("Falling back to generic seq2seq model")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
        logger.info("Successfully loaded model and tokenizer")
    
    return model, tokenizer

def translate_batch(batch_texts, model, tokenizer, device, max_length, num_beams):
    """Translate a batch of texts"""
    # Tokenize input texts
    batch_encoding = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = batch_encoding.input_ids.to(device)
    attention_mask = batch_encoding.attention_mask.to(device)
    
    # Generate translations
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
    
    # Decode the generated tokens
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return translations

def evaluate_translations(references, predictions):
    """Evaluate translations using multiple metrics"""
    # Prepare references for BLEU (BLEU expects a list of references for each prediction)
    bleu_references = [[ref] for ref in references]
    
    # Load metrics
    bleu_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    
    # Compute metrics
    bleu_score = bleu_metric.compute(predictions=predictions, references=bleu_references)
    rouge_scores = rouge_metric.compute(predictions=predictions, references=references)
    meteor_score = meteor_metric.compute(predictions=predictions, references=references)
    
    # Format results
    results = {
        "bleu": round(bleu_score["score"], 2),
        "rouge1": round(rouge_scores["rouge1"] * 100, 2),
        "rouge2": round(rouge_scores["rouge2"] * 100, 2),
        "rougeL": round(rouge_scores["rougeL"] * 100, 2),
        "meteor": round(meteor_score["meteor"] * 100, 2),
    }
    
    return results

def main():
    args = setup_argparse()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.device)
    
    # Load test data
    test_df = pd.read_csv(args.test_file)
    source_texts = test_df[args.source_lang].tolist()
    reference_texts = test_df[args.target_lang].tolist()
    
    logger.info(f"Loaded {len(source_texts)} test examples")
    
    # Translate in batches
    predictions = []
    for i in tqdm(range(0, len(source_texts), args.batch_size), desc="Translating"):
        batch_texts = source_texts[i:i + args.batch_size]
        batch_translations = translate_batch(
            batch_texts, model, tokenizer, args.device, args.max_length, args.num_beams
        )
        predictions.extend(batch_translations)
    
    # Evaluate translations
    logger.info("Evaluating translations...")
    results = evaluate_translations(reference_texts, predictions)
    
    # Display evaluation results
    logger.info("Evaluation Results:")
    for metric, score in results.items():
        logger.info(f"{metric.upper()}: {score}")
    
    # Save translations if output file is specified
    if args.output_file:
        output_df = pd.DataFrame({
            args.source_lang: source_texts,
            f"{args.target_lang}_reference": reference_texts,
            f"{args.target_lang}_prediction": predictions
        })
        output_df.to_csv(args.output_file, index=False)
        logger.info(f"Saved translations to {args.output_file}")
    
    # Print some example translations
    logger.info("\nExample Translations:")
    num_examples = min(5, len(source_texts))
    for i in range(num_examples):
        logger.info(f"Source ({args.source_lang}): {source_texts[i]}")
        logger.info(f"Reference ({args.target_lang}): {reference_texts[i]}")
        logger.info(f"Prediction ({args.target_lang}): {predictions[i]}")
        logger.info("---")

if __name__ == "__main__":
    main()
