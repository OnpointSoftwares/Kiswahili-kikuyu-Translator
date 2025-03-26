#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Translation script for Kiswahili to Kikuyu

This script uses a trained model to translate Kiswahili text to Kikuyu.
It can operate in either interactive mode or batch processing mode.
"""

import argparse
import logging
import os
import sys
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, MarianTokenizer

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def setup_argparse():
    parser = argparse.ArgumentParser(description='Translate Kiswahili to Kikuyu')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing the trained model and tokenizer')
    parser.add_argument('--input_text', type=str, default=None,
                        help='Kiswahili text to translate')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Path to a file containing Kiswahili text (one sentence per line)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save the translations')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum length for generated translations')
    parser.add_argument('--num_beams', type=int, default=5,
                        help='Number of beams for beam search')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computation (cuda or cpu)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
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

def translate_text(text, model, tokenizer, device, max_length, num_beams):
    """Translate a single text from Kiswahili to Kikuyu"""
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    # Generate translation
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
    
    # Decode the generated tokens
    translation = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return translation

def interactive_mode(model, tokenizer, device, max_length, num_beams):
    """Run an interactive translation session"""
    print("\n=== Kiswahili to Kikuyu Interactive Translator ===\n")
    print("Type your Kiswahili text to translate, or 'exit' to quit.\n")
    
    while True:
        try:
            # Get user input
            user_input = input("\nKiswahili > ")
            
            # Check if user wants to exit
            if user_input.lower() in ['exit', 'quit', 'q', 'bye']:
                print("\nGoodbye! / Kwaheri! / Thiü!\n")
                break
            
            # Skip empty input
            if not user_input.strip():
                continue
            
            # Translate the input
            translation = translate_text(user_input, model, tokenizer, device, max_length, num_beams)
            
            # Display the translation
            print(f"Kikuyu   > {translation}")
            
        except KeyboardInterrupt:
            print("\nGoodbye! / Kwaheri! / Thiü!\n")
            break
        except Exception as e:
            print(f"Error: {e}")

def process_file(input_file, output_file, model, tokenizer, device, max_length, num_beams):
    """Process a file of Kiswahili sentences and translate them to Kikuyu"""
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(lines)} sentences from {input_file}")
    
    # Translate each line
    translations = []
    for i, line in enumerate(lines):
        if i % 10 == 0:
            logger.info(f"Translating sentence {i+1}/{len(lines)}")
        translation = translate_text(line, model, tokenizer, device, max_length, num_beams)
        translations.append(translation)
    
    # Write translations to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (source, target) in enumerate(zip(lines, translations)):
            f.write(f"Kiswahili: {source}\nKikuyu: {target}\n\n")
    
    logger.info(f"Saved {len(translations)} translations to {output_file}")

def main():
    args = setup_argparse()
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        logger.error(f"Model directory {args.model_dir} does not exist!")
        sys.exit(1)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.device)
    
    if args.interactive:
        # Run in interactive mode
        interactive_mode(model, tokenizer, args.device, args.max_length, args.num_beams)
    elif args.input_text:
        # Translate a single text
        translation = translate_text(args.input_text, model, tokenizer, args.device, args.max_length, args.num_beams)
        print(f"Kiswahili: {args.input_text}")
        print(f"Kikuyu:    {translation}")
        
        # Save to output file if specified
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(f"Kiswahili: {args.input_text}\nKikuyu: {translation}\n")
            logger.info(f"Saved translation to {args.output_file}")
    elif args.input_file:
        # Check if output file is specified
        if not args.output_file:
            logger.error("Output file must be specified when using input_file")
            sys.exit(1)
        
        # Process the input file
        process_file(args.input_file, args.output_file, model, tokenizer, args.device, args.max_length, args.num_beams)
    else:
        logger.error("Either --input_text, --input_file, or --interactive must be specified")
        sys.exit(1)

if __name__ == "__main__":
    main()
