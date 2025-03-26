#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for the Kiswahili to Kikuyu Translation Model

This script provides a unified interface for data preparation, model training,
evaluation, and inference for the Kiswahili to Kikuyu translation system.
"""

import argparse
import logging
import os
import sys
import subprocess
import time

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Get the absolute path of the project directory
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def setup_argparse():
    parser = argparse.ArgumentParser(description='Kiswahili to Kikuyu Translation System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Data preparation command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare data for training')
    prepare_parser.add_argument('--input_file', type=str, default=os.path.join(PROJECT_DIR, 'data/raw/sw_ki_parallel.txt'),
                             help='Path to raw parallel corpus file (only used if --kiswahili_file and --kikuyu_file are not provided)')
    prepare_parser.add_argument('--kiswahili_file', type=str, default=os.path.join(PROJECT_DIR, 'data/raw/kiswahili.txt'),
                             help='Path to Kiswahili text file')
    prepare_parser.add_argument('--kikuyu_file', type=str, default=os.path.join(PROJECT_DIR, 'data/raw/kikuyu.txt'),
                             help='Path to Kikuyu text file')
    prepare_parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_DIR, 'data/processed'),
                             help='Directory to save processed files')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train translation model')
    train_parser.add_argument('--train_file', type=str, default=os.path.join(PROJECT_DIR, 'data/processed/train.csv'),
                           help='Path to training file')
    train_parser.add_argument('--validation_file', type=str, default=os.path.join(PROJECT_DIR, 'data/processed/validation.csv'),
                           help='Path to validation file')
    train_parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_DIR, 'models/sw-ki-translation'),
                           help='Directory to save model')
    train_parser.add_argument('--num_train_epochs', type=int, default=10,
                           help='Number of training epochs')
    train_parser.add_argument('--learning_rate', type=float, default=5e-5,
                           help='Learning rate')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate translation model')
    eval_parser.add_argument('--model_dir', type=str, default=os.path.join(PROJECT_DIR, 'models/sw-ki-translation'),
                          help='Directory containing the trained model')
    eval_parser.add_argument('--test_file', type=str, default=os.path.join(PROJECT_DIR, 'data/processed/test.csv'),
                          help='Path to test file')
    
    # Translation command
    translate_parser = subparsers.add_parser('translate', help='Translate text from Kiswahili to Kikuyu')
    translate_parser.add_argument('--model_dir', type=str, default=os.path.join(PROJECT_DIR, 'models/sw-ki-translation'),
                               help='Directory containing the trained model')
    translate_parser.add_argument('--input_text', type=str, default=None,
                               help='Kiswahili text to translate')
    translate_parser.add_argument('--interactive', action='store_true',
                               help='Run in interactive mode')
    
    # Complete pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline (prepare, train, evaluate)')
    pipeline_parser.add_argument('--num_train_epochs', type=int, default=10,
                              help='Number of training epochs')
    
    return parser.parse_args()

def run_prepare_data(args):
    """Run data preparation script"""
    logger.info("Running data preparation...")
    cmd = [
        sys.executable,
        os.path.join(PROJECT_DIR, 'scripts/prepare_data.py'),
        '--input_file', args.input_file,
        '--kiswahili_file', args.kiswahili_file,
        '--kikuyu_file', args.kikuyu_file,
        '--output_dir', args.output_dir
    ]
    # Use subprocess.call for older Python versions compatibility
    ret_code = subprocess.call(cmd)
    if ret_code != 0:
        raise subprocess.CalledProcessError(ret_code, cmd)

def run_train(args):
    """Run model training script"""
    logger.info("Running model training...")
    cmd = [
        sys.executable,
        os.path.join(PROJECT_DIR, 'src/train.py'),
        '--train_file', args.train_file,
        '--validation_file', args.validation_file,
        '--output_dir', args.output_dir,
        '--num_train_epochs', str(args.num_train_epochs),
        '--learning_rate', str(args.learning_rate)
    ]
    # Use subprocess.call for older Python versions compatibility
    ret_code = subprocess.call(cmd)
    if ret_code != 0:
        raise subprocess.CalledProcessError(ret_code, cmd)

def run_evaluate(args):
    """Run model evaluation script"""
    logger.info("Running model evaluation...")
    cmd = [
        sys.executable,
        os.path.join(PROJECT_DIR, 'src/evaluate.py'),
        '--model_dir', args.model_dir,
        '--test_file', args.test_file
    ]
    # Use subprocess.call for older Python versions compatibility
    ret_code = subprocess.call(cmd)
    if ret_code != 0:
        raise subprocess.CalledProcessError(ret_code, cmd)

def run_translate(args):
    """Run translation script"""
    logger.info("Running translation...")
    cmd = [
        sys.executable,
        os.path.join(PROJECT_DIR, 'src/translate.py'),
        '--model_dir', args.model_dir
    ]
    
    if args.input_text:
        cmd.extend(['--input_text', args.input_text])
    
    if args.interactive:
        cmd.append('--interactive')
    
    # Use subprocess.call for older Python versions compatibility
    ret_code = subprocess.call(cmd)
    if ret_code != 0:
        raise subprocess.CalledProcessError(ret_code, cmd)

def run_pipeline(args):
    """Run the complete pipeline"""
    logger.info("Running complete pipeline")
    
    # Prepare data
    prepare_args = argparse.Namespace(
        input_file=os.path.join(PROJECT_DIR, 'data/raw/sw_ki_parallel.txt'),
        kiswahili_file=os.path.join(PROJECT_DIR, 'data/raw/kiswahili.txt'),
        kikuyu_file=os.path.join(PROJECT_DIR, 'data/raw/kikuyu.txt'),
        output_dir=os.path.join(PROJECT_DIR, 'data/processed')
    )
    run_prepare_data(prepare_args)
    
    # Train model
    train_args = argparse.Namespace(
        train_file=os.path.join(PROJECT_DIR, 'data/processed/train.csv'),
        validation_file=os.path.join(PROJECT_DIR, 'data/processed/validation.csv'),
        output_dir=os.path.join(PROJECT_DIR, 'models/sw-ki-translation'),
        num_train_epochs=args.num_train_epochs,
        learning_rate=5e-5
    )
    run_train(train_args)
    
    # Evaluate model
    eval_args = argparse.Namespace(
        model_dir=os.path.join(PROJECT_DIR, 'models/sw-ki-translation'),
        test_file=os.path.join(PROJECT_DIR, 'data/processed/test.csv')
    )
    run_evaluate(eval_args)

def main():
    args = setup_argparse()
    
    # Execute the appropriate command
    if args.command == 'prepare':
        run_prepare_data(args)
    elif args.command == 'train':
        run_train(args)
    elif args.command == 'evaluate':
        run_evaluate(args)
    elif args.command == 'translate':
        run_translate(args)
    elif args.command == 'pipeline':
        run_pipeline(args)
    else:
        logger.error("No command specified. Use -h for help.")
        sys.exit(1)
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
