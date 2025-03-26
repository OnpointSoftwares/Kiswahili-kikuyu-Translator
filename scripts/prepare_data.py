#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data preparation script for Kiswahili to Kikuyu translation

This script processes raw parallel data and prepares it for training:
1. Loads raw parallel data
2. Cleans and normalizes text
3. Splits into train/validation/test sets
4. Saves processed data in formats compatible with the Hugging Face datasets library
"""

import os
import pandas as pd
import argparse
import random
from pathlib import Path
import re
from tqdm import tqdm

def setup_argparse():
    parser = argparse.ArgumentParser(description='Prepare data for Kiswahili-Kikuyu translation')
    parser.add_argument('--input_file', type=str, default='data/raw/sw_ki_parallel.txt',
                        help='Path to raw parallel corpus file (only used if --kiswahili_file and --kikuyu_file are not provided)')
    parser.add_argument('--kiswahili_file', type=str, default='data/raw/kiswahili.txt',
                        help='Path to Kiswahili text file')
    parser.add_argument('--kikuyu_file', type=str, default='data/raw/kikuyu.txt',
                        help='Path to Kikuyu text file')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Directory to save processed files')
    parser.add_argument('--train_size', type=float, default=0.8,
                        help='Proportion of data to use for training')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Proportion of data to use for validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def clean_text(text):
    """Basic text cleaning and normalization"""
    # Convert to lowercase
    text = text.lower()
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def load_and_process_parallel_data(input_file):
    """
    Load and process the parallel corpus.
    Expected format: one sentence pair per line, source and target separated by tab
    """
    source_texts = []
    target_texts = []
    
    print("Loading data from {}".format(input_file))
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            parts = line.strip().split('\t')
            if len(parts) == 2:
                sw_text, ki_text = parts
                source_texts.append(clean_text(sw_text))
                target_texts.append(clean_text(ki_text))
    
    print("Loaded {} parallel sentences".format(len(source_texts)))
    
    return pd.DataFrame({
        'sw': source_texts,
        'ki': target_texts
    })

def load_and_process_separate_files(kiswahili_file, kikuyu_file):
    """
    Load and process separate Kiswahili and Kikuyu files.
    Expected format: one sentence per line in each file, with lines aligned across files.
    """
    source_texts = []
    target_texts = []
    
    print("Loading Kiswahili data from {}".format(kiswahili_file))
    print("Loading Kikuyu data from {}".format(kikuyu_file))
    
    # Load Kiswahili text
    with open(kiswahili_file, 'r', encoding='utf-8') as f:
        sw_lines = [clean_text(line.strip()) for line in f if line.strip()]
    
    # Load Kikuyu text
    with open(kikuyu_file, 'r', encoding='utf-8') as f:
        ki_lines = [clean_text(line.strip()) for line in f if line.strip()]
    
    # Check if the files have the same number of lines
    if len(sw_lines) != len(ki_lines):
        print("Warning: Files have different number of lines. Kiswahili: {}, Kikuyu: {}".format(len(sw_lines), len(ki_lines)))
        # Use the smaller number of lines
        min_lines = min(len(sw_lines), len(ki_lines))
        sw_lines = sw_lines[:min_lines]
        ki_lines = ki_lines[:min_lines]
    
    print("Loaded {} parallel sentences".format(len(sw_lines)))
    
    return pd.DataFrame({
        'sw': sw_lines,
        'ki': ki_lines
    })

def split_data(df, train_size, val_size, seed):
    """Split data into train, validation, and test sets"""
    random.seed(seed)
    
    # Create a list of indices and shuffle it
    indices = list(range(len(df)))
    random.shuffle(indices)
    
    # Calculate split points
    train_end = int(train_size * len(indices))
    val_end = train_end + int(val_size * len(indices))
    
    # Split indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create DataFrames
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    
    return train_df, val_df, test_df

def main():
    args = setup_argparse()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if we're using separate files or a parallel corpus file
    kiswahili_path = Path(args.kiswahili_file)
    kikuyu_path = Path(args.kikuyu_file)
    
    if kiswahili_path.exists() and kikuyu_path.exists():
        print("Using separate Kiswahili and Kikuyu files")
        df = load_and_process_separate_files(args.kiswahili_file, args.kikuyu_file)
    else:
        print("Separate files not found, checking for parallel corpus file")
        # Check if input file exists
        input_path = Path(args.input_file)
        if not input_path.exists():
        # Create a sample parallel corpus for demonstration
        print("Input file {} not found. Creating a sample corpus...".format(args.input_file))
        os.makedirs(input_path.parent, exist_ok=True)
        
        # Sample Kiswahili-Kikuyu pairs (these are just examples and may not be accurate translations)
        sample_pairs = [
            "Habari yako\tWĩkĩrie atĩa",
            "Jina langu ni John\tRĩĩtwa rĩakwa nĩ John",
            "Ninafurahi kukutana nawe\tNĩngũkenerete gũgũcemania",
            "Asante sana\tNĩ wega mũno",
            "Karibu\tWamũkirwo",
            "Tafadhali\tNdagũthaitha",
            "Samahani\tNdagũthima",
            "Leo ni siku nzuri\tŨmũthĩ nĩ mũthenya mwega",
            "Nitakuona kesho\tNĩngũkuona rũciũ",
            "Ninakupenda\tNĩngwendete",
            "Chakula ni tayari\tIrio nĩ ihaarĩrie",
            "Unatoka wapi?\tUmuma kũ?",
            "Ninaenda sokoni\tNgũthiĩ ndũnyũ",
            "Ninahitaji msaada\tNĩbataire ũteithio",
            "Nitarudi baadaye\tNĩgũcooka thutha",
            "Unasema Kiswahili?\tNo mwarie Kiswahili?",
            "Ninasema Kikuyu kidogo\tNo njarie Gĩkũyũ kanini",
            "Maji ni baridi\tMaaĩ nĩ mahoro",
            "Jua ni kali leo\tRiũa nĩ igũrũ mũno ũmũthĩ",
            "Ninasoma kitabu\tNĩndĩrathoma ibuku",
            "Nyumba yangu ni karibu\tNyũmba yakwa nĩ hakuhĩ",
            "Ninataka kunywa kahawa\tNĩngwenda kũnyua kahawa",
            "Ninapenda muziki\tNĩnyendete nyĩmbo",
            "Ninahitaji kulala\tNĩbataire gũkoma",
            "Usiku mwema\tŨtukũ mwega",
            "Mtoto wangu\tKaana gakwa",
            "Rafiki yangu\tMũrata wakwa",
            "Nimechoka sana\tNĩnogete mũno",
            "Niko njaa\tNĩ mũhũtu",
            "Shule iko wapi?\tShule irĩ kũ?",
            "Mti huu ni mrefu\tMũtĩ ũyũ nĩ mũraihu",
            "Nipe maji tafadhali\tHe maaĩ ndagũthaithaĩ",
            "Nitakuja kesho\tNĩgũũka rũciũ",
            "Mvua inanyesha\tMbura nĩĩgũkiura",
            "Ninajua kuendesha gari\tNĩũĩ gũtwaara ngari",
            "Shamba langu ni kubwa\tMũgũnda wakwa nĩ mũnene",
            "Ninapika chakula\tNĩndĩraruga irio",
            "Ninapenda kusoma\tNĩnyendete gũthoma",
            "Nilikuletea zawadi\tNĩndĩrakũrehera kĩheo",
            "Wewe ni mwalimu?\tWee nĩwe mũrutani?",
            "Mimi ni mkulima\tNiĩ ndĩ mũrĩmi",
            "Kuku huyu ni mzuri\tNgũkũ ĩyo nĩ njega",
            "Mbuzi yangu iko wapi?\tMbũri yakwa ĩrĩ kũ?",
            "Milima ni mirefu\tIrĩma nĩ ndaaya",
            "Sungura ni mnyama mdogo\tGĩtũngũyũ nĩ nyamũ nini",
            "Simba ni mnyama mkubwa\tMũrũthi nĩ nyamũ nene",
            "Mto huu ni mpana\tRũũĩ rũrũ nĩ rwariĩ",
            "Wanafunzi wanasoma\tArutwo nĩmerathoma",
            "Wakulima wanalima\tArĩmi nĩmerĩma",
            "Tunda hili ni tamu\tMatunda maya nĩ mathũngũ",
            "Ninaamini katika Mungu\tNĩnjĩtĩkĩtie Ngai",
        ]
        
        with open(input_path, 'w', encoding='utf-8') as f:
            for pair in sample_pairs:
                f.write("{0}\n".format(pair))
        
        print("Created sample corpus with {} sentence pairs".format(len(sample_pairs)))
        
        # Create sample separate files as well for demonstration
        os.makedirs(kiswahili_path.parent, exist_ok=True)
        os.makedirs(kikuyu_path.parent, exist_ok=True)
        
        with open(kiswahili_path, 'w', encoding='utf-8') as f:
            for pair in sample_pairs:
                sw_text = pair.split('\t')[0]
                f.write("{0}\n".format(sw_text))
        
        with open(kikuyu_path, 'w', encoding='utf-8') as f:
            for pair in sample_pairs:
                ki_text = pair.split('\t')[1]
                f.write("{0}\n".format(ki_text))
        
        print("Created sample Kiswahili file with {} sentences".format(len(sample_pairs)))
        print("Created sample Kikuyu file with {} sentences".format(len(sample_pairs)))
        
        # Load and process data from parallel corpus file
        df = load_and_process_parallel_data(args.input_file)
    else:
        # If input_path exists, load the parallel corpus file
        df = load_and_process_parallel_data(args.input_file)
    
    # Split data
    train_df, val_df, test_df = split_data(df, args.train_size, args.val_size, args.seed)
    
    print("Train set: {} examples".format(len(train_df)))
    print("Validation set: {} examples".format(len(val_df)))
    print("Test set: {} examples".format(len(test_df)))
    
    # Save processed data
    train_df.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(args.output_dir, 'validation.csv'), index=False)
    test_df.to_csv(os.path.join(args.output_dir, 'test.csv'), index=False)
    
    print("Data successfully processed and saved to {}".format(args.output_dir))

if __name__ == "__main__":
    main()
