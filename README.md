# Kiswahili to Kikuyu Translation Model

This project implements a neural machine translation system for translating Kiswahili (Swahili) to Kikuyu, using transformer-based architecture via the Hugging Face Transformers library.

## Setup

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended for training)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd languageTranslation

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### Using Separate Text Files (Recommended)

1. Place your Kiswahili text file (`kiswahili.txt`) and Kikuyu text file (`kikuyu.txt`) in the `data/raw` directory
    - Each file should contain one sentence per line
    - The files must have the same number of lines, with corresponding translations on the same line numbers
2. Run the data preparation script:
   ```bash
   python src/main.py prepare --kiswahili_file data/raw/kiswahili.txt --kikuyu_file data/raw/kikuyu.txt
   ```

### Using Parallel Corpus (Alternative)

1. Place your Kiswahili-Kikuyu parallel corpus in the `data/raw` directory
    - Format: one sentence pair per line, with source and target separated by a tab character
2. Run the data preparation script:
   ```bash
   python src/main.py prepare --input_file data/raw/sw_ki_parallel.txt
   ```

## Training

To train the translation model:

```bash
python src/train.py \
    --train_file data/processed/train.csv \
    --validation_file data/processed/validation.csv \
    --source_lang sw \
    --target_lang ki \
    --output_dir models/sw-ki-translation \
    --num_train_epochs 10 \
    --learning_rate 5e-5
```

## Evaluation

To evaluate the trained model:

```bash
python src/evaluate.py \
    --model_dir models/sw-ki-translation \
    --test_file data/processed/test.csv
```

## Translation

To translate Kiswahili text to Kikuyu:

```bash
python src/translate.py \
    --model_dir models/sw-ki-translation \
    --input_text "Habari ya asubuhi" \
    --output_file translations.txt
```

Alternatively, use the interactive mode:

```bash
python src/translate.py --model_dir models/sw-ki-translation --interactive
```

## Model Architecture

The translation system is based on the transformer encoder-decoder architecture, which has proven highly effective for machine translation tasks. We fine-tune a pre-trained model from Hugging Face to adapt it to the Kiswahili-Kikuyu language pair.

## Challenges and Considerations

- Kikuyu and Kiswahili are both Bantu languages but have distinct grammatical structures and vocabulary
- Low-resource context: parallel data may be limited compared to high-resource language pairs
- Specialized terminology and cultural references may require special handling

## License

This project is licensed under the MIT License - see the LICENSE file for details.
