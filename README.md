# Kiswahili to Kikuyu Translation Model
[![GitHub stars](https://img.shields.io/github/stars/OnpointSoftwares/Kiswahili-kikuyu-Translator?style=social)](https://github.com/OnpointSoftwares/Kiswahili-kikuyu-Translator)
[![GitHub forks](https://img.shields.io/github/forks/OnpointSoftwares/Kiswahili-kikuyu-Translator?style=social)](https://github.com/OnpointSoftwares/Kiswahili-kikuyu-Translator)
[![GitHub license](https://img.shields.io/github/license/OnpointSoftwares/Kiswahili-kikuyu-Translator)](https://github.com/OnpointSoftwares/Kiswahili-kikuyu-Translator/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.7+-blue)](https://www.python.org/downloads/release/python-370/)

<p align="center">
  <img src="https://raw.githubusercontent.com/OnpointSoftwares/Kiswahili-kikuyu-Translator/main/assets/banner.png" alt="Kiswahili to Kikuyu Translation" width="100%">
</p>

This project implements a state-of-the-art neural machine translation system for translating between Kiswahili (Swahili) and Kikuyu languages, utilizing transformer-based architecture via the Hugging Face Transformers library.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/OnpointSoftwares/Kiswahili-kikuyu-Translator.git
cd Kiswahili-kikuyu-Translator

# Install dependencies
pip install -r requirements.txt
```

## 📚 Data Preparation

### Using Separate Text Files (Recommended)

1. Place your Kiswahili and Kikuyu text files in the `data/raw` directory:
   - Each file should contain one sentence per line
   - Files must have the same number of lines with corresponding translations
2. Run the preparation script:
   ```bash
   python src/main.py prepare --kiswahili_file data/raw/kiswahili.txt --kikuyu_file data/raw/kikuyu.txt
   ```

### Using Parallel Corpus (Alternative)

1. Place your parallel corpus in the `data/raw` directory
   - Format: one sentence pair per line, separated by a tab
2. Run the preparation script:
   ```bash
   python src/main.py prepare --input_file data/raw/sw_ki_parallel.txt
   ```

## 🏋️ Training

Train the translation model:

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

## 📊 Evaluation

Evaluate the trained model:

```bash
python src/evaluate.py \
    --model_dir models/sw-ki-translation \
    --test_file data/processed/test.csv
```

## 🔄 Translation

Translate Kiswahili text to Kikuyu:

```bash
python src/translate.py \
    --model_dir models/sw-ki-translation \
    --input_text "Habari ya asubuhi" \
    --output_file translations.txt
```

Or use interactive mode:

```bash
python src/translate.py --model_dir models/sw-ki-translation --interactive
```

## 🤖 Model Architecture

The system utilizes the transformer encoder-decoder architecture, fine-tuned from pre-trained Hugging Face models to adapt to the Kiswahili-Kikuyu language pair.

## 📊 Project Statistics

[![GitHub contributors](https://img.shields.io/github/contributors/OnpointSoftwares/Kiswahili-kikuyu-Translator)](https://github.com/OnpointSoftwares/Kiswahili-kikuyu-Translator/graphs/contributors)
[![GitHub last commit](https://img.shields.io/github/last-commit/OnpointSoftwares/Kiswahili-kikuyu-Translator)](https://github.com/OnpointSoftwares/Kiswahili-kikuyu-Translator/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/OnpointSoftwares/Kiswahili-kikuyu-Translator)](https://github.com/OnpointSoftwares/Kiswahili-kikuyu-Translator/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/OnpointSoftwares/Kiswahili-kikuyu-Translator)](https://github.com/OnpointSoftwares/Kiswahili-kikuyu-Translator/pulls)

## 🤔 Challenges and Considerations

- Both languages are Bantu languages with distinct grammatical structures and vocabulary
- Limited parallel data availability in low-resource context
- Specialized terminology and cultural references require special handling

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Documentation

For detailed documentation, please refer to the [docs](docs/) directory.

## 🙏 Acknowledgments

- Thanks to the Hugging Face community for their excellent transformers library
- Special thanks to contributors who have helped improve this project

## 📱 Contact

For questions or support, please open an issue or contact us at [support@onpointsoftwares.com](mailto:support@onpointsoftwares.com).
