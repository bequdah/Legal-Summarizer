# Legal-Summarizer

**AI-powered tool for summarizing legal documents with simplified language**

## 📋 Overview

Legal-Summarizer is an intelligent tool designed to automatically summarize complex legal documents (such as privacy policies, contracts, constitutions, and finance bills) into concise, easy-to-understand summaries.

### Key Features:
- ✅ **Simplifies Complex Legal Language** - Makes legal texts accessible to non-experts
- ✅ **Fast & Accurate** - Fine-tuned T5 model optimized for legal domain
- ✅ **Easy Integration** - Works seamlessly with Hugging Face Transformers
- ✅ **Preserves Key Information** - Retains essential points while reducing length

## 🎯 Use Cases

- 📜 Privacy Policy Summarization
- 📋 Contract Analysis
- ⚖️ Legal Document Review
- 📖 Constitution & Bill Summarization
- 🔍 Legal Research & Compliance

## 📦 Installation

### Requirements:
- Python 3.7+
- PyTorch
- Hugging Face Transformers

### Setup:

```bash
# Clone the repository
git clone https://github.com/bequdah/Legal-Summarizer.git
cd Legal-Summarizer

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load pre-trained model and tokenizer
model_name = "VincentMuriuki/legal-summarizer"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Example legal text
legal_text = """
We are committed to protecting your privacy. We will never sell, trade, or 
rent your personal information to third parties. All data we collect is used 
solely to improve your experience with our services. We employ industry-standard 
security measures to protect your information from unauthorized access.
"""

# Tokenize input
inputs = tokenizer.encode(legal_text, return_tensors="pt", max_length=512, truncation=True)

# Generate summary
summary_ids = model.generate(
    inputs,
    num_beams=4,
    min_length=30,
    max_length=150,
    early_stopping=True
)

# Decode and print summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Original Text:")
print(legal_text)
print("\nGenerated Summary:")
print(summary)
```

### Using Pipeline (Easier):

```python
from transformers import pipeline

# Create summarization pipeline
summarizer = pipeline("summarization", model="VincentMuriuki/legal-summarizer")

legal_text = "Your legal document here..."

# Generate summary
summary = summarizer(legal_text, max_length=150, min_length=50, do_sample=False)
print(summary[0]['summary_text'])
```

## 📊 Model Details

- **Base Model:** T5-base
- **Fine-tuned on:** Legal documents corpus
- **Task:** Abstractive Text Summarization
- **Max Input Length:** 512 tokens
- **Recommended Output Length:** 50-150 tokens
- **License:** Apache 2.0

## 💾 Model Source

This project uses the pre-trained model from:
**[VincentMuriuki/legal-summarizer on Hugging Face](https://huggingface.co/VincentMuriuki/legal-summarizer)**

## 📚 Examples

### Example 1: Privacy Policy
```
INPUT: "We are committed to protecting your privacy. We will never sell..."
OUTPUT: "Company protects user privacy and does not share data with third parties."
```

### Example 2: Contract Clause
```
INPUT: "In consideration of the mutual covenants and agreements contained herein..."
OUTPUT: "Both parties agree to the terms and conditions outlined in this agreement."
```

## 🔧 Advanced Usage

### Fine-tune on Custom Data:

```python
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

# Configure training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=500,
    save_total_limit=2,
)

# Create trainer and train
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

## 📖 Documentation

For more information, visit:
- [Hugging Face Model Card](https://huggingface.co/VincentMuriuki/legal-summarizer)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [T5 Paper](https://arxiv.org/abs/1910.10683)

## 🤝 Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a new branch
3. Make your improvements
4. Submit a pull request

## ⚖️ License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.

## 🙏 Acknowledgments

- Original model by [VincentMuriuki](https://huggingface.co/VincentMuriuki)
- Built with [Hugging Face Transformers](https://huggingface.co/transformers/)
- T5 model from [Google Research](https://github.com/google-research/text-to-text-transfer-transformer)

---

**Made with ❤️ for legal professionals and researchers**
