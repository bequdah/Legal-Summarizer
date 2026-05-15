#!/usr/bin/env python3
"""
Basic Legal Document Summarization Example

This script demonstrates how to use the legal-summarizer model
to summarize legal documents.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load pre-trained model and tokenizer
model_name = "VincentMuriuki/legal-summarizer"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Example legal texts
examples = [
    {
        "title": "Privacy Policy",
        "text": """We are committed to protecting your privacy. We will never sell, trade, or 
rent your personal information to third parties. All data we collect is used 
solely to improve your experience with our services. We employ industry-standard 
security measures to protect your information from unauthorized access. Your data 
is stored securely on our servers and is only accessed by authorized personnel. 
We comply with all applicable data protection regulations including GDPR and CCPA."""
    },
    {
        "title": "Service Agreement",
        "text": """This Service Agreement ('Agreement') is entered into as of the date of acceptance 
by the Client ('You') with the Company ('We'). By accessing or using our services, 
you agree to be bound by the terms and conditions outlined herein. The Company 
provides software-as-a-service solutions designed to enhance business operations. 
The Client is responsible for maintaining the confidentiality of login credentials 
and for all activities that occur under the account. Any unauthorized use should 
be reported immediately to our support team."""
    }
]

def summarize_text(text, max_length=150, min_length=50):
    """
    Summarize legal text using the pre-trained model.
    
    Args:
        text (str): Legal document text to summarize
        max_length (int): Maximum length of summary
        min_length (int): Minimum length of summary
    
    Returns:
        str: Generated summary
    """
    # Tokenize input
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate summary
    summary_ids = model.generate(
        inputs,
        num_beams=4,
        min_length=min_length,
        max_length=max_length,
        early_stopping=True
    )
    
    # Decode and return summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    print("=" * 80)
    print("LEGAL DOCUMENT SUMMARIZER")
    print("=" * 80)
    
    for example in examples:
        print(f"\n{'='*80}")
        print(f"Example: {example['title']}")
        print(f"{'='*80}")
        
        print("\n📄 Original Text:")
        print("-" * 80)
        print(example['text'])
        
        print("\n✂️  Generated Summary:")
        print("-" * 80)
        summary = summarize_text(example['text'])
        print(summary)
        print()
