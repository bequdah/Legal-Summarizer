#!/usr/bin/env python3
"""
Easy Pipeline-based Legal Document Summarization

This example uses the Hugging Face pipeline API for simpler usage.
"""

from transformers import pipeline

# Create summarization pipeline
summarizer = pipeline(
    "summarization",
    model="VincentMuriuki/legal-summarizer"
)

# Example legal texts
legal_documents = [
    {
        "name": "Privacy Policy",
        "content": """We are committed to protecting your privacy. We will never sell, trade, or 
rent your personal information to third parties. All data we collect is used 
solely to improve your experience with our services. We employ industry-standard 
security measures to protect your information from unauthorized access. Your data 
is stored securely on our servers and is only accessed by authorized personnel."""
    },
    {
        "name": "Terms of Service",
        "content": """These Terms of Service govern your use of our platform and services. By 
accessing our website or using our services, you agree to comply with these terms. 
We reserve the right to modify these terms at any time. Your continued use of our 
services constitutes acceptance of any modifications. Users must be at least 18 
years old to use our services. We are not responsible for any unauthorized access 
to your account due to negligence on your part."""
    }
]

if __name__ == "__main__":
    print("\n" + "="*80)
    print("LEGAL DOCUMENT SUMMARIZER - PIPELINE METHOD")
    print("="*80 + "\n")
    
    for doc in legal_documents:
        print(f"\n📋 Document: {doc['name']}")
        print("-" * 80)
        print("Original:")
        print(doc['content'])
        
        # Generate summary
        summary = summarizer(
            doc['content'],
            max_length=100,
            min_length=40,
            do_sample=False
        )
        
        print("\nSummary:")
        print(summary[0]['summary_text'])
        print()
