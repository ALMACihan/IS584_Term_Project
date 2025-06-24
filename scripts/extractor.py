import os
import json
import pandas as pd
import nltk

# Ensure punkt is downloaded
nltk.download('punkt')

from nltk.tokenize.punkt import PunktSentenceTokenizer

def extract_sentences_from_review_json(json_path, paper_id):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sentences = []
    reviews = data.get("reviews", [])
    tokenizer = PunktSentenceTokenizer()

    for review_entry in reviews:
        review_text = review_entry.get("review", "")
        if not review_text:
            continue

        for sent in tokenizer.tokenize(review_text):
            sentences.append({"sentence": sent.strip(), "paper_id": paper_id})

    return sentences

def collect_all_sentences(root_folder):
    all_sentences = []

    for root, dirs, files in os.walk(root_folder):
        for file_name in files:
            if file_name.endswith("_review.json"):
                paper_id = file_name.replace("_review.json", "")
                file_path = os.path.join(root, file_name)
                print(f"📝 Found review file: {file_path}")
                try:
                    sentences = extract_sentences_from_review_json(file_path, paper_id)
                    all_sentences.extend(sentences)
                    print(f"✅ Extracted {len(sentences)} sentences from {file_name}")
                except Exception as e:
                    print(f"❌ Failed to process {file_name}: {e}")

    return pd.DataFrame(all_sentences)

if __name__ == "__main__":
    input_folder = "C:/Users/cihan/Desktop/Spring 2025/[IS 584 Section 1] Deep Learning for Text Analytics/dataset"  # Change this if needed
    output_path = "clean_review_sentences.csv"

    df = collect_all_sentences(input_folder)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved {len(df)} sentences to {output_path}")
