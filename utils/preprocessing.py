import argparse
import json
import re
from pathlib import Path

import nltk
import pymorphy3
from nltk.corpus import stopwords
from tqdm import tqdm

nltk.download('stopwords')
russian_stopwords = set(stopwords.words('russian'))
morph = pymorphy3.MorphAnalyzer()
russian_stopwords.add('который')
russian_stopwords.add('такой')


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    processed_words = [lemma for lemma in lemmas if lemma not in russian_stopwords]
    processed_text = ' '.join(processed_words)

    return processed_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Path to the input JSONL file.')
    parser.add_argument('output_file', help='Path to the output JSONL file.')
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input_file
    output_file = args.output_file

    with open(input_file, 'r', encoding='utf-8') as infile:
        total_lines = sum(1 for _ in infile)

    corpus = {}
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in tqdm(file, total=total_lines, desc='Preprocessing text'):
            record = json.loads(line)
            record['processed_text'] = preprocess_text(record['text'])
            corpus[record['_id']] = record

    Path(output_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for cid in corpus.keys():
            outfile.write(json.dumps(corpus[cid], ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
