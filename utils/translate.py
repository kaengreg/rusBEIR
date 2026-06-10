import json
from deep_translator import GoogleTranslator
import re
from pathlib import Path
from tqdm import tqdm


def split_text(text, max_chars=5000):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []

    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence) + 1 > max_chars:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence) + 1

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def translate_text(text, src='en', dest='ru', max_chars=5000):
    translator = GoogleTranslator(source=src, target=dest)

    chunks = split_text(text, max_chars)
    translated_chunks = [translator.translate(chunk) for chunk in chunks]
    return ' '.join(translated_chunks)


def translate_jsonl(input_file, output_file):

    with open(input_file, 'r', encoding='utf-8') as infile:
        total_lines = sum(1 for _ in infile)

    Path(output_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
    with (open(input_file, 'r', encoding='utf-8') as infile,
          open(output_file, 'w', encoding='utf-8') as outfile):
        for line in tqdm(infile, total=total_lines, desc="Processing lines"):
            data = json.loads(line.strip())
            if 'title' in data:
                data['title'] = translate_text(data['title'])
            if 'text' in data:
                data['text'] = translate_text(data['text'])
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

input_file = 'queries_default.jsonl'
output_file = 'ru_queries.jsonl'
translate_jsonl(input_file, output_file)
