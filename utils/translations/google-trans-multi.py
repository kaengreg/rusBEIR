import json
import os
import re
import time
from tqdm import tqdm
import multiprocessing
import sys
from googletrans import Translator
from pathlib import Path

PROGRAMM_START_TIME = time.time()

def read_checkpoint(checkpoint_file="line_count_checkpoint.txt"):
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return int(f.read().strip())
    except:
        return 0

def write_checkpoint(value, checkpoint_file="line_count_checkpoint.txt"):
    Path(checkpoint_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        f.write(str(value))

def count_total_lines(filepath):
    total = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                total += 1
    return total

def split_into_sentences(text):
    text = text.replace('\n', ' ')
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return sentences

def chunk_sentence(sentence, max_chunk_size=5000):
    if len(sentence) <= max_chunk_size:
        return [sentence]
    
    chunks = []
    words = sentence.split()
    current_words = []
    current_len = 0
    
    for w in words:
        if current_len + len(w) + 1 <= max_chunk_size:
            current_words.append(w)
            current_len += len(w) + 1
        else:
            chunks.append(' '.join(current_words))
            current_words = [w]
            current_len = len(w)
    
    if current_words:
        chunks.append(' '.join(current_words))
    
    return chunks

def chunk_text(text, max_chunk_size=5000):
    sentences = split_into_sentences(text)
    all_chunks = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= max_chunk_size:
            all_chunks.append(sent)
        else:
            all_chunks.extend(chunk_sentence(sent, max_chunk_size))
    return all_chunks

class GTranslatorWrapper:
    def __init__(self, source, target):
        self.source = None if source == 'auto' else source  
        self.target = target
        self.translator = Translator()

    def translate(self, text):
        result = self.translator.translate(text, src=self.source, dest=self.target)
        return result.text

def translate_text(text, translator_obj, max_chunk_size=5000):
    text = text.strip()
    if not text:
        return ""
    
    chunks = chunk_text(text, max_chunk_size=max_chunk_size)
    translated_chunks = []
    
    for ch in chunks:
        if not ch.strip():
            translated_chunks.append("")
            continue

        attempt = 0
        max_attempts = 3
        translated = None
        while attempt < max_attempts:
            try:
                translated = translator_obj.translate(ch)
                break  
            except Exception as e:
                error_message = str(e)
                if "too many requests" in error_message.lower():
                    overall_end_time = time.time()
                    total_time = overall_end_time - PROGRAMM_START_TIME
                    log_message = f"Error: Too many requests. Overall time: {total_time:.2f} seconds\n"
                    log_file = '/Users/kaengreg/Documents/Работа /НИВЦ/rusBeIR-home/rusBeIR/translations/touche-2020/time_log.txt'
                    Path(log_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(log_message)
                    print(log_message)
                    sys.exit(1)
                elif "no translation was found using the current translator" in error_message.lower():
                    attempt += 1
                    print("No translation was found using the current translator. Retrying translation for the same text...")
                    time.sleep(1)  
                else:
                    print(f"Error with translation: {e}")
                    print(f"Error caused by text: {ch}")
                    translated = ""
                    break

        if translated is None:
            translated = ""
        translated_chunks.append(translated)
    
    return " ".join(translated_chunks)

def load_already_translated_ids(output_file_path):
    if not os.path.exists(output_file_path):
        return set()
    
    translated_ids = set()
    with open(output_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                doc_id = data.get("_id")
                if doc_id:
                    translated_ids.add(doc_id)
            except:
                pass
    return translated_ids

def main(
    input_file_path,
    output_file_path,
    source_lang='en',
    target_lang='ru',
    max_chunk_size=5000,
    pause_each=10000,
    pause_time=5*60,
    checkpoint_file="line_count_checkpoint.txt",
    position=0
):
    total_lines = count_total_lines(input_file_path)
    line_count = read_checkpoint(checkpoint_file)

    translator_obj = GTranslatorWrapper(source_lang, target_lang)

    already_translated_ids = load_already_translated_ids(output_file_path)
    processed_new = 0  

    with open(input_file_path, 'r', encoding='utf-8') as fin:
        for _ in range(line_count):
            next(fin, None)
        
        with tqdm(
            total=total_lines,
            initial=line_count,
            desc=f"Processing {os.path.basename(input_file_path)}",
            position=position,
            leave=True
        ) as pbar:
            Path(output_file_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, 'a', encoding='utf-8') as fout:
            
                for line in fin:
                    line = line.strip()
                    pbar.update(1)
                    line_count += 1

                    if not line:
                        continue
                
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    doc_id = data.get("_id")
                    if doc_id in already_translated_ids:
                        continue

                    title = data.get("title", "")
                    text = data.get("text", "")
                
                    if title.strip():
                        data["title"] = translate_text(title, translator_obj, max_chunk_size)
                    else:
                        data["title"] = ""
                
                    if text.strip():
                        data["text"] = translate_text(text, translator_obj, max_chunk_size)
                    else:
                        data["text"] = ""

                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                    already_translated_ids.add(doc_id)
                    processed_new += 1

                    if processed_new % pause_each == 0:
                        print(f"Translated {processed_new} new documents. Pause {pause_time} seconds...")
                        fout.flush()
                        time.sleep(pause_time)

                    if line_count % 1000 == 0:
                        write_checkpoint(line_count, checkpoint_file)

    write_checkpoint(line_count, checkpoint_file)
    print(f"File {input_file_path} processing ended. Lines read overall: {line_count}. Newly translated: {processed_new}")

def process_file(params):
    main(
        input_file_path=params["input_file_path"],
        output_file_path=params["output_file_path"],
        source_lang=params.get("source_lang", "auto"),
        target_lang=params.get("target_lang", "en"),
        max_chunk_size=params.get("max_chunk_size", 5000),
        pause_each=params.get("pause_each", 10000),
        pause_time=params.get("pause_time", 5*60),
        checkpoint_file=params.get("checkpoint_file", "line_count_checkpoint.txt"),
        position=params.get("position", 0)
    )

if __name__ == "__main__":
    input_files = [
        '/Users/kaengreg/Documents/Работа /НИВЦ/rusBeIR-home/rusBeIR/translations/touche-2020/corpus_part1.jsonl',
        '/Users/kaengreg/Documents/Работа /НИВЦ/rusBeIR-home/rusBeIR/translations/touche-2020/corpus_part2.jsonl',
        '/Users/kaengreg/Documents/Работа /НИВЦ/rusBeIR-home/rusBeIR/translations/touche-2020/corpus_part3.jsonl'
    ]

    output_files = [
        '/Users/kaengreg/Documents/Работа /НИВЦ/rusBeIR-home/rusBeIR/translations/touche-2020/corpus_part1_ru_google.jsonl',
        '/Users/kaengreg/Documents/Работа /НИВЦ/rusBeIR-home/rusBeIR/translations/touche-2020/corpus_part2_ru_google.jsonl',
        '/Users/kaengreg/Documents/Работа /НИВЦ/rusBeIR-home/rusBeIR/translations/touche-2020/corpus_part3_ru_google.jsonl'
    ]

    checkpoint_files = [
        '/Users/kaengreg/Documents/Работа /НИВЦ/rusBeIR-home/rusBeIR/translations/touche-2020/line_count-chekcpoint-part1.jsonl',
        '/Users/kaengreg/Documents/Работа /НИВЦ/rusBeIR-home/rusBeIR/translations/touche-2020/line_count-chekcpoint-part2.jsonl',
        '/Users/kaengreg/Documents/Работа /НИВЦ/rusBeIR-home/rusBeIR/translations/touche-2020/line_count-chekcpoint-part3.jsonl'
    ]

    source_lang = "en"
    target_lang = "ru"
    max_chunk_size = 5000
    pause_each = 10000
    pause_time = 5 * 60  

    processes = []
    try: 
        for i, (inp, out, chk) in enumerate(zip(input_files, output_files, checkpoint_files)):
            params = {
                "input_file_path": inp,
                "output_file_path": out,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "max_chunk_size": max_chunk_size,
                "pause_each": pause_each,
                "pause_time": pause_time,
                "checkpoint_file": chk,
                "position": i  
            }
            p = multiprocessing.Process(target=process_file, args=(params,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt. Terminating all processes...")
        for p in processes:
            if p.is_alive():
                p.terminate()

        for p in processes:
            p.join()

    finally:
        overall_end_time = time.time()
        total_time = overall_end_time - PROGRAMM_START_TIME

        log_message = f"Overall time: {total_time:.2f} seconds\n"
        log_file = '/Users/kaengreg/Documents/Работа /НИВЦ/rusBeIR-home/rusBeIR/translations/touche-2020/time_log.txt'
        Path(log_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_message)

        print(log_message)
