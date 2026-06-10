import re
import json
import argparse
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from typing import List
from openai import OpenAI
from tqdm import tqdm
import os
from pathlib import Path


class QwenInstructTranslatorViaVLLM:
    def __init__(
        self,
        base_url: str = "",
        api_key: str = "123",
        model: str = "Qwen2.5-72B-Instruct",
        source_language: str = "Russian",
        target_language: str = "English",
        prompt_system: str = (
            "You are a professional translator with expertise in both {source_language} and {target_language}. "
        ),
        temperature: float = 0.0,
        max_chunk_size: int = 5000,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
        self.source_language = source_language
        self.target_language = target_language
        self.temperature = temperature
        self.max_chunk_size = max_chunk_size
        self.prompt_system = prompt_system.format(
            source_language=self.source_language,
            target_language=self.target_language
        )

    def split_into_sentences(self, text: str) -> List[str]:
        text = text.replace('\n', ' ')
        sentences = re.split(r'(?<=[.?!])\s+', text)
        return sentences

    def chunk_sentence(self, sentence: str) -> List[str]:
        if len(sentence) <= self.max_chunk_size:
            return [sentence]
        chunks = []
        words = sentence.split()
        current_words = []
        current_len = 0
        for w in words:
            if current_len + len(w) + 1 <= self.max_chunk_size:
                current_words.append(w)
                current_len += len(w) + 1
            else:
                chunks.append(' '.join(current_words))
                current_words = [w]
                current_len = len(w)
        if current_words:
            chunks.append(' '.join(current_words))
        return chunks

    def chunk_text(self, text: str) -> List[str]:
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
        
            if len(sentence) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                long_sentence_chunks = self.chunk_sentence(sentence)
                chunks.extend(long_sentence_chunks)
                continue
            
            
            additional_length = len(sentence) + (1 if current_chunk else 0)
            
            if current_length + additional_length > self.max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += additional_length
        
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def fix_json(self, broken_str: str) -> str:
        pattern = r'"translation"\s*:\s*"([^"]*)"'
        match = re.search(pattern, broken_str)

        if match:
            translation_text = match.group(1)
        else:
            print("BROKEN STR", broken_str)
            translation_text = broken_str.replace("\"translation\":", "").strip()
            print("BROKEN STR FIXED", translation_text)
            
        fixed = json.dumps({"translation": translation_text}, ensure_ascii=False)
        return fixed


    def translate_chunk(self, text_chunk: str) -> str:
        max_attempts = 3
        attempt = 1
        translated_text = ""
        hieroglyphs_pattern = re.compile(r'[\u4e00-\u9fff]')

        while attempt <= max_attempts:
            messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional translator with expertise in both."
                            "You are a machine and you have only one task - to translate text."
                            f"{self.source_language} and {self.target_language}."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "Please produce a clear, accurate, and grammatically correct translation "
                            f"from {self.source_language} to {self.target_language}.\n"
                            "- The text may contain extra punctuation, such as quotes (" "), commas, or brackets ( () )."
                            "- Consider such punctuation or formatting as non-essential unless it is crucial to the meaning"
                            "- The output must be valid JSON with a single key \"translation\".\n"
                            "- Do not add commentary, notes, or explanations about terminology.\n"
                            "- If there are multiple choices, choose the best one.\n"
                            "- All text should be in commas or double quotes at the begging and at the end.\n"
                            "- If double quotes are present in the text, they should be escaped with a backslash.\n"
                            "**Text to translate**\n"
                            "Hello world!"
                        )
                    },
                    {"role": "assistant",   "content": "{\"translation\": \"Привет, мир!\"}"},
                    {
                        "role": "user",
                        "content": (
                            "Please produce a clear, accurate, and grammatically correct translation "
                            f"from {self.source_language} to {self.target_language}.\n"
                            "- The text may contain extra punctuation, such as quotes (" "), commas, or brackets ( () )."
                            "- Consider such punctuation or formatting as non-essential unless it is crucial to the meaning"
                            "- The output must be valid JSON with a single key \"translation\".\n"
                            "- Do not add commentary, notes, or explanations about terminology.\n"
                            "- If there are multiple choices, choose the best one.\n"
                            "- All text should be in commas or double quotes at the begging and at the end.\n"
                            "- If double quotes are present in the text, they should be escaped with a backslash.\n"
                            "**Text to translate**\n"
                            "Has there been a proposal to confiscate firearms from individuals who already possess them?"
                        )
                    },
                    {"role": "assistant",   "content": "{\"translation\": \"Поступало ли предложение конфисковывать огнестрельное оружие у лиц, которые им уже владеют?\"}"},
                    {
                        "role": "user",
                        "content": (
                            "Please produce a clear, accurate, and grammatically correct translation "
                            f"from {self.source_language} to {self.target_language}.\n"
                            "- The text may contain extra punctuation, such as quotes (" "), commas, or brackets ( () )."
                            "- Consider such punctuation or formatting as non-essential unless it is crucial to the meaning"
                            "- The output must be valid JSON with a single key \"translation\".\n"
                            "- Do not add commentary, notes, or explanations about terminology.\n"
                            "- If there are multiple choices, choose the best one.\n"
                            "- All text should be in commas or double quotes at the begging and at the end.\n"
                            "- If double quotes are present in the text, they should be escaped with a backslash.\n"
                            "**Text to translate**\n"
                            "\"Who is\" or \"Who are\"?"
                        )
                    },
                    {"role": "assistant",   "content": "{\"translation\": \"\"Кто есть\" или \"Кто являются\"?\"}"},
                    {
                        "role": "user",
                        "content": (
                            "Please produce a clear, accurate, and grammatically correct translation "
                            f"from {self.source_language} to {self.target_language}.\n"
                            "- The text may contain extra punctuation, such as quotes (" "), commas, or brackets ( () )."
                            "- Consider such punctuation or formatting as non-essential unless it is crucial to the meaning"
                            "- The output must be valid JSON with a single key \"translation\".\n"
                            "- Do not add commentary, notes, or explanations about terminology.\n"
                            "- If there are multiple choices, choose the best one.\n"
                            "- All text should be in commas or double quotes at the begging and at the end.\n"
                            "- If double quotes are present in the text, they should be escaped with a backslash.\n"
                            "**Text to translate**\n"
                            "What for LLM is used?"
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "Please produce a clear, accurate, and grammatically correct translation "
                            f"from {self.source_language} to {self.target_language}.\n"
                            "- The text may contain extra punctuation, such as quotes (" "), commas, or brackets ( () )."
                            "- Consider such punctuation or formatting as non-essential unless it is crucial to the meaning"
                            "- The output must be valid JSON with a single key \"translation\".\n"
                            "- Do not add commentary, notes, or explanations about terminology.\n"
                            "- If there are multiple choices, choose the best one.\n"
                            "- All text should be in commas or double quotes at the begging and at the end.\n"
                            "- If double quotes are present in the text, they should be escaped with a backslash.\n"
                            "**Text to translate**\n"
                            f"{text_chunk}"
                        )
                    },
                    {"role": "assistant",  "content": "{\"translation\": "},
            ]
            
            try: 
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=0.9,
                    extra_body={
                        "repetition_penalty": 1.0,
                        "add_generation_prompt": False,
                        "continue_final_message": True
                    }
                )
                #print(response)
            except Exception as e:
                tqdm.write(f"[Process {os.getpid()}] [Chunk Error] Attempt {attempt}/{max_attempts} for chunk failed with error: {e}")
                attempt += 1
                time.sleep(1)
                continue

            
            candidate = response.choices[0].message.content.strip()
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                candidate = self.fix_json(candidate)
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    tqdm.write("JSON not valid even after fix. Retrying…")
                    attempt += 1
                    time.sleep(1)
                    continue
            
            print(candidate)
            candidate_translation = parsed.get("translation", "").strip()

            error_candidate = candidate_translation
            if (candidate_translation and not hieroglyphs_pattern.search(candidate_translation)):
                translated_text = candidate_translation
                break

            if translated_text:
                break
            else:
                print("ERROR!!!")
                tqdm.write(f"[Process {os.getpid()}] [Chunk Warning] No valid translation returned in attempt {attempt}/{max_attempts}. "
                           f"Error occured on translation: {error_candidate} Retrying...")
                
                attempt += 1
                time.sleep(1)
            
        return translated_text

    def translate_long_text(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        chunks = self.chunk_text(text)
        translated_parts = []
        for chunk in chunks:
            translated_parts.append(self.translate_chunk(chunk))
        return " ".join(translated_parts)

    def translate(self, title: str, text: str) -> (str, str):
        translated_title = self.translate_long_text(title) if title else ""
        translated_text = self.translate_long_text(text) if text else ""
        return translated_title, translated_text

    def count_file_lines(self, filepath: str) -> int:
        count = 0
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        count += 1
        return count

    def translate_jsonl(
        self,
        input_file_path: str,
        output_file_path: str,
        checkpoint_file: str = "line_count_checkpoint.txt",
        progress_bar_position: int = 0,
    ):

        total_lines = self.count_file_lines(input_file_path)
        resume_line = self.count_file_lines(output_file_path)

        processed_new = 0

        Path(output_file_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with tqdm(total=total_lines, initial=resume_line, desc=f"Process {os.getpid()} File {progress_bar_position+1}", unit="lines", position=progress_bar_position) as pbar, \
                open(output_file_path, 'a', encoding='utf-8') as fout:
            with open(input_file_path, 'r', encoding='utf-8') as fin:
                for _ in range(resume_line):
                    next(fin, None)

                for line in fin:
                    pbar.update(1)
                    pbar.refresh()
                    resume_line += 1

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    title = data.get("title", "")
                    text = data.get("text", "")

                    max_line_attempts = 3
                    line_attempt = 0
                    translated_title, translated_text = "", ""
                    while line_attempt < max_line_attempts:
                        try:
                            translated_title, translated_text = self.translate(title, text)
                            break
                        except Exception as e:
                            line_attempt += 1
                            tqdm.write(f"[Line {resume_line} Error in File {progress_bar_position + 1}] Translation attempt {line_attempt}/{max_line_attempts} failed with error: {e}")
                            time.sleep(1)

                    data["title"] = translated_title
                    data["text"] = translated_text

                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")

                    processed_new += 1

        print(f"Translation completed for File {progress_bar_position+1}. Lines read: {resume_line}. Lines translated: {processed_new}.")

def run_translation(
    input_file_path: str,
    output_file_path: str,
    checkpoint_file: str,
    base_url: str,
    api_key: str,
    model: str,
    source_language: str,
    target_language: str,
    temperature: float,
    max_chunk_size: int,
    progress_bar_position: int = 0,
):
    translator = QwenInstructTranslatorViaVLLM(
        base_url=base_url,
        api_key=api_key,
        model=model,
        source_language=source_language,
        target_language=target_language,
        temperature=temperature,
        max_chunk_size=max_chunk_size
    )
    translator.translate_jsonl(
        input_file_path=input_file_path,
        output_file_path=output_file_path,
        checkpoint_file=checkpoint_file,
        progress_bar_position=progress_bar_position
    )


def process_file(params: dict):
    run_translation(
        input_file_path=params["input_file_path"],
        output_file_path=params["output_file_path"],
        checkpoint_file=params.get("checkpoint_file", "line_count_checkpoint.txt"),
        base_url=params.get("base_url", ""),
        api_key=params.get("api_key", "123"),
        model=params.get("model", "Qwen2.5-72B-Instruct"),
        source_language=params.get("source_lang", "English"),
        target_language=params.get("target_lang", "Russian"),
        temperature=params.get("temperature", 0.0),
        max_chunk_size=params.get("max_chunk_size", 5000),
        progress_bar_position=params.get("progress_bar_position", 0)
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true",
                        help="Multiprocessing mode (batch mode)")
    parser.add_argument(
        "--input", "-i", help="Path to input JSONL file (single mode)")
    parser.add_argument(
        "--output", "-o", help="Path to output JSONL file (single mode)")
    parser.add_argument("--checkpoint", "-c", default="line_count_checkpoint.txt",
                        help="Path to checkpoint file (single mode)")
    parser.add_argument(
        "--base_url", default="", help="API base URL")
    parser.add_argument("--api_key", default="123", help="API key")
    parser.add_argument(
        "--model", default="Qwen2.5-72B-Instruct", help="Model's path")
    parser.add_argument("--source_language",
                        default="English", help="Source language")
    parser.add_argument("--target_language",
                        default="Russian", help="Target language")
    parser.add_argument("--temperature", type=float,
                        default=0.0, help="model's temperature")
    parser.add_argument("--max_chunk_size", type=int,
                        default=5000, help="Max chunk size")
    return parser.parse_args()


def get_file_paths(base_dir, parts_count=5):
    input_files = [f"{base_dir}/corpus_part{i}.jsonl" for i in range(1, parts_count + 1)]
    output_files = [f"{base_dir}/corpus_part{i}_translated.jsonl" for i in range(1, parts_count + 1)]
    checkpoint_files = [f"{base_dir}/line_count_checkpoint_part{i}.txt" for i in range(1, parts_count + 1)]
    return input_files, output_files, checkpoint_files

def main():
    args = parse_args()

    if args.batch:
        base_dir = "fiqa/fiqa-parts/qwen-2"
        parts_count = 5
        input_files, output_files, checkpoint_files = get_file_paths(base_dir, parts_count)

        params_list = []
        for idx, (inp, out, chk) in enumerate(zip(input_files, output_files, checkpoint_files)):
            params = {
                "input_file_path": inp,
                "output_file_path": out,
                "checkpoint_file": chk,
                "base_url": args.base_url,
                "api_key": args.api_key,
                "model": args.model,
                "source_lang": args.source_language,
                "target_lang": args.target_language,
                "temperature": args.temperature,
                "max_chunk_size": args.max_chunk_size,
                "progress_bar_position": idx
            }
            params_list.append(params)

        overall_start_time = time.time()

        with ProcessPoolExecutor(max_workers=parts_count) as executor:
            futures = {executor.submit(process_file, params): params for params in params_list}

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    params = futures[future]
                    print(f"Error with file: {params['input_file_path']}: {exc}")

        overall_end_time = time.time()
        total_time = overall_end_time - overall_start_time
        log_message = f"Total time for all processes: {total_time:.2f} seconds\n"
        log_path = f"{base_dir}/total_time_qwen_log.txt"
        Path(log_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_message)
        print(log_message)

    else:
        if not args.input or not args.output:
            print("Error: Single mode requires --input and --output arguments.")
            exit(1)
        run_translation(
            input_file_path=args.input,
            output_file_path=args.output,
            checkpoint_file=args.checkpoint,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            source_language=args.source_language,
            target_language=args.target_language,
            temperature=args.temperature,
            max_chunk_size=args.max_chunk_size,
            progress_bar_position=0
        )

if __name__ == '__main__':
    main()
