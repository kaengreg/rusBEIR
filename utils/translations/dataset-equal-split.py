import os
from pathlib import Path


def split_jsonl(input_file, parts=3, output_prefix='corpus_part'):

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"Overall lines: {total_lines}")

    chunk_size = total_lines // parts
    remainder = total_lines % parts

    start = 0
    for i in range(parts):
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        chunk_lines = lines[start:start+current_chunk_size]

        output_file = f"{output_prefix}{i+1}.jsonl"
        Path(output_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as out_f:
            out_f.writelines(chunk_lines)

        print(f" {len(chunk_lines)} lines written in {output_file}")
        start += current_chunk_size


if __name__ == "__main__":
    input_filename = 'corpus.jsonl'

    if not os.path.exists(input_filename):
        print(f"File {input_filename} not found!")
    else:
        split_jsonl(input_filename)
