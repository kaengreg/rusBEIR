# %%
import os 
import json
import argparse
import re
from pathlib import Path

# %%
def merge_jsonl(input_dir, output_file):
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    jsonl_files.sort()

    Path(output_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as out_f:
        for filename in jsonl_files:
            file_path = os.path.join(input_dir, filename)
            print(f"Processing file: {file_path}")
            with open(file_path, 'r') as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line)
    print(f"\n Merge complete. Result saved to {output_file}")

# %%
def sort_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]

    sorted_records = sorted(records, key=lambda x: x['_id'])

    Path(output_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in sorted_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing jsonl files to merge")
    parser.add_argument("--output_file", type=str, required=True, help="Output file to save merged jsonl files")
    args = parser.parse_args()

    merge_jsonl(args.input_dir, args.output_file)
    sort_jsonl(args.output_file, args.output_file)


if __name__ == '__main__':
    main()

# %%


