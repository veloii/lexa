#!/usr/bin/env python3
"""
Stream a large JSON dataset and produce train / eval JSONL conversation files.

This is a refactor of the prior convert_json_to_jsonl.py script. All file paths
(and the system message) must be provided at runtime; this module does NOT
import constants.py.

Input JSON is expected to be an array of objects shaped like:
[
  {
    "category": "some.dot.separated.label",
    "content": "user question text ..."
  },
  ...
]

Each output line (JSONL) is a list of role/content dicts:
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "label"}
]

Usage:
    python sample_data.py \
        --input path/to/input.json \
        --system-message "Your system prompt" \
        --train-out path/to/train.jsonl \
        --eval-out path/to/eval.jsonl \
        [--eval-duplicate-out path/to/duplicate_eval.jsonl] \
        [--max-entries 10000]

You may also supply the system prompt via a file:
    python sample_data.py ... --system-message-file prompt.txt
(If both --system-message and --system-message-file are given, the direct
--system-message takes precedence.)
"""

import argparse
import json
import sys
import ijson
import shutil
from collections import defaultdict
from pathlib import Path


def extract_category_label(category: str) -> str:
    return category.split('.')[-1]


def create_conversation(content: str, category_label: str, system_message: str):
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": content},
        {"role": "assistant", "content": category_label}
    ]


def stream_parse_json(file_path: Path, system_message: str):
    """
    Stream parse large JSON file to avoid loading everything into memory.
    Yields (category_label, conversation) tuples.
    """
    print(f"Streaming parse of {file_path}...")
    with open(file_path, 'rb') as file:
        parser = ijson.items(file, 'item')
        for i, item in enumerate(parser):
            if i and i % 10000 == 0:
                print(f"Processed {i:,} items...")
            try:
                category = item['category']
                content = item['content']
            except KeyError as e:
                print(f"Warning: Skipping item {i} - missing key {e}")
                continue

            category_label = extract_category_label(category)
            conversation = create_conversation(content, category_label, system_message)
            yield category_label, conversation


def group_by_category(file_path: Path, system_message: str):
    print("Grouping conversations by category...")
    categories = defaultdict(list)
    total_processed = 0
    for category_label, conversation in stream_parse_json(file_path, system_message):
        categories[category_label].append(conversation)
        total_processed += 1
        if total_processed % 10000 == 0:
            print(f"Grouped {total_processed:,} conversations into {len(categories)} categories")
    print(f"Final: {total_processed:,} conversations in {len(categories)} categories")
    return categories


def distribute_entries_evenly(conversations, max_entries):
    """
    Distribute conversations evenly across categories up to max_entries limit.
    Returns a list of selected conversations.
    """
    if not conversations or max_entries <= 0:
        return []
    
    if len(conversations) <= max_entries:
        return conversations
    
    # Calculate how many entries per category we should target
    num_categories = len(set(conv[2]["content"] for conv in conversations))  # assistant content = category label
    target_per_category = max(1, max_entries // num_categories)
    
    # Group by category and select up to target_per_category from each
    category_groups = defaultdict(list)
    for conv in conversations:
        category_label = conv[2]["content"]  # assistant content = category label
        category_groups[category_label].append(conv)
    
    selected = []
    remaining_quota = max_entries
    
    # First pass: take target_per_category from each category
    for category_label in sorted(category_groups.keys()):
        convs = category_groups[category_label]
        take = min(target_per_category, len(convs), remaining_quota)
        selected.extend(convs[:take])
        remaining_quota -= take
        if remaining_quota <= 0:
            break
    
    # Second pass: if we still have quota, fill from categories that have more entries
    if remaining_quota > 0:
        for category_label in sorted(category_groups.keys()):
            if remaining_quota <= 0:
                break
            convs = category_groups[category_label]
            already_taken = min(target_per_category, len(convs))
            available = len(convs) - already_taken
            if available > 0:
                take = min(available, remaining_quota)
                selected.extend(convs[already_taken:already_taken + take])
                remaining_quota -= take
    
    return selected


def split_categories(categories, max_entries=None):
    print("Splitting categories for train/eval...")
    all_train_conversations = []
    all_eval_conversations = []
    
    for category_label, conversations in categories.items():
        count = len(conversations)
        split_point = count // 2
        train_part = conversations[:split_point]
        eval_part = conversations[split_point:]
        all_train_conversations.extend(train_part)
        all_eval_conversations.extend(eval_part)
        print(f"  {category_label}: {len(train_part)} train, {len(eval_part)} eval")
    
    # Apply max_entries limit if specified
    if max_entries is not None:
        print(f"\nApplying max entries limit of {max_entries:,} per file...")
        original_train_count = len(all_train_conversations)
        original_eval_count = len(all_eval_conversations)
        
        all_train_conversations = distribute_entries_evenly(all_train_conversations, max_entries)
        all_eval_conversations = distribute_entries_evenly(all_eval_conversations, max_entries)
        
        print(f"Train: {original_train_count:,} → {len(all_train_conversations):,}")
        print(f"Eval:  {original_eval_count:,} → {len(all_eval_conversations):,}")
    
    return all_train_conversations, all_eval_conversations


def write_jsonl(conversations, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(conversations):,} conversations to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for conversation in conversations:
            f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
    print(f"✅ Written {output_path}")


def duplicate_eval(eval_path: Path, duplicate_path: Path):
    duplicate_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(eval_path, duplicate_path)
    print(f"📄 Duplicated eval file to {duplicate_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate train/eval JSONL conversation files.")
    parser.add_argument("--input", required=True, dest="input_file", type=Path,
                        help="Path to input JSON file.")
    parser.add_argument("--system-message", dest="system_message",
                        help="System message string.")
    parser.add_argument("--system-message-file", dest="system_message_file", type=Path,
                        help="File containing system message text.")
    parser.add_argument("--train-out", required=True, dest="train_out", type=Path,
                        help="Path to write train JSONL file.")
    parser.add_argument("--eval-out", required=True, dest="eval_out", type=Path,
                        help="Path to write eval JSONL file.")
    parser.add_argument("--eval-duplicate-out", dest="eval_duplicate_out", type=Path,
                        help="Optional additional path to also copy the eval JSONL.")
    parser.add_argument("--max-entries", dest="max_entries", type=int,
                        help="Maximum number of entries per output file (train and eval). "
                             "Categories will be distributed as evenly as possible within this limit.")
    return parser.parse_args()


def resolve_system_message(args) -> str:
    if args.system_message:
        return args.system_message
    if args.system_message_file:
        return args.system_message_file.read_text(encoding="utf-8")
    raise SystemExit("Error: You must supply --system-message or --system-message-file")


def main():
    args = parse_args()
    if not args.input_file.exists():
        print(f"Error: {args.input_file} not found", file=sys.stderr)
        sys.exit(1)

    system_message = resolve_system_message(args)
    print(f"System message preview: {system_message[:100]}{'...' if len(system_message) > 100 else ''}")
    
    if args.max_entries is not None:
        print(f"Max entries per file: {args.max_entries:,}")
    print()

    try:
        categories = group_by_category(args.input_file, system_message)
        train_conversations, eval_conversations = split_categories(categories, args.max_entries)
        print(f"\nTotal: {len(train_conversations):,} train, {len(eval_conversations):,} eval")

        write_jsonl(train_conversations, args.train_out)
        write_jsonl(eval_conversations, args.eval_out)

        if args.eval_duplicate_out:
            duplicate_eval(args.eval_out, args.eval_duplicate_out)

        print("\n🎉 Conversion complete!")
        print(f"   Train: {args.train_out}")
        print(f"   Eval:  {args.eval_out}")
        if args.eval_duplicate_out:
            print(f"   Eval duplicate: {args.eval_duplicate_out}")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
