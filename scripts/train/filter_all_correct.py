#!/usr/bin/env python3
"""
Filter dataset to remove samples with perfect accuracy (100%) from rollout data.
"""

import argparse
import pandas as pd
import glob
import os
from tqdm import tqdm
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Filter dataset to remove 100% accuracy samples")
    parser.add_argument("--dataset", required=True, help="Path to input parquet dataset")
    parser.add_argument("--rollout-dir", required=True, help="Directory containing rollout JSONL files")
    parser.add_argument("--output", required=True, help="Output path for filtered dataset")
    parser.add_argument("--step-min", type=int, default=1, help="Minimum step number (default: 1)")
    parser.add_argument("--step-max", type=int, default=135, help="Maximum step number (default: 135)")
    parser.add_argument("--plot", action="store_true", help="Generate accuracy distribution plot")
    parser.add_argument("--plot-output", help="Output path for plot (default: same as output with .png)")
    parser.add_argument("--threads", type=int, default=32, help="Number of threads for parallel reading")
    return parser.parse_args()

def load_dataset_and_build_index(dataset_path):
    """Load dataset and build prompt index for O(1) lookup."""
    print(f"Loading dataset from {dataset_path}")
    ds = pd.read_parquet(dataset_path)
    print(f"Dataset shape: {ds.shape}")

    def extract_prompt_content(p):
        try:
            return p[0]['content'].strip()
        except Exception:
            return None

    print("Building prompt hash index...")
    ds['clean_prompt'] = ds['prompt'].apply(extract_prompt_content)

    # Filter out rows where extraction failed
    valid_prompts = ds.dropna(subset=['clean_prompt'])
    prompt_to_idx = dict(zip(valid_prompts['clean_prompt'], valid_prompts.index))
    print(f"Established index mapping for {len(prompt_to_idx)} prompts")

    return ds, prompt_to_idx


def read_rollout_files(rollout_dir, step_min, step_max, threads=32):
    """Parallel read rollout JSONL files within step range."""
    print(f"Searching for JSONL files from Step {step_min} to {step_max}...")

    all_files = glob.glob(os.path.join(rollout_dir, "*.jsonl"))
    target_files = []

    for f in all_files:
        try:
            step = int(os.path.basename(f).split('.')[0])
            if step_min <= step <= step_max:
                target_files.append(f)
        except Exception:
            continue

    target_files = sorted(target_files, key=lambda x: int(os.path.basename(x).split('.')[0]))
    print(f"Found {len(target_files)} target files")

    def read_jsonl(file):
        try:
            return pd.read_json(file, lines=True)
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            return pd.DataFrame()

    print("Reading rollout data...")
    with ThreadPoolExecutor(max_workers=threads) as executor:
        rollout_dfs = list(tqdm(executor.map(read_jsonl, target_files),
                               total=len(target_files), desc="Reading JSONL"))

    if not rollout_dfs:
        raise ValueError("No rollout data read")

    t_all = pd.concat(rollout_dfs, ignore_index=True)
    print(f"Total rollout data rows: {len(t_all)}")

    return t_all


def extract_and_match_prompts(rollout_df, prompt_to_idx):
    """Extract prompts from rollout data and match to dataset indices."""
    print("Extracting and matching prompts...")

    pattern = r'(?s)(?:^|\n)user\s+(.*?)(?=\s+assistant|$)'
    extracted = rollout_df['input'].str.extract(pattern)
    rollout_df['extracted_prompt'] = extracted[0].str.strip()
    rollout_df['ds_index'] = rollout_df['extracted_prompt'].map(prompt_to_idx)

    num_matched = rollout_df['ds_index'].notna().sum()
    print(f"Successfully matched rows: {num_matched} / {len(rollout_df)} ({num_matched/len(rollout_df):.2%})")

    return rollout_df


def compute_accuracy_stats(rollout_df, dataset_df):
    """Compute aggregated accuracy statistics."""
    print("Computing accuracy statistics...")

    matched_df = rollout_df.dropna(subset=['ds_index'])
    stats = matched_df.groupby('ds_index')['acc'].agg(['mean', 'count'])

    dataset_df['acc_mean@8'] = dataset_df.index.map(stats['mean'])
    dataset_df['response_count'] = dataset_df.index.map(stats['count'])

    # Fill unmatched samples with random values in [0.5, 0.6)
    na_mask = dataset_df['acc_mean@8'].isna()
    num_na = int(na_mask.sum())
    if num_na > 0:
        print(f"Filling {num_na} unmatched samples with random values in [0.5, 0.6)")
        dataset_df.loc[na_mask, 'acc_mean@8'] = np.random.uniform(0.5, 0.6, size=num_na)

    return dataset_df


def filter_perfect_accuracy(dataset_df):
    """Filter out samples with perfect accuracy (100%)."""
    print("Filtering out perfect accuracy samples...")

    original_count = len(dataset_df)
    filtered_df = dataset_df[dataset_df['acc_mean@8'] < 0.99999].copy()
    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count

    print(f"Original: {original_count}, Filtered: {filtered_count}, Removed: {removed_count}")

    if filtered_count > 0:
        avg_acc = filtered_df['acc_mean@8'].mean()
        print(f"Average accuracy after filtering: {avg_acc:.4f}")

    return filtered_df


def plot_accuracy_distribution(dataset_df, output_path):
    """Plot accuracy distribution histogram."""
    print(f"Generating accuracy distribution plot: {output_path}")

    dataset_df = dataset_df.copy()
    dataset_df['acc_bucket'] = (dataset_df['acc_mean@8'] * 8).round().astype(int).clip(0, 8)
    counts = dataset_df['acc_bucket'].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    buckets = range(9)
    bar_counts = [counts.get(b, 0) for b in buckets]
    colors = ['skyblue'] * 8 + ['lightgray']

    bars = plt.bar(buckets, bar_counts, color=colors, edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({height/len(dataset_df):.1%})',
                    ha='center', va='bottom')

    plt.title(f'Accuracy Distribution (Total: {len(dataset_df)})')
    plt.xlabel('Accuracy Bucket (Correct/Total 8)')
    plt.ylabel('Count')
    plt.xticks(buckets, [f"{b}/8" for b in buckets])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_filtered_dataset(dataset_df, output_path):
    """Save filtered dataset to parquet file."""
    print(f"Saving filtered dataset to: {output_path}")

    # Clean up intermediate columns
    dataset_df = dataset_df.drop(columns=['clean_prompt'], errors='ignore')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset_df.to_parquet(output_path)
    print(f"Saved {len(dataset_df)} samples")


def main():
    args = parse_args()

    # Set default plot output path if not specified
    if args.plot and not args.plot_output:
        args.plot_output = args.output.replace('.parquet', '_distribution.png')

    try:
        # Load dataset and build index
        ds, prompt_to_idx = load_dataset_and_build_index(args.dataset)

        # Read rollout data
        rollout_df = read_rollout_files(args.rollout_dir, args.step_min, args.step_max, args.threads)

        # Extract and match prompts
        rollout_df = extract_and_match_prompts(rollout_df, prompt_to_idx)

        # Compute accuracy statistics
        ds = compute_accuracy_stats(rollout_df, ds)

        # Filter perfect accuracy samples
        filtered_ds = filter_perfect_accuracy(ds)

        # Generate plot if requested
        if args.plot:
            plot_accuracy_distribution(ds, args.plot_output)

        # Save results
        save_filtered_dataset(filtered_ds, args.output)

        print("Processing completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())