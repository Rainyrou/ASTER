import json, os, glob, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from math_verify import math_metric, LatexExtractionConfig, ExprExtractionConfig
except ImportError:
    class TimeoutException(Exception): pass
    assert 0
    print("Warning: math_verify not found.")


def compute_single_score(model_output, ground_truth):
    if not model_output:
        return 0.0

    tail = model_output[-500:]
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig())
    )

    gt_boxed = "\\boxed{" + str(ground_truth) + "}"

    try:
        ret, _ = verify_func([gt_boxed], [tail])
        return float(ret[0]) if isinstance(ret, list) else float(ret)
    except:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing 0.jsonl, 10.jsonl, etc.")
    parser.add_argument("--outdir", default="step_report", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    target_sources = ["aime2025", "hmmt25"]
    step_scores = {src: {"step": [], "score": []} for src in target_sources}

    jsonl_files = sorted(glob.glob(os.path.join(args.input_dir, "*.jsonl")),
                         key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    for jsonl_path in jsonl_files:
        step = int(os.path.splitext(os.path.basename(jsonl_path))[0])
        scores = {src: [] for src in target_sources}

        with open(jsonl_path) as f:
            for line in f:
                try:
                    item = json.loads(line)
                    src = item.get("non_tensor_data_source")

                    if src not in target_sources:
                        continue

                    gt = item.get("non_tensor_reward_model", {}).get("ground_truth") or item.get("ground_truth")

                    if not gt:
                        continue

                    sc = compute_single_score(item.get("output", ""), gt)
                    scores[src].append(sc)
                except:
                    continue

        for src in target_sources:
            print(src, len(scores[src]))
            avg = np.mean(scores[src]) if scores[src] else 0.0
            step_scores[src]["step"].append(step)
            step_scores[src]["score"].append(avg)

    print(step_scores)

    # 1. Plot the curve
    plt.figure(figsize=(7, 4))
    for src in target_sources:
        plt.plot(step_scores[src]["step"], step_scores[src]["score"],
                 marker='o', label=src)

    plt.xlabel("Step")
    plt.ylabel("Average Score")
    plt.title("AIME2025 vs HMMT25 Score Changes by Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    png_path = os.path.join(args.outdir, "step_curve.png")
    plt.savefig(png_path, dpi=300)
    plt.close()

    print(f"Saved -> {png_path}")

    # 2. Save CSV (one file per dataset)
    for src in target_sources:
        df = pd.DataFrame({"step": step_scores[src]["step"],
                           "avg_score": step_scores[src]["score"]})
        csv_path = os.path.join(args.outdir, f"{src}.csv")
        df.to_csv(csv_path, index=False)
        print(f"CSV -> {csv_path}")


if __name__ == "__main__":
    main()
