from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from code.experiment import (
    read_jsonl, preprocess_split, dataset_statistics, run_plan_item, RUN_PLAN, SEEDS,
)
from code.figures import (
    plot_dataset_statistics, plot_main_comparison, plot_ablation_mapr,
    plot_seed_stability, plot_structure_gain,
)


def main():
    data_dir = ROOT / "data"
    fig_dir = ROOT / "figures"
    res_dir = ROOT / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = data_dir / "poj104"
    train_raw = read_jsonl(raw_dir / "train.jsonl")
    valid_raw = read_jsonl(raw_dir / "valid.jsonl")
    test_raw = read_jsonl(raw_dir / "test.jsonl")

    train_proc = preprocess_split(train_raw)
    valid_proc = preprocess_split(valid_raw)
    test_proc = preprocess_split(test_raw)

    stats = [
        dataset_statistics(train_proc, "train"),
        dataset_statistics(valid_proc, "valid"),
        dataset_statistics(test_proc, "test"),
    ]
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(res_dir / "dataset_statistics.csv", index=False)

    per_seed_records = []
    for plan in RUN_PLAN:
        for seed in SEEDS:
            metrics = run_plan_item(plan, seed, train_proc, test_proc)
            rec = {"model": plan["model"], "scope": plan["scope"], "seed": seed}
            rec.update(metrics)
            per_seed_records.append(rec)

    per_seed_df = pd.DataFrame(per_seed_records)
    per_seed_df.to_csv(res_dir / "per_seed_metrics.csv", index=False)

    metric_keys = ["MAP@R", "Recall@1", "Recall@5", "Recall@10", "MRR"]
    summary = {"models": {}}
    for model in sorted(set(r["model"] for r in per_seed_records)):
        rows = [r for r in per_seed_records if r["model"] == model]
        entry = {}
        for k in metric_keys:
            vals = [r[k] for r in rows]
            entry[k] = {"mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "n": len(vals)}
        summary["models"][model] = entry

    (res_dir / "result_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_dataset_statistics(stats_df, fig_dir)
    plot_main_comparison(summary, fig_dir)
    plot_ablation_mapr(summary, fig_dir)
    plot_seed_stability(per_seed_df, fig_dir)
    plot_structure_gain(summary, fig_dir)

    print("Reproduction summary")
    print("--------------------")
    print("Dataset split sizes (rows / labels / mean tokens):")
    for s in stats:
        print("  {:<8s} {:>6d} / {:>2d} / {:>7.2f}".format(
            s["split"], s["rows"], s["labels"], s["mean_code_tokens"]))
    print("Frozen retrieval metrics (3-seed mean +/- std on POJ-104 test):")
    print("  {:<32s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "model", "MAP@R", "Recall@1", "Recall@5", "Recall@10", "MRR"))
    for model in ["tfidf_token", "tfidf_token_normid", "ast_proxy_only",
                  "gac_tfidf_structure", "gac_tfidf_structure_normid"]:
        e = summary["models"][model]
        print("  {:<32s} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            model, e["MAP@R"]["mean"], e["Recall@1"]["mean"],
            e["Recall@5"]["mean"], e["Recall@10"]["mean"], e["MRR"]["mean"]))
    print("Figures generated:")
    for p in sorted(fig_dir.glob("*.png")):
        print("  " + p.name)


if __name__ == "__main__":
    main()
