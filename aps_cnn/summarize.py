from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_runs(root: Path):
    rows = []
    for path in sorted(root.glob("*/*/metrics.json")):
        data = json.loads(path.read_text())
        test = data["test"]
        rows.append({
            "dataset": data["dataset"], "mode": data["mode"],
            "epochs": data["epochs"], "seed": data["seed"],
            "accuracy": test["accuracy"], "macro_precision": test["macro_precision"],
            "macro_recall": test["macro_recall"], "macro_f1": test["macro_f1"],
            "loss": test["loss"], "duration_seconds": data["duration_seconds"],
        })
    return rows


def ablation_deltas(root: Path, datasets: list[str]):
    deltas = []
    for dataset in datasets:
        files = {
            mode: root / dataset / mode / "metrics.json"
            for mode in ("baseline", "ablation")
        }
        if not all(path.exists() for path in files.values()):
            continue
        baseline, ablation = (json.loads(files[mode].read_text())
                              for mode in ("baseline", "ablation"))
        for field in ("seed", "epochs", "batch_size", "limit"):
            if baseline[field] != ablation[field]:
                raise SystemExit(f"Configuração incompatível da ablação em {dataset}: {field}")
        if not baseline["dropout_enabled"] or ablation["dropout_enabled"]:
            raise SystemExit(f"Esperava dropout ligado no baseline e desligado na ablação ({dataset})")
        count_base = sum(map(sum, baseline["test"]["confusion_matrix"]))
        count_ablation = sum(map(sum, ablation["test"]["confusion_matrix"]))
        if count_base != count_ablation:
            raise SystemExit(f"Tamanho de teste diferente entre baseline e ablação ({dataset})")
        base_test, abl_test = baseline["test"], ablation["test"]
        deltas.append({
            "dataset": dataset,
            "test_examples": count_base,
            "seed": baseline["seed"],
            "epochs": baseline["epochs"],
            "dropout_accuracy": base_test["accuracy"],
            "no_dropout_accuracy": abl_test["accuracy"],
            "delta_accuracy_pp": 100 * (abl_test["accuracy"] - base_test["accuracy"]),
            "dropout_macro_f1": base_test["macro_f1"],
            "no_dropout_macro_f1": abl_test["macro_f1"],
            "delta_macro_f1_pp": 100 * (abl_test["macro_f1"] - base_test["macro_f1"]),
        })
    return deltas


def main():
    parser = argparse.ArgumentParser(description="Resume métricas e gera gráficos dos experimentos.")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    rows = read_runs(args.run_root)
    if not rows:
        raise SystemExit(f"Nenhum metrics.json encontrado em {args.run_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metrics_path in sorted(args.run_root.glob("*/*/metrics.json")):
        data = json.loads(metrics_path.read_text())
        stem = f"{data['dataset']}_{data['mode']}"
        (args.output_dir / f"metrics_{stem}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        )
        curve = metrics_path.parent / "training.png"
        if curve.exists():
            shutil.copy2(curve, args.output_dir / f"training_{stem}.png")

    columns = list(rows[0])
    with (args.output_dir / "metrics_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    datasets = sorted({row["dataset"] for row in rows})
    deltas = ablation_deltas(args.run_root, datasets)
    modes = [mode for mode in ("baseline", "pretrained", "ablation")
             if any(row["mode"] == mode for row in rows)]
    metrics = [("accuracy", "Acurácia"), ("macro_f1", "F1 macro")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5))
    x = np.arange(len(datasets))
    width = 0.8 / len(modes)
    for axis, (metric, title) in zip(axes, metrics):
        for index, mode in enumerate(modes):
            values = []
            for dataset in datasets:
                match = next((row[metric] for row in rows
                              if row["dataset"] == dataset and row["mode"] == mode), np.nan)
                values.append(match)
            axis.bar(x + (index - (len(modes) - 1) / 2) * width, values,
                     width=width, label=mode)
        axis.set_xticks(x, datasets)
        axis.set_ylim(0, 1.02)
        axis.set_ylabel("Escore")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle("Comparação dos experimentos no conjunto de teste")
    fig.tight_layout()
    fig.savefig(args.output_dir / "comparison.png", dpi=180)
    plt.close(fig)

    if deltas:
        with (args.output_dir / "ablation_summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(deltas[0]))
            writer.writeheader()
            writer.writerows(deltas)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for axis, (metric, title) in zip(axes, metrics):
            values_base = [next((row[metric] for row in rows
                                 if row["dataset"] == dataset and row["mode"] == "baseline"), np.nan)
                           for dataset in datasets]
            values_abl = [next((row[metric] for row in rows
                                if row["dataset"] == dataset and row["mode"] == "ablation"), np.nan)
                          for dataset in datasets]
            for mode, values in (("dropout ligado", values_base), ("dropout desligado", values_abl)):
                axis.plot(datasets, values, marker="o", linewidth=2, label=mode)
            axis.set_ylim(0, 1.02)
            axis.set_ylabel("Escore")
            axis.set_title(title)
            axis.grid(alpha=0.25)
            axis.legend()
        fig.suptitle("Ablação: dropout ligado e desligado")
        fig.tight_layout()
        fig.savefig(args.output_dir / "ablation.png", dpi=180)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(7.5, 4.5))
        x = np.arange(len(deltas))
        width = 0.34
        accuracy = [item["delta_accuracy_pp"] for item in deltas]
        f1 = [item["delta_macro_f1_pp"] for item in deltas]
        bars_a = axis.bar(x - width / 2, accuracy, width, label="Acurácia")
        bars_f = axis.bar(x + width / 2, f1, width, label="F1 macro")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylim(min(0, min(accuracy + f1)) - 0.8,
                      max(0, max(accuracy + f1)) + 0.9)
        axis.set_xticks(x, [item["dataset"] for item in deltas])
        axis.set_ylabel("Sem dropout − com dropout (pontos percentuais)")
        axis.set_title("Efeito de remover o dropout no conjunto de teste")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        for bars in (bars_a, bars_f):
            for bar in bars:
                offset = 0.08 if bar.get_height() >= 0 else -0.08
                axis.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + offset,
                          f"{bar.get_height():+.2f}", ha="center",
                          va="bottom" if offset > 0 else "top", fontsize=9)
        fig.tight_layout()
        fig.savefig(args.output_dir / "ablation_effect.png", dpi=180)
        plt.close(fig)

    for row in rows:
        print(f"{row['dataset']:7s} {row['mode']:10s} accuracy={row['accuracy']:.4f} "
              f"macro-F1={row['macro_f1']:.4f} loss={row['loss']:.4f}")
    print(f"Resumo salvo em {args.output_dir}")


if __name__ == "__main__":
    main()
