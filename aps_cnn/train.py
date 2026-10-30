from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from aps_cnn.data import make_loaders
from aps_cnn.models import SmallCNN, make_pretrained_resnet18


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epoch_pass(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if getattr(model, "_frozen_feature_extractor", False):
        # Mantenha BatchNorms do backbone em eval durante o linear probe.
        model.eval()
        model.fc.train(training)
    loss_sum = correct = count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        batch = labels.size(0)
        loss_sum += loss.item() * batch
        correct += (logits.argmax(1) == labels).sum().item()
        count += batch
    return {"loss": loss_sum / count, "accuracy": correct / count}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: int = 10):
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    tp = np.diag(matrix).astype(float)
    precision = tp / np.maximum(matrix.sum(axis=0), 1)
    recall = tp / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(tp.sum() / max(matrix.sum(), 1)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": matrix.tolist(),
    }


@torch.inference_mode()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = total = 0.0
    truths, predictions = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        total += labels.size(0)
        truths.append(labels.cpu().numpy())
        predictions.append(logits.argmax(1).cpu().numpy())
    result = classification_metrics(np.concatenate(truths), np.concatenate(predictions))
    result["loss"] = total_loss / total
    return result


def plot_history(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="treino")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validação")
    axes[0].set(title="Perda", xlabel="Época", ylabel="Cross-entropy")
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="treino")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="validação")
    axes[1].set(title="Acurácia", xlabel="Época", ylabel="Acurácia")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Treina e avalia os experimentos da APS CNN.")
    parser.add_argument("--dataset", choices=("mnist", "cifar10"), required=True)
    parser.add_argument("--mode", choices=("baseline", "ablation", "pretrained"), default="baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    # O ambiente de execução pode bloquear multiprocessing; 0 funciona em CPU e Windows/Linux.
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="Continua do melhor checkpoint do mesmo dataset/mode.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limita exemplos de treino e teste para uma execução curta de verificação.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or (args.limit is not None and args.limit < 2):
        raise SystemExit("epochs/batch-size devem ser positivos e limit deve ser >= 2")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = args.mode == "pretrained"
    run_dir = args.output_root / args.dataset / args.mode
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(
        args.dataset, args.data_root, args.batch_size, args.seed, args.workers,
        pretrained=pretrained, limit=args.limit,
    )
    if pretrained:
        weights_file = args.data_root / "weights" / "resnet18-f37072fd.pth"
        model = make_pretrained_resnet18(weights_path=weights_file)
        lr = args.lr or 1e-4
        optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr,
                          weight_decay=1e-4)
    else:
        channels = 1 if args.dataset == "mnist" else 3
        model = SmallCNN(channels, dropout=args.mode != "ablation")
        lr = args.lr or 1e-3
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    history, best_accuracy, prior_duration = [], -1.0, 0.0
    checkpoint_path = run_dir / "best_model.pt"
    metrics_path = run_dir / "metrics.json"
    if args.resume:
        if not checkpoint_path.exists() or not metrics_path.exists():
            raise SystemExit("--resume requer best_model.pt e metrics.json da execução anterior")
        previous = json.loads(metrics_path.read_text())
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        history = previous["history"]
        best_accuracy = previous["best_validation_accuracy"]
        prior_duration = previous.get("duration_seconds", 0.0)
        if len(history) >= args.epochs:
            raise SystemExit("--epochs precisa ser maior que a quantidade de épocas já registrada")

    started = time.time()
    for epoch in range(len(history) + 1, args.epochs + 1):
        train = epoch_pass(model, train_loader, loss_fn, device, optimizer)
        val = epoch_pass(model, val_loader, loss_fn, device)
        row = {"epoch": epoch, "train_loss": train["loss"],
               "train_accuracy": train["accuracy"], "val_loss": val["loss"],
               "val_accuracy": val["accuracy"]}
        history.append(row)
        print(f"época {epoch:02d}/{args.epochs} | treino acc {train['accuracy']:.4f}"
              f" | validação acc {val['accuracy']:.4f}", flush=True)
        if val["accuracy"] > best_accuracy:
            best_accuracy = val["accuracy"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device,
                                     weights_only=True))
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    result = {
        "dataset": args.dataset, "mode": args.mode,
        "component": "dropout" if args.mode in {"baseline", "ablation"} else "ImageNet pretrained ResNet18",
        "transfer_strategy": "frozen feature extractor + trained linear head" if pretrained else None,
        "pretrained_input_size": 64 if pretrained else None,
        "resumed": args.resume,
        "dropout_enabled": args.mode == "baseline",
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": lr, "device": str(device), "limit": args.limit,
        "best_validation_accuracy": best_accuracy,
        "test": test_metrics, "history": history,
        "duration_seconds": round(prior_duration + time.time() - started, 2),
    }
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    plot_history(history, run_dir / "training.png")
    print(json.dumps({"run_dir": str(run_dir), "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
