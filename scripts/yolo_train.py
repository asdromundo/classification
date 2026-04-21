"""
yolo_train.py
=============
Wrapper de entrenamiento YOLO-cls (Ultralytics) que produce el mismo
formato de resultados JSON que train.py, para integrarse al flujo de
Experiments.ipynb sin cambios.

Uso:
    python scripts/yolo_train.py --split full      --model yolo26n-cls --epochs 100
    python scripts/yolo_train.py --split subset_25 --model yolo26s-cls --epochs 100
    python scripts/yolo_train.py --split subset_10 --model yolo26n-cls --epochs 100

Modelos disponibles (clasificación):
    yolo26n-cls  ← nano,  el más rápido
    yolo26s-cls  ← small
    yolo26m-cls  ← medium
    yolo26l-cls  ← large
    yolo26x-cls  ← extra-large

Nota sobre la API de Ultralytics para clasificación:
    A partir de versiones recientes, YOLO-cls NO acepta YAML como dataset.
    Espera directamente un directorio con estructura:
        dataset/
        ├── train/
        │   ├── clase_1/
        │   └── clase_2/
        ├── val/
        └── test/
    Como nuestro test set fijo está fuera del split directory, creamos
    un directorio temporal con symlinks para unificar todo.

Salida:
    models/yolo/<model>/<split>/   ← checkpoints y logs de Ultralytics
    results/yolo_<model>_<split>.json  ← misma estructura que train.py
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

torch.set_float32_matmul_precision("high")

# ─── Constantes ──────────────────────────────────────────────────────────────

CLASSES = ["Necrotic-Tumor", "Non-Tumor", "Viable-Tumor"]   # orden alfabético
NUM_CLASSES = 3

YOLO_CLS_MODELS = [
    "yolo26n-cls",
    "yolo26s-cls",
    "yolo26m-cls",
    "yolo26l-cls",
    "yolo26x-cls",
]


# ─── Directorio temporal con symlinks ────────────────────────────────────────

def make_dataset_dir(splits_dir: Path, split: str, tmp_dir: Path) -> Path:
    """
    Crea un directorio temporal con la estructura que espera Ultralytics:
        tmp/
        ├── train -> splits/<split>/train   (symlink)
        ├── val   -> splits/<split>/val     (symlink)
        └── test  -> splits/test            (symlink)

    Usar symlinks evita copiar las imágenes y mantiene el test set fijo
    completamente separado del split de entrenamiento.
    """
    dataset_dir = tmp_dir / f"dataset_{split}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    links = {
        "train": splits_dir / split / "train",
        "val":   splits_dir / split / "val",
        "test":  splits_dir / "test",
    }

    for name, target in links.items():
        link_path = dataset_dir / name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        os.symlink(target.resolve(), link_path)

    print(f"\nDataset dir: {dataset_dir}")
    for name, target in links.items():
        print(f"  {name}/ → {target.resolve()}")

    return dataset_dir


# ─── Extracción de métricas de Ultralytics ────────────────────────────────────

def extract_metrics(val_results) -> dict:
    """
    Extrae accuracy, F1, precision, recall y confusion matrix
    de los resultados de model.val() de Ultralytics.
    """
    rd = val_results.results_dict if hasattr(val_results, "results_dict") else {}

    top1 = float(rd.get("metrics/accuracy_top1", 0.0))

    # Confusion matrix
    cm = None
    if hasattr(val_results, "confusion_matrix") and val_results.confusion_matrix is not None:
        cm_obj = val_results.confusion_matrix
        if hasattr(cm_obj, "matrix"):
            arr = np.array(cm_obj.matrix)
            # Recortar fila/col de background si existe (herencia del modo detección)
            if arr.shape == (NUM_CLASSES + 1, NUM_CLASSES + 1):
                arr = arr[:NUM_CLASSES, :NUM_CLASSES]
            cm = arr.astype(int).tolist()

    # F1, precision y recall macro desde la confusion matrix
    f1_macro = precision_macro = recall_macro = None
    if cm is not None:
        arr = np.array(cm)
        precisions, recalls, f1s = [], [], []
        for i in range(NUM_CLASSES):
            tp = arr[i, i]
            fp = arr[:, i].sum() - tp
            fn = arr[i, :].sum() - tp
            p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
        f1_macro        = float(np.mean(f1s))
        precision_macro = float(np.mean(precisions))
        recall_macro    = float(np.mean(recalls))

    return {
        "test_acc":       top1,
        "test_f1":        f1_macro,
        "test_precision": precision_macro,
        "test_recall":    recall_macro,
        "confusion_matrix": cm,
    }


# ─── Entrenamiento + Evaluación ───────────────────────────────────────────────

def run(args):
    splits_dir  = Path(args.splits_dir)
    results_dir = Path(args.results_dir)
    ckpt_dir    = Path(args.ckpt_dir) / "yolo" / args.model / args.split
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Modelo : YOLO {args.model}")
    print(f"  Split  : {args.split}")
    print(f"  Epochs : {args.epochs}")
    print(f"{'='*60}\n")

    tmp_dir = Path(tempfile.mkdtemp())

    try:
        # ── Crear directorio de dataset con symlinks ──
        dataset_dir = make_dataset_dir(splits_dir, args.split, tmp_dir)

        # ── Cargar modelo base ──
        # Ultralytics descarga los pesos al directorio de trabajo actual.
        # Para controlarlo cambiamos cwd temporalmente a nuestra carpeta
        # de pesos antes de instanciar YOLO.
        weights_dir = Path(args.ckpt_dir) / "yolo" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / f"{args.model}.pt"

        original_cwd = Path.cwd()
        try:
            os.chdir(weights_dir)
            model = YOLO(f"{args.model}.pt")  # descarga aquí si no existe
        finally:
            os.chdir(original_cwd)

        # ── Entrenamiento ──
        model.train(
            data      = str(dataset_dir),
            epochs    = args.epochs,
            imgsz     = args.imgsz,
            batch     = args.batch_size,
            patience  = args.patience,
            project   = str(ckpt_dir),
            name      = "train",
            exist_ok  = True,
            device    = 0 if torch.cuda.is_available() else "cpu",
            seed      = 42,
            plots     = True,
            save      = True,
            verbose   = True,
        )

        best_ckpt = ckpt_dir / "train" / "weights" / "best.pt"
        print(f"\nMejor checkpoint: {best_ckpt}")

        # ── Evaluación sobre test set fijo ──
        print("\nEvaluando sobre test set fijo...")
        eval_model = YOLO(str(best_ckpt))

        val_results = eval_model.val(
            data     = str(dataset_dir),
            split    = "test",
            imgsz    = args.imgsz,
            batch    = args.batch_size,
            device   = 0 if torch.cuda.is_available() else "cpu",
            plots    = True,
            project  = str(ckpt_dir),
            name     = "test_eval",
            exist_ok = True,
            verbose  = True,
        )

        metrics = extract_metrics(val_results)

        # ── Guardar resultados en formato común ──
        result = {
            "model":          f"yolo_{args.model}",
            "split":          args.split,
            "pretrained":     True,
            "epochs_run":     args.epochs,
            "best_val_acc":   None,
            "test_acc":       metrics["test_acc"],
            "test_f1":        metrics["test_f1"],
            "test_precision": metrics["test_precision"],
            "test_recall":    metrics["test_recall"],
            "test_loss":      None,
            "confusion_matrix": metrics["confusion_matrix"],
            "classes":        CLASSES,
            "checkpoint":     str(best_ckpt),
        }

        result_path = results_dir / f"yolo_{args.model}_{args.split}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n✓ Resultados guardados en: {result_path}")
        print(f"  test_acc       : {result['test_acc']:.4f}")
        if result["test_f1"] is not None:
            print(f"  test_f1        : {result['test_f1']:.4f}")
            print(f"  test_precision : {result['test_precision']:.4f}")
            print(f"  test_recall    : {result['test_recall']:.4f}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Entrena YOLO-cls y evalúa en test set fijo (mismo formato que train.py)"
    )
    parser.add_argument(
        "--split", required=True,
        choices=["full", "subset_25", "subset_10", "subset_05"],
    )
    parser.add_argument(
        "--model", default="yolo26s-cls",
        choices=YOLO_CLS_MODELS,
    )
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--imgsz",      type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience",   type=int, default=20)
    parser.add_argument("--splits_dir",  default="../data/splits")
    parser.add_argument("--results_dir", default="../results")
    parser.add_argument("--ckpt_dir",    default="../models")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()