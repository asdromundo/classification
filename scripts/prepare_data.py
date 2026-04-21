"""
prepare_data.py
===============
Genera los splits de datos para el experimento de clasificación de osteosarcoma.

Estructura de salida:
    data/splits/
    ├── test/                  ← 20% fijo, NUNCA toca entrenamiento
    │   ├── Non-Tumor/
    │   ├── Viable-Tumor/
    │   └── Necrotic-Tumor/
    ├── full/                  ← 100% del pool de train
    │   ├── train/             ← 80% del pool
    │   └── val/               ← 20% del pool
    ├── subset_25/             ← 25% del pool (muestra independiente)
    │   ├── train/
    │   └── val/
    └── subset_10/             ← 10% del pool (muestra independiente)
        ├── train/
        └── val/

Uso:
    python prepare_data.py --src ../data/osteosarcoma2019 --dst ../data/splits --seed 42

El src debe tener la estructura de ImageFolder:
    osteosarcoma2019/
    ├── train/
    │   ├── Non-Tumor/
    │   ├── Viable-Tumor/
    │   └── Necrotic-Tumor/
    └── test/         ← se ignora, reconstruimos todo desde cero
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict


CLASSES = ["Non-Tumor", "Viable-Tumor", "Necrotic-Tumor"]
TRAIN_FRACTIONS = {"full": 1.0, "subset_25": 0.25, "subset_10": 0.10, "subset_05": 0.05}
TEST_RATIO = 0.20      # 20% del total va a test fijo
VAL_RATIO = 0.20       # 20% del pool restante va a val en cada experimento


def collect_images(src_dir: Path) -> dict[str, list[Path]]:
    """
    Recopila todas las imágenes del directorio fuente agrupadas por clase.
    Acepta tanto estructura flat (clase/) como train/clase/.
    """
    images = defaultdict(list)
    for cls in CLASSES:
        # Buscar en subdirectorios comunes
        for subdir in ["train", "test", "val", ""]:
            search_path = src_dir / subdir / cls if subdir else src_dir / cls
            if search_path.exists():
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
                    images[cls].extend(search_path.glob(ext))

    # Eliminar duplicados manteniendo paths únicos
    images = {cls: list(set(paths)) for cls, paths in images.items()}

    for cls, paths in images.items():
        print(f"  {cls}: {len(paths)} imágenes encontradas")

    return images


def stratified_split(images: dict[str, list[Path]], ratio: float, seed: int) -> tuple[dict, dict]:
    """
    Divide cada clase en dos grupos manteniendo proporciones.
    Retorna (grupo_a, grupo_b) donde grupo_a tiene 'ratio' fracción.
    """
    rng = random.Random(seed)
    group_a = {}
    group_b = {}

    for cls, paths in images.items():
        shuffled = paths.copy()
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * ratio)
        group_a[cls] = shuffled[:split_idx]
        group_b[cls] = shuffled[split_idx:]

    return group_a, group_b


def copy_split(images: dict[str, list[Path]], dst: Path, split_name: str) -> None:
    """Copia imágenes al directorio destino manteniendo estructura de clases."""
    total = 0
    for cls, paths in images.items():
        cls_dir = dst / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for src_path in paths:
            shutil.copy2(src_path, cls_dir / src_path.name)
            total += 1
    print(f"    → {split_name}: {total} imágenes copiadas")
    for cls, paths in images.items():
        print(f"       {cls}: {len(paths)}")


def build_splits(src_dir: Path, dst_dir: Path, seed: int) -> None:
    print("\n[1] Recopilando imágenes...")
    all_images = collect_images(src_dir)

    total = sum(len(v) for v in all_images.values())
    print(f"\n  Total: {total} imágenes en {len(all_images)} clases")

    # ── Paso 1: separar test set fijo ────────────────────────────────────────
    print(f"\n[2] Separando test set fijo ({TEST_RATIO*100:.0f}% del total)...")
    pool, test_images = stratified_split(all_images, 1 - TEST_RATIO, seed=seed)

    test_dir = dst_dir / "test"
    copy_split(test_images, test_dir, "test")

    pool_total = sum(len(v) for v in pool.values())
    test_total = sum(len(v) for v in test_images.values())
    print(f"\n  Pool de entrenamiento: {pool_total} | Test fijo: {test_total}")

    # ── Paso 2: generar variantes de entrenamiento ────────────────────────────
    print("\n[3] Generando subsets de entrenamiento...")

    for name, fraction in TRAIN_FRACTIONS.items():
        print(f"\n  [{name}] fracción={fraction}")

        # Samplear independientemente del pool completo
        # Usamos seed diferente por subset para independencia
        sub_seed = seed + hash(name) % 1000
        subset, _ = stratified_split(pool, fraction, seed=sub_seed)

        # Dividir en train/val
        train_images, val_images = stratified_split(subset, 1 - VAL_RATIO, seed=sub_seed + 1)

        exp_dir = dst_dir / name
        copy_split(train_images, exp_dir / "train", f"{name}/train")
        copy_split(val_images,   exp_dir / "val",   f"{name}/val")

    # ── Resumen final ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESUMEN DE SPLITS")
    print("="*60)
    for split_dir in sorted(dst_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        print(f"\n{split_dir.name}/")
        for sub in sorted(split_dir.iterdir()):
            if sub.is_dir():
                n = sum(1 for cls in sub.iterdir() if cls.is_dir()
                        for _ in cls.glob("*") if _.is_file())
                print(f"  {sub.name}/: {n} imágenes")
            else:
                # Es directamente clase (caso test/)
                n = sum(1 for _ in sub.glob("*") if _.is_file())
                if n:
                    print(f"  {sub.name}/: {n} imágenes")

    print("\n✓ Listo. Test set fijo guardado en:", dst_dir / "test")
    print("  Siempre evalúa sobre ese directorio. Nunca tocar durante train/val.")


def main():
    parser = argparse.ArgumentParser(description="Prepara splits de datos para experimentos few-shot")
    parser.add_argument("--src", type=str, default="../data/osteosarcoma2019",
                        help="Directorio fuente con imágenes organizadas por clase")
    parser.add_argument("--dst", type=str, default="../data/splits",
                        help="Directorio destino para los splits generados")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria para reproducibilidad")
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio fuente: {src_dir}")

    if dst_dir.exists():
        print(f"El directorio destino ya existe: {dst_dir}")
        resp = input("  ¿Sobreescribir? [s/N]: ").strip().lower()
        if resp != "s":
            print("Abortado.")
            return
        shutil.rmtree(dst_dir)

    dst_dir.mkdir(parents=True)

    print(f"Fuente : {src_dir.resolve()}")
    print(f"Destino: {dst_dir.resolve()}")
    print(f"Semilla: {args.seed}")

    build_splits(src_dir, dst_dir, seed=args.seed)


if __name__ == "__main__":
    main()