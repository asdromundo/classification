"""
train.py
========
Entrena un modelo sobre un split específico y evalúa sobre el test set fijo.

Uso:
    python scripts/train.py --split full --model resnet50 --epochs 50
    python scripts/train.py --split subset_25 --model vit_b_16 --epochs 100 --pretrained
    python scripts/train.py --split subset_10 --model efficientnet_v2_s --epochs 100 --pretrained

Splits disponibles:   full | subset_25 | subset_10
Modelos disponibles:  ver MODEL_REGISTRY abajo

Estructura esperada en data/splits/:
    splits/
    ├── test/           ← fijo, nunca cambia
    ├── full/
    │   ├── train/
    │   └── val/
    ├── subset_25/
    │   ├── train/
    │   └── val/
    └── subset_10/
        ├── train/
        └── val/

Salida:
    models/<model>/<split>/
    ├── best.ckpt           ← mejor checkpoint según val_acc
    └── hparams.yaml

    results/<model>_<split>.json  ← métricas finales sobre test set fijo
"""

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torchmetrics import Accuracy, F1Score, ConfusionMatrix

torch.set_float32_matmul_precision('high')


# ─── Registro de modelos ────────────────────────────────────────────────────
# Cada entrada: nombre_clave → (función_constructor, atributo_del_head, nombre_weights)
#
# atributo_del_head puede ser:
#   "classifier"  → modelo.classifier o modelo.classifier[-1]   (AlexNet, VGG, DenseNet, EfficientNet)
#   "fc"          → modelo.fc                                    (ResNet)
#   "heads"       → modelo.heads.head                           (ViT)

MODEL_REGISTRY = {
    # AlexNet
    "alexnet":            ("alexnet",            "AlexNet_Weights",              "classifier"),
    # DenseNet
    "densenet121":        ("densenet121",         "DenseNet121_Weights",          "classifier"),
    "densenet169":        ("densenet169",         "DenseNet169_Weights",          "classifier"),
    "densenet201":        ("densenet201",         "DenseNet201_Weights",          "classifier"),
    # EfficientNet
    "efficientnet_b0":    ("efficientnet_b0",     "EfficientNet_B0_Weights",      "classifier"),
    "efficientnet_b3":    ("efficientnet_b3",     "EfficientNet_B3_Weights",      "classifier"),
    "efficientnet_v2_s":  ("efficientnet_v2_s",   "EfficientNet_V2_S_Weights",    "classifier"),
    "efficientnet_v2_m":  ("efficientnet_v2_m",   "EfficientNet_V2_M_Weights",    "classifier"),
    # ResNet
    "resnet18":           ("resnet18",            "ResNet18_Weights",             "fc"),
    "resnet50":           ("resnet50",            "ResNet50_Weights",             "fc"),
    "resnet101":          ("resnet101",           "ResNet101_Weights",            "fc"),
    # VGG
    "vgg11_bn":           ("vgg11_bn",            "VGG11_BN_Weights",             "classifier"),
    "vgg16_bn":           ("vgg16_bn",            "VGG16_BN_Weights",             "classifier"),
    # ViT
    "vit_b_16":           ("vit_b_16",            "ViT_B_16_Weights",             "heads"),
    "vit_b_32":           ("vit_b_32",            "ViT_B_32_Weights",             "heads"),
    "vit_l_16":           ("vit_l_16",            "ViT_L_16_Weights",             "heads"),
}

NUM_CLASSES = 3
CLASSES = ["Necrotic-Tumor", "Non-Tumor", "Viable-Tumor"]   # orden de ImageFolder (alfabético)


# ─── Construcción del modelo ─────────────────────────────────────────────────

def build_model(model_key: str, pretrained: bool, num_classes: int) -> nn.Module:
    """
    Carga el backbone desde torchvision y reemplaza la capa de clasificación
    para adaptarlo a num_classes.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo '{model_key}' no encontrado.\n"
            f"Disponibles: {sorted(MODEL_REGISTRY.keys())}"
        )

    constructor_name, weights_name, head_type = MODEL_REGISTRY[model_key]
    constructor = getattr(tv_models, constructor_name)

    if pretrained:
        weights_class = getattr(tv_models, weights_name)
        model = constructor(weights=weights_class.DEFAULT)
    else:
        model = constructor(weights=None)

    # Reemplazar capa final
    if head_type == "fc":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif head_type == "heads":                          # ViT
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    elif head_type == "classifier":
        clf = model.classifier
        if isinstance(clf, nn.Sequential):
            # DenseNet, VGG, EfficientNet: reemplazar última capa linear
            for i in range(len(clf) - 1, -1, -1):
                if isinstance(clf[i], nn.Linear):
                    in_features = clf[i].in_features
                    clf[i] = nn.Linear(in_features, num_classes)
                    break
            model.classifier = clf
        else:
            in_features = clf.in_features
            model.classifier = nn.Linear(in_features, num_classes)

    return model


def get_pretrained_transforms(model_key: str):
    """
    Devuelve las transformaciones oficiales del modelo preentrenado.
    Si no hay pesos registrados, devuelve un transform genérico razonable.
    """
    _, weights_name, _ = MODEL_REGISTRY[model_key]
    try:
        weights_class = getattr(tv_models, weights_name)
        return weights_class.DEFAULT.transforms()
    except Exception:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ─── DataModule ──────────────────────────────────────────────────────────────

class OsteoDataModule(L.LightningDataModule):
    """
    Carga train y val desde el split elegido.
    El test set siempre viene de splits/test/ (fijo).
    """

    def __init__(self, splits_dir: str, split: str, transform, batch_size: int = 32):
        super().__init__()
        self.splits_dir = Path(splits_dir)
        self.split = split                  # "full" | "subset_25" | "subset_10"
        self.transform = transform
        self.batch_size = batch_size
        self.num_classes = NUM_CLASSES
        self.num_workers = min(os.cpu_count(), 8)

    def setup(self, stage=None):
        split_dir = self.splits_dir / self.split
        test_dir  = self.splits_dir / "test"

        if stage in ("fit", None):
            self.train_ds = ImageFolder(
                root=str(split_dir / "train"),
                transform=self.transform,
            )
            self.val_ds = ImageFolder(
                root=str(split_dir / "val"),
                transform=self.transform,
            )

        if stage in ("test", None):
            self.test_ds = ImageFolder(
                root=str(test_dir),
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


# ─── LightningModule ─────────────────────────────────────────────────────────

class LitClassifier(L.LightningModule):
    def __init__(self, model_key: str, pretrained: bool, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(model_key, pretrained, num_classes)
        self.criterion = nn.CrossEntropyLoss()

        # Métricas — val_acc es la que monitorea el checkpoint
        task = "multiclass"
        self.train_acc = Accuracy(task=task, num_classes=num_classes)
        self.val_acc   = Accuracy(task=task, num_classes=num_classes)
        self.test_acc  = Accuracy(task=task, num_classes=num_classes)
        self.test_f1   = F1Score(task=task, num_classes=num_classes, average="macro")
        self.test_cm   = ConfusionMatrix(task=task, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        return loss, preds, y

    def training_step(self, batch, _):
        loss, preds, y = self._shared_step(batch)
        self.train_acc(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc",  self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        loss, preds, y = self._shared_step(batch)
        self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc",  self.val_acc, prog_bar=True)

    def test_step(self, batch, _):
        loss, preds, y = self._shared_step(batch)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_cm(preds, y)
        self.log("test_loss", loss)
        self.log("test_acc",  self.test_acc)
        self.log("test_f1",   self.test_f1)

    def on_test_epoch_end(self):
        # Guarda la confusion matrix como atributo para leerla después del test
        self.confusion_matrix = self.test_cm.compute().cpu().tolist()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ─── Entrenamiento + Evaluación ──────────────────────────────────────────────

def run(args):
    L.seed_everything(42, workers=True)

    # Paths
    splits_dir  = Path(args.splits_dir)
    results_dir = Path(args.results_dir)
    ckpt_dir    = Path(args.ckpt_dir) / args.model / args.split
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Modelo : {args.model}")
    print(f"  Split  : {args.split}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Pretrained: {args.pretrained}")
    print(f"{'='*60}\n")

    # Transforms y DataModule
    transform   = get_pretrained_transforms(args.model)
    datamodule  = OsteoDataModule(splits_dir, args.split, transform, batch_size=args.batch_size)

    # Modelo
    lit_model = LitClassifier(
        model_key   = args.model,
        pretrained  = args.pretrained,
        num_classes = NUM_CLASSES,
        lr          = args.lr,
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath    = str(ckpt_dir),
        filename   = "best",
        monitor    = "val_acc",
        mode       = "max",
        save_top_k = 1,
        save_last  = False,
    )
    early_stop_cb = EarlyStopping(
        monitor  = "val_acc",
        patience = args.patience,
        mode     = "max",
        verbose  = True,
    )

    # Logger — guarda metrics.csv en models/<model>/<split>/
    csv_logger = CSVLogger(
        save_dir = str(ckpt_dir),
        name     = "",
        version  = "",
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs           = args.epochs,
        accelerator          = "auto",
        devices              = "auto",
        precision            = "16-mixed",
        callbacks            = [checkpoint_cb, early_stop_cb],
        logger               = csv_logger,
        enable_progress_bar  = True,
        benchmark            = True,
        deterministic        = False,
        log_every_n_steps    = 5,
    )

    # ── Entrenamiento ──
    trainer.fit(lit_model, datamodule=datamodule)
    print(f"\nMejor checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Mejor val_acc   : {checkpoint_cb.best_model_score:.4f}")

    # ── Evaluación sobre test set fijo ──
    print("\nEvaluando sobre test set fijo...")
    test_results = trainer.test(
        ckpt_path  = checkpoint_cb.best_model_path,
        datamodule = datamodule,
        verbose    = True,
    )

    # ── Guardar resultados ──
    result = {
        "model":      args.model,
        "split":      args.split,
        "pretrained": args.pretrained,
        "epochs_run": trainer.current_epoch,
        "best_val_acc": float(checkpoint_cb.best_model_score),
        "test_acc":   test_results[0].get("test_acc"),
        "test_f1":    test_results[0].get("test_f1"),
        "test_loss":  test_results[0].get("test_loss"),
        "confusion_matrix": lit_model.confusion_matrix,
        "classes":    CLASSES,
        "checkpoint": checkpoint_cb.best_model_path,
    }

    result_path = results_dir / f"{args.model}_{args.split}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Resultados guardados en: {result_path}")
    print(f"  test_acc : {result['test_acc']:.4f}")
    print(f"  test_f1  : {result['test_f1']:.4f}")

    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Entrena un modelo sobre un split y evalúa en test set fijo."
    )
    parser.add_argument(
        "--split", required=True,
        choices=["full", "subset_25", "subset_10"],
        help="Subconjunto de datos a usar para entrenamiento"
    )
    parser.add_argument(
        "--model", required=True,
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Arquitectura a entrenar"
    )
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=15,
                        help="Épocas sin mejora de val_acc antes de early stopping")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Usar pesos preentrenados en ImageNet (default: True)")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--splits_dir",  default="../data/splits")
    parser.add_argument("--results_dir", default="../results")
    parser.add_argument("--ckpt_dir",    default="../models")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()