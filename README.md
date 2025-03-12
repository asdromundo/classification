# Classification - Comparación de Modelos de IA para Clasificación de Imágenes

Este proyecto tiene como objetivo comparar el desempeño de distintos enfoques para la clasificación de imágenes, incluyendo redes neuronales convolucionales (CNNs), Transformers para visión, YOLO y Few-Shot Learning. Se evalúan métricas de desempeño como precisión, recall y F1-score en distintos datasets.

## 📂 Estructura del Proyecto

```
classification/
│── data/ # Almacena los datasets 
│ ├── raw/ # Datos originales sin procesar
│ ├── processed/ # Datos transformados/listos para entrenamiento
│ ├── test/ # Datasets de prueba 
│ └── train/ # Datasets de entrenamiento 
│── models/ # Modelos entrenados organizados por tipo 
│ ├── cnn/ # Modelos CNN (ej. ResNet, EfficientNet)
│ ├── transformer/ # Modelos basados en Transformers (ej. ViT, DeiT)
│ ├── yolo/ # Modelos basados en YOLO
│ ├── few_shot/ # Modelos Few-Shot Learning (ej. Prototypical Networks)
│ └── checkpoints/ # Checkpoints de entrenamiento
|── docs/ # Documentos relacionados que no se deben subir
│── notebooks/ # Jupyter notebooks organizados por experimentos
│── scripts/ # Scripts reutilizables (entrenamiento, evaluación, etc.)
│── results/ # Reportes y métricas de evaluación 
|── README.md # Este archivo
│── .gitignore # Archivos a ignorar en Git
└── pyproject.toml # Configuración de dependencias con uv
```

---

## 🚀 Instalación del Entorno

Este proyecto usa [`uv`](https://github.com/astral-sh/uv) como gestor de paquetes para facilitar la instalación de dependencias. Se puede configurar el entorno para ejecutarse en **CPU** o con **CUDA 12.6** para aceleración con GPU.

### 1️⃣ Instalar `uv`
Si no tienes `uv` instalado, puedes seguir la [Guía Rápida](https://docs.astral.sh/uv/getting-started/installation/).