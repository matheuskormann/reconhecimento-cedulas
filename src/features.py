"""Extração de características visuais usada pelo MoneyVision."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2hsv

IMAGE_SIZE = (192, 96)
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def listar_imagens(pasta: Path) -> list[Path]:
    return sorted(
        arquivo
        for arquivo in pasta.rglob("*")
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES
    )


def normalizar_imagem(imagem: Image.Image) -> Image.Image:
    return ImageOps.pad(
        imagem.convert("RGB"),
        IMAGE_SIZE,
        method=Image.Resampling.LANCZOS,
        color=(20, 20, 20),
        centering=(0.5, 0.5),
    )


def extrair_caracteristicas(imagem_ou_caminho: Image.Image | str | Path) -> np.ndarray:
    """Representa as cores da imagem com histogramas RGB e HSV."""
    if isinstance(imagem_ou_caminho, Image.Image):
        imagem = imagem_ou_caminho
    else:
        with Image.open(imagem_ou_caminho) as arquivo:
            imagem = arquivo.copy()

    rgb = np.asarray(normalizar_imagem(imagem), dtype=np.float32) / 255.0
    hsv = rgb2hsv(rgb)
    histogramas_rgb = []
    for canal in range(3):
        histograma, _ = np.histogram(rgb[:, :, canal], bins=16, range=(0.0, 1.0))
        histograma = histograma.astype(np.float32)
        histograma /= max(histograma.sum(), 1.0)
        histogramas_rgb.append(histograma)

    histogramas_hsv = []
    for canal, bins in ((0, 24), (1, 16), (2, 16)):
        histograma, _ = np.histogram(hsv[:, :, canal], bins=bins, range=(0.0, 1.0))
        histograma = histograma.astype(np.float32)
        histograma /= max(histograma.sum(), 1.0)
        histogramas_hsv.append(histograma)

    return np.concatenate([*histogramas_rgb, *histogramas_hsv])
