"""Conta as imagens originais de cada classe."""

from pathlib import Path
from features import listar_imagens

RAIZ = Path(__file__).resolve().parents[1] / "dataset" / "original"
CLASSES = ("cedula_2", "cedula_5", "cedula_10")
META = 200

total = 0
for classe in CLASSES:
    quantidade = len(listar_imagens(RAIZ / classe))
    total += quantidade
    situacao = "OK" if quantidade >= META else f"faltam {META - quantidade}"
    print(f"{classe:10s}: {quantidade:4d} ({situacao})")
print(f"Total: {total}")
