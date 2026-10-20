"""Classifica uma imagem individual com o modelo treinado."""

import argparse
from pathlib import Path

import joblib

from features import extrair_caracteristicas

RAIZ_PROJETO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classifica uma foto de cédula.")
    parser.add_argument("imagem", type=Path)
    parser.add_argument("--modelo", type=Path, default=RAIZ_PROJETO / "output" / "modelo" / "moneyvision_rgb_hsv_svm.joblib")
    args = parser.parse_args()
    pacote = joblib.load(args.modelo)
    vetor = extrair_caracteristicas(args.imagem).reshape(1, -1)
    probabilidades = pacote["modelo"].predict_proba(vetor)[0]
    ranking = sorted(zip(pacote["modelo"].classes_, probabilidades), key=lambda item: item[1], reverse=True)
    print(f"Imagem: {args.imagem.resolve()}")
    print(f"Previsão: {pacote['nomes'][ranking[0][0]]}")
    for classe, probabilidade in ranking:
        print(f"  {pacote['nomes'][classe]:5s}: {probabilidade * 100:6.2f}%")


if __name__ == "__main__":
    main()
