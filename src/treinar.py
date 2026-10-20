"""Treina, seleciona e avalia o classificador local RGB/HSV + SVM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import IMAGE_SIZE, extrair_caracteristicas, listar_imagens

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
CLASSES = ("cedula_2", "cedula_5", "cedula_10")
NOMES = {"cedula_2": "R$ 2", "cedula_5": "R$ 5", "cedula_10": "R$ 10"}
VALORES_C = (0.1, 1.0, 10.0, 100.0)


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina e avalia o MoneyVision local.")
    parser.add_argument("--dataset", type=Path, default=RAIZ_PROJETO / "dataset" / "processado")
    parser.add_argument("--saida", type=Path, default=RAIZ_PROJETO / "output")
    return parser.parse_args()


def carregar_divisao(raiz: Path, divisao: str) -> tuple[np.ndarray, np.ndarray]:
    caracteristicas: list[np.ndarray] = []
    rotulos: list[str] = []
    for classe in CLASSES:
        arquivos = listar_imagens(raiz / divisao / classe)
        if not arquivos:
            raise SystemExit(f"Nenhuma imagem em {raiz / divisao / classe}")
        for arquivo in arquivos:
            caracteristicas.append(extrair_caracteristicas(arquivo))
            rotulos.append(classe)
        print(f"{divisao:10s} | {classe:10s}: {len(arquivos):4d}")
    return np.stack(caracteristicas), np.asarray(rotulos)


def criar_modelo(c: float):
    return make_pipeline(
        StandardScaler(),
        SVC(C=c, kernel="rbf", gamma="scale", class_weight="balanced", random_state=42),
    )


def salvar_matriz(y_real: np.ndarray, y_pred: np.ndarray, destino: Path) -> np.ndarray:
    matriz = confusion_matrix(y_real, y_pred, labels=CLASSES)
    figura, eixo = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matriz, display_labels=[NOMES[c] for c in CLASSES]).plot(ax=eixo, cmap="Blues", colorbar=False)
    eixo.set_title("Matriz de confusão — conjunto de teste")
    figura.tight_layout()
    figura.savefig(destino, dpi=180)
    plt.close(figura)
    return matriz


def main() -> None:
    args = argumentos()
    pasta_modelo = args.saida / "modelo"
    pasta_metricas = args.saida / "metricas"
    pasta_figuras = args.saida / "figuras"
    for pasta in (pasta_modelo, pasta_metricas, pasta_figuras):
        pasta.mkdir(parents=True, exist_ok=True)

    x_treino, y_treino = carregar_divisao(args.dataset, "treino")
    x_validacao, y_validacao = carregar_divisao(args.dataset, "validacao")
    x_teste, y_teste = carregar_divisao(args.dataset, "teste")

    resultados_validacao = []
    for c in VALORES_C:
        candidato = criar_modelo(c)
        candidato.fit(x_treino, y_treino)
        previsoes = candidato.predict(x_validacao)
        resultado = {"C": c, "acuracia": float(accuracy_score(y_validacao, previsoes)), "acuracia_balanceada": float(balanced_accuracy_score(y_validacao, previsoes))}
        resultados_validacao.append(resultado)
        print(f"Validação | C={c:g}: acurácia={resultado['acuracia']:.3f}, balanceada={resultado['acuracia_balanceada']:.3f}")

    melhor = max(resultados_validacao, key=lambda item: (item["acuracia_balanceada"], item["acuracia"], -item["C"]))
    melhor_c = float(melhor["C"])
    x_desenvolvimento = np.concatenate([x_treino, x_validacao])
    y_desenvolvimento = np.concatenate([y_treino, y_validacao])
    modelo = CalibratedClassifierCV(
        criar_modelo(melhor_c), method="sigmoid", cv=5, ensemble=False
    )
    modelo.fit(x_desenvolvimento, y_desenvolvimento)
    y_pred = modelo.predict(x_teste)

    relatorio = classification_report(y_teste, y_pred, labels=CLASSES, target_names=[NOMES[c] for c in CLASSES], output_dict=True, zero_division=0)
    acuracia = float(accuracy_score(y_teste, y_pred))
    acuracia_balanceada = float(balanced_accuracy_score(y_teste, y_pred))
    matriz = salvar_matriz(y_teste, y_pred, pasta_figuras / "matriz_confusao.png")
    pacote = {"modelo": modelo, "classes": list(CLASSES), "nomes": NOMES, "image_size": IMAGE_SIZE, "metodo": "Histogramas RGB e HSV + SVM RBF"}
    joblib.dump(pacote, pasta_modelo / "moneyvision_rgb_hsv_svm.joblib")

    with (pasta_metricas / "resultados_validacao.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=["C", "acuracia", "acuracia_balanceada"])
        escritor.writeheader()
        escritor.writerows(resultados_validacao)

    resultado_final = {
        "metodo": pacote["metodo"], "melhor_C": melhor_c,
        "quantidades": {"treino": len(y_treino), "validacao": len(y_validacao), "teste": len(y_teste)},
        "acuracia_teste": acuracia, "acuracia_balanceada_teste": acuracia_balanceada,
        "matriz_confusao": matriz.tolist(), "ordem_classes_matriz": [NOMES[c] for c in CLASSES],
        "relatorio_classificacao": relatorio,
    }
    with (pasta_metricas / "resultados_teste.json").open("w", encoding="utf-8") as arquivo:
        json.dump(resultado_final, arquivo, ensure_ascii=False, indent=2)

    print(f"\nMelhor C pela validação: {melhor_c:g}")
    print(f"Acurácia no teste: {acuracia:.3f}")
    print(f"Acurácia balanceada no teste: {acuracia_balanceada:.3f}")
    print("\n" + classification_report(y_teste, y_pred, labels=CLASSES, target_names=[NOMES[c] for c in CLASSES], zero_division=0))
    print(f"Modelo e resultados salvos em: {args.saida.resolve()}")


if __name__ == "__main__":
    main()
