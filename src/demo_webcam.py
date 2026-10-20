"""Demonstração em tempo real do MoneyVision com a webcam."""

import argparse
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image

from features import extrair_caracteristicas

RAIZ_PROJETO = Path(__file__).resolve().parents[1]


def ajustar_para_16_9(quadro: np.ndarray) -> np.ndarray:
    """Centraliza e recorta o quadro para a proporção usada no treinamento."""
    altura, largura = quadro.shape[:2]
    proporcao_alvo = 16 / 9
    proporcao_atual = largura / altura
    if proporcao_atual > proporcao_alvo:
        nova_largura = int(altura * proporcao_alvo)
        inicio = (largura - nova_largura) // 2
        return quadro[:, inicio : inicio + nova_largura]
    if proporcao_atual < proporcao_alvo:
        nova_altura = int(largura / proporcao_alvo)
        inicio = (altura - nova_altura) // 2
        return quadro[inicio : inicio + nova_altura, :]
    return quadro


def main() -> None:
    parser = argparse.ArgumentParser(description="Classifica cédulas em tempo real.")
    parser.add_argument(
        "--modelo",
        type=Path,
        default=RAIZ_PROJETO / "output" / "modelo" / "moneyvision_rgb_hsv_svm.joblib",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--janela",
        type=int,
        default=3,
        help="Quantidade de quadros recentes usados na média.",
    )
    parser.add_argument("--limiar", type=float, default=0.50)
    parser.add_argument("--margem", type=float, default=0.08)
    parser.add_argument("--tempo-analise", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--tempo-resultado", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--confirmacoes", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()

    pacote = joblib.load(args.modelo)
    modelo = pacote["modelo"]
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit("Não foi possível abrir a webcam. Tente --camera 1.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    historico = deque(maxlen=max(1, args.janela))
    previsao = "--"
    confianca = 0.0

    try:
        while True:
            ok, quadro = camera.read()
            if not ok:
                break
            quadro = ajustar_para_16_9(quadro)
            altura, largura = quadro.shape[:2]

            rgb = cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)
            vetor = extrair_caracteristicas(Image.fromarray(rgb)).reshape(1, -1)
            historico.append(modelo.predict_proba(vetor)[0])
            medias = np.mean(historico, axis=0)
            ordem = np.argsort(medias)[::-1]
            melhor_indice = int(ordem[0])
            confianca_atual = float(medias[melhor_indice])
            margem = confianca_atual - float(medias[ordem[1]])

            if confianca_atual >= args.limiar and margem >= args.margem:
                classe = modelo.classes_[melhor_indice]
                previsao = pacote["nomes"][classe]
                confianca = confianca_atual

            cor = (40, 210, 70)
            x1, y1 = int(largura * 0.03), int(altura * 0.05)
            x2, y2 = int(largura * 0.97), int(altura * 0.95)
            cv2.rectangle(quadro, (x1, y1), (x2, y2), cor, 3)
            cv2.rectangle(quadro, (0, 0), (largura, 82), (20, 20, 20), -1)
            cv2.putText(
                quadro,
                f"Previsao: {previsao}",
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                quadro,
                f"Confianca: {confianca * 100:.1f}% | Tempo real | Q para sair",
                (18, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                cor,
                2,
            )
            cv2.imshow("MoneyVision - cedulas brasileiras", quadro)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
