"""Divide as imagens originais em treino, validação e teste."""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from pathlib import Path

from features import listar_imagens

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
CLASSES = ("cedula_2", "cedula_5", "cedula_10")
DIVISOES = ("treino", "validacao", "teste")


def remover_duplicadas(arquivos: list[Path]) -> tuple[list[Path], int]:
    """Mantém um caminho para cada conteúdo, sem alterar os arquivos originais."""
    unicas = []
    hashes = set()
    for arquivo in arquivos:
        digestor = hashlib.sha256()
        with arquivo.open("rb") as entrada:
            for bloco in iter(lambda: entrada.read(1024 * 1024), b""):
                digestor.update(bloco)
        assinatura = digestor.digest()
        if assinatura not in hashes:
            hashes.add(assinatura)
            unicas.append(arquivo)
    return unicas, len(arquivos) - len(unicas)


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cria uma divisão 70/15/15 reproduzível.")
    parser.add_argument("--originais", type=Path, default=RAIZ_PROJETO / "dataset" / "original")
    parser.add_argument("--destino", type=Path, default=RAIZ_PROJETO / "dataset" / "processado")
    parser.add_argument("--semente", type=int, default=42)
    parser.add_argument("--forcar", action="store_true", help="Refaz a divisão existente.")
    return parser.parse_args()


def main() -> None:
    args = argumentos()
    destino = args.destino.resolve()
    if destino.exists() and listar_imagens(destino):
        if not args.forcar:
            raise SystemExit(f"{destino} já contém imagens. Use --forcar para recriar.")
        shutil.rmtree(destino)

    imagens_por_classe = {}
    for classe in CLASSES:
        origem_classe = args.originais / classe
        imagens = listar_imagens(origem_classe) if origem_classe.exists() else []
        if len(imagens) < 7:
            raise SystemExit(
                f"A classe {classe} tem {len(imagens)} imagem(ns). "
                "Use pelo menos 7; o recomendado é 200."
            )
        imagens_unicas, duplicadas = remover_duplicadas(imagens)
        if len(imagens_unicas) < 7:
            raise SystemExit(
                f"A classe {classe} tem apenas {len(imagens_unicas)} fotos únicas."
            )
        imagens_por_classe[classe] = imagens_unicas
        print(
            f"{classe:10s} | arquivos: {len(imagens):4d} | "
            f"fotos únicas: {len(imagens_unicas):4d} | duplicadas ignoradas: {duplicadas:4d}"
        )

    quantidade_balanceada = max(len(imagens) for imagens in imagens_por_classe.values())
    alvo_treino = int(quantidade_balanceada * 0.70)
    alvo_validacao = int(quantidade_balanceada * 0.15)
    alvos = {
        "treino": alvo_treino,
        "validacao": alvo_validacao,
        "teste": quantidade_balanceada - alvo_treino - alvo_validacao,
    }
    print(
        f"Balanceamento por repetição: {quantidade_balanceada} imagens por classe "
        "(originais preservadas)\n"
    )

    totais = {nome: 0 for nome in DIVISOES}
    for classe in CLASSES:
        imagens = imagens_por_classe[classe].copy()
        random.Random(f"{args.semente}-{classe}").shuffle(imagens)
        n_treino = int(len(imagens) * 0.70)
        n_validacao = int(len(imagens) * 0.15)
        grupos = {
            "treino": imagens[:n_treino],
            "validacao": imagens[n_treino : n_treino + n_validacao],
            "teste": imagens[n_treino + n_validacao :],
        }
        for divisao, arquivos in grupos.items():
            rng = random.Random(f"{args.semente}-{classe}-{divisao}")
            arquivos_balanceados = arquivos.copy()
            while len(arquivos_balanceados) < alvos[divisao]:
                arquivos_balanceados.append(rng.choice(arquivos))
            rng.shuffle(arquivos_balanceados)

            pasta = destino / divisao / classe
            pasta.mkdir(parents=True, exist_ok=True)
            for indice, arquivo in enumerate(arquivos_balanceados, start=1):
                novo_nome = f"{classe}_{indice:04d}{arquivo.suffix.lower()}"
                shutil.copy2(arquivo, pasta / novo_nome)
            repetidas = len(arquivos_balanceados) - len(arquivos)
            totais[divisao] += len(arquivos_balanceados)
            print(
                f"{classe:10s} | {divisao:10s}: {len(arquivos_balanceados):4d} "
                f"({len(arquivos):3d} originais + {repetidas:3d} repetidas)"
            )
    print(f"\nDivisão concluída (semente = {args.semente})")
    print(" | ".join(f"{nome}: {total}" for nome, total in totais.items()))


if __name__ == "__main__":
    main()
