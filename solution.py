import numpy as np
import pandas as pd

def solution(arquivo):
    df = pd.read_csv(arquivo)

    area = np.array(df["area_util_m2"].to_list(), dtype=float)
    quartos = np.array(df["quantidade_quartos"].to_list(), dtype=float)
    preco = np.array(df["preco_venda"].to_list(), dtype=float)

    if len(df) == 0:
        raise ValueError("O arquivo CSV está vazio.")

    if not np.isfinite(area).all() or not np.isfinite(quartos).all() or not np.isfinite(preco).all():
        raise ValueError("Existem valores inválidos nos dados.")

    X = np.column_stack((
        np.ones(len(df)),
        area,
        quartos
    ))

    beta, _, _, _ = np.linalg.lstsq(X, preco, rcond=None)

    return beta.tolist()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("arquivo", help="Caminho do arquivo CSV")

    args = parser.parse_args()

    b0, b1, b2 = solution(args.arquivo)

    print("Coeficientes encontrados:")
    print([b0, b1, b2])

    print("\nEquação da regressão linear:")
    print(f"Preco = {b0:.6f} + ({b1:.6f}) * Area + ({b2:.6f}) * Quartos")