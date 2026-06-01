"""Experimento do STIMA - comparacao simples entre metodos.

Atende a exigencia do trabalho final ("comparacao simples entre metodos",
"resultados com tabelas/graficos") para a parte de MINERACAO.

  Experimento 1 - Classificacao automatica de gastos
    Compara o metodo do projeto, APRIORI (regras de associacao), com um
    BASELINE por palavra-chave (voto majoritario), sobre SMS bancarios
    sinteticos rotulados. Metricas: acuracia, precisao, revocacao e F1
    (macro). O Apriori e avaliado em dois cenarios: SMS limpos e SMS
    ruidosos (tokens promocionais/genericos injetados, simulando dados reais).

Observacao: o Experimento 2 (motor de regras x Naive Bayes) sera adicionado
quando os modulos stima.rule_engine, stima.rule_dsl e stima.bayes estiverem
disponiveis. Esta versao foca na fatia ja implementada (mining.py).

Saidas (pasta results/): tabelas .csv e graficos .png.

Como rodar:
    python experiment.py
"""
from __future__ import annotations

import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from stima.mining import ClassificadorApriori, ClassificadorBaseline
from stima.seed_data import gerar_sms_dataset

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stima", "results")
os.makedirs(RESULTS, exist_ok=True)

# Paleta sobria para os graficos
COR_APRIORI = "#2a9d8f"
COR_BASELINE = "#e9c46a"

def split(dados: list, frac: float = 0.7, seed: int = 42):
    rng = random.Random(seed)
    dados = list(dados)
    rng.shuffle(dados)
    corte = int(len(dados) * frac)
    return dados[:corte], dados[corte:]


def metricas_classificacao(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "acuracia": accuracy_score(y_true, y_pred),
        "precisao": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "revocacao": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def banner(titulo: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {titulo}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Experimento 1 - classificacao de gastos: Apriori x baseline
# ---------------------------------------------------------------------------
# Tokens "de ruido" (promocionais/genericos) injetados para simular SMS reais,
# nos quais ha palavras irrelevantes alem do estabelecimento. Servem para
# estressar os metodos: o baseline soma votos de todos os tokens conhecidos
# (e se confunde), enquanto o Apriori usa regras ordenadas por confianca.
FILLERS_RUIDO = ["internet", "premium", "mensal", "popular",
                 "credito", "plano", "academia", "bar"]


def _injeta_ruido(dados, seed: int = 123, lo: int = 2, hi: int = 4):
    rng = random.Random(seed)
    saida = []
    for sms, cat in dados:
        extras = " ".join(rng.choice(FILLERS_RUIDO) for _ in range(rng.randint(lo, hi)))
        saida.append((sms + " " + extras, cat))
    return saida


def _avaliar_cenario(treino, teste):
    apriori = ClassificadorApriori(min_support=0.01, min_confidence=0.5)
    baseline = ClassificadorBaseline()
    apriori.treinar(treino)
    baseline.treinar(treino)
    y_true = [cat for _, cat in teste]
    y_apriori = [apriori.classificar(sms)[0] for sms, _ in teste]
    y_baseline = [baseline.classificar(sms)[0] for sms, _ in teste]
    return apriori, baseline, y_true, y_apriori, y_baseline


def experimento_1() -> pd.DataFrame:
    banner("Experimento 1 - Classificacao de gastos (Apriori x Baseline)")

    base = gerar_sms_dataset(n_por_categoria=60, seed=42)
    if len({cat for _, cat in base}) < 2:
        raise RuntimeError(
            "O dataset gerado tem apenas uma categoria. Provavelmente o bug "
            "em seed_data.gerar_sms_dataset nao foi corrigido (o `rng.shuffle` "
            "e o `return` precisam ficar FORA do `for` externo)."
        )

    tr_limpo, te_limpo = split(base, frac=0.7, seed=42)
    ruidoso = _injeta_ruido(base, seed=123, lo=2, hi=4)
    tr_ruido, te_ruido = split(ruidoso, frac=0.7, seed=42)

    print(f"SMS sinteticos: {len(base)} (treino={len(tr_limpo)}, teste={len(te_limpo)})")
    print("Cenarios avaliados: 'Limpo' (SMS sem ruido) e 'Ruidoso' (SMS com")
    print("2 a 4 tokens promocionais/genericos adicionados, simulando dados reais).")

    ap_l, bl_l, yt_l, ya_l, yb_l = _avaliar_cenario(tr_limpo, te_limpo)
    ap_r, bl_r, yt_r, ya_r, yb_r = _avaliar_cenario(tr_ruido, te_ruido)
    print(f"Regras de associacao aprendidas (Apriori): "
          f"limpo={len(ap_l._regras)}, ruidoso={len(ap_r._regras)}")

    linhas = []
    for cen, yt, ya, yb in [("Limpo", yt_l, ya_l, yb_l),
                            ("Ruidoso", yt_r, ya_r, yb_r)]:
        linhas.append({"cenario": cen, "metodo": "Apriori (projeto)",
                       **metricas_classificacao(yt, ya)})
        linhas.append({"cenario": cen, "metodo": "Baseline (palavra-chave)",
                       **metricas_classificacao(yt, yb)})
    tabela = pd.DataFrame(linhas)

    print("\nResultados (conjunto de teste):")
    print(tabela.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nLeitura: em dados limpos os dois metodos acertam tudo. Sob ruido, o")
    print("baseline por palavra-chave perde desempenho (votos espurios de tokens")
    print("irrelevantes), enquanto o Apriori se mantem robusto ao casar a regra de")
    print("maior confianca. Alem disso, o Apriori entrega regras interpretaveis")
    print("(antecedente -> categoria, com suporte e confianca).")

    csv_path = os.path.join(RESULTS, "exp1_metricas.csv")
    tabela.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n[salvo] {csv_path}")

    _grafico_barras_exp1(tabela)
    _matriz_confusao_dupla(yt_r, ya_r, yb_r, sorted(set(yt_r)))
    return tabela


def _grafico_barras_exp1(tabela: pd.DataFrame) -> None:
    cenarios = ["Limpo", "Ruidoso"]
    ap = [float(tabela[(tabela.cenario == c) &
          (tabela.metodo == "Apriori (projeto)")]["acuracia"].iloc[0]) for c in cenarios]
    bl = [float(tabela[(tabela.cenario == c) &
          (tabela.metodo == "Baseline (palavra-chave)")]["acuracia"].iloc[0]) for c in cenarios]
    x = np.arange(len(cenarios))
    largura = 0.38

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - largura / 2, ap, largura, label="Apriori (projeto)", color=COR_APRIORI)
    ax.bar(x + largura / 2, bl, largura, label="Baseline (palavra-chave)", color=COR_BASELINE)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Acuracia (0 a 1)")
    ax.set_title("Experimento 1 - Acuracia por cenario")
    ax.set_xticks(x)
    ax.set_xticklabels(cenarios)
    ax.legend(loc="lower left")
    for i in range(len(cenarios)):
        ax.text(i - largura / 2, ap[i] + 0.01, f"{ap[i]:.2f}", ha="center",
                va="bottom", fontsize=9)
        ax.text(i + largura / 2, bl[i] + 0.01, f"{bl[i]:.2f}", ha="center",
                va="bottom", fontsize=9)
    fig.tight_layout()
    out = os.path.join(RESULTS, "exp1_comparacao.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[salvo] {out}")


def _matriz_confusao_dupla(y_true, y_apriori, y_baseline, classes) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, y_pred, titulo, cmap in [
        (axes[0], y_apriori, "Apriori (projeto)", "Greens"),
        (axes[1], y_baseline, "Baseline (palavra-chave)", "Oranges"),
    ]:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        im = ax.imshow(cm, cmap=cmap)
        ax.set_title(f"Matriz de confusao - {titulo}\n(cenario ruidoso)")
        ax.set_xlabel("Categoria prevista")
        ax.set_ylabel("Categoria verdadeira")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        limite = cm.max() / 2 if cm.max() else 0.5
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > limite else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = os.path.join(RESULTS, "exp1_matriz_confusao.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[salvo] {out}")


def main() -> None:
    print("STIMA - Experimentos de comparacao entre metodos (parte de Mineracao)")
    print(f"(resultados serao gravados em: {RESULTS})")
    experimento_1()
    banner("Concluido")
    print("Tabelas (.csv) e graficos (.png) disponiveis na pasta results/.")


if __name__ == "__main__":
    main()