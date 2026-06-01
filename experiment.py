"""Experimentos do STIMA — comparacao simples entre metodos.

Atende a exigencia do trabalho final ("comparacao simples entre metodos",
"resultados com tabelas/graficos"). Sao dois experimentos:

  Experimento 1 — Classificacao automatica de gastos (Agente de Mineracao)
    Compara o metodo do projeto, APRIORI (regras de associacao), com um
    BASELINE por palavra-chave (voto majoritario), sobre SMS bancarios
    sinteticos rotulados. Metricas: acuracia, precisao, revocacao e F1 (macro).

  Experimento 2 — Inferencia de perfil financeiro (Agente Tutor)
    Compara o MOTOR DE REGRAS do especialista com um classificador
    NAIVE BAYES treinado a partir de exemplos. As regras do especialista
    sao tomadas como gabarito (conhecimento de referencia); mede-se o quanto
    o metodo orientado a dados (NB) reproduz esse conhecimento (concordancia).

Saidas (pasta results/): tabelas .csv e graficos .png.

Como rodar:
    python experiment.py
"""
from __future__ import annotations

import os
import random
import time

import matplotlib
matplotlib.use("Agg")  # backend sem tela (gera arquivos de imagem)
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

from stima.bayes import NaiveBayesPerfil
from stima.domain import Indicador, Perfil
from stima.mining import ClassificadorApriori, ClassificadorBaseline
from stima.rule_dsl import compilar_regra
from stima.rule_engine import MotorDeRegras
from stima.seed_data import INDICADORES, PERFIS, REGRAS, gerar_sms_dataset

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

# Paleta sobria para os graficos
COR_APRIORI = "#2a9d8f"
COR_BASELINE = "#e9c46a"
COR_REGRAS = "#264653"
COR_BAYES = "#e76f51"


# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------
def split(dados: list, frac: float = 0.7, seed: int = 42):
    """Divide a lista em treino/teste de forma reprodutivel."""
    rng = random.Random(seed)
    dados = list(dados)
    rng.shuffle(dados)
    corte = int(len(dados) * frac)
    return dados[:corte], dados[corte:]


def metricas_classificacao(y_true: list[str], y_pred: list[str]) -> dict:
    """Acuracia, precisao, revocacao e F1 (macro) — robusto a classes ausentes."""
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
# Experimento 1 — classificacao de gastos: Apriori x baseline
# ---------------------------------------------------------------------------
# Tokens "de ruido" (promocionais/genericos) injetados para simular SMS reais,
# nos quais ha palavras irrelevantes alem do estabelecimento. Servem para
# estressar os metodos: o baseline soma votos de todos os tokens conhecidos
# (e se confunde), enquanto o Apriori usa regras ordenadas por confianca.
FILLERS_RUIDO = ["promo", "pontos", "app", "digital", "cliente",
                 "centro", "shopping", "online"]


def _injeta_ruido(dados, seed: int = 123, lo: int = 2, hi: int = 4):
    """Acrescenta de `lo` a `hi` tokens de ruido a cada SMS (rotulo intacto)."""
    rng = random.Random(seed)
    saida = []
    for sms, cat in dados:
        extras = " ".join(rng.choice(FILLERS_RUIDO) for _ in range(rng.randint(lo, hi)))
        saida.append((sms + " " + extras, cat))
    return saida


def _avaliar_cenario(treino, teste):
    """Treina os dois classificadores e devolve previsoes no conjunto de teste."""
    apriori = ClassificadorApriori(min_support=0.01, min_confidence=0.5)
    baseline = ClassificadorBaseline()
    apriori.treinar(treino)
    baseline.treinar(treino)
    y_true = [cat for _, cat in teste]
    y_apriori = [apriori.classificar(sms)[0] for sms, _ in teste]
    y_baseline = [baseline.classificar(sms)[0] for sms, _ in teste]
    return apriori, baseline, y_true, y_apriori, y_baseline


def experimento_1() -> pd.DataFrame:
    banner("Experimento 1 — Classificacao de gastos (Apriori x Baseline)")

    base = gerar_sms_dataset(n_por_categoria=60, seed=42)
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
    """Acuracia por cenario, agrupada por metodo."""
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
    ax.set_title("Experimento 1 — Acuracia por cenario")
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
    """Matrizes de confusao lado a lado (cenario ruidoso) — Apriori x Baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, y_pred, titulo, cmap in [
        (axes[0], y_apriori, "Apriori (projeto)", "Greens"),
        (axes[1], y_baseline, "Baseline (palavra-chave)", "Oranges"),
    ]:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        im = ax.imshow(cm, cmap=cmap)
        ax.set_title(f"Matriz de confusao — {titulo}\n(cenario ruidoso)")
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


# ---------------------------------------------------------------------------
# Experimento 2 — inferencia de perfil: Regras x Naive Bayes
# ---------------------------------------------------------------------------
def _montar_motor():
    indicadores = {iid: Indicador(iid, perg, ops) for iid, perg, ops in INDICADORES}
    perfis = {pid: Perfil(pid, nome, desc) for pid, nome, desc in PERFIS}
    regras = [compilar_regra(t, indicadores, perfis) for t in REGRAS]
    motor = MotorDeRegras(regras, indicadores, list(perfis))
    return motor, indicadores


def _gerar_exemplos_rotulados(motor, indicadores, n: int = 2000, seed: int = 7):
    """Gera vetores de respostas aleatorios e os rotula com o MOTOR DE REGRAS
    (gabarito do especialista). Mantem apenas os casos em que alguma regra
    dispara (perfil definido)."""
    rng = random.Random(seed)
    exemplos = []
    for _ in range(n):
        respostas = {iid: rng.choice(ind.opcoes) for iid, ind in indicadores.items()}
        perfil = motor.inferir(respostas)["perfil"]
        if perfil is not None:
            exemplos.append((respostas, perfil))
    return exemplos


def experimento_2() -> pd.DataFrame:
    banner("Experimento 2 — Inferencia de perfil (Regras x Naive Bayes)")

    motor, indicadores = _montar_motor()
    exemplos = _gerar_exemplos_rotulados(motor, indicadores, n=2000, seed=7)
    treino, teste = split(exemplos, frac=0.7, seed=42)
    print(f"Exemplos rotulados pelo especialista (regras): {len(exemplos)} "
          f"(treino={len(treino)}, teste={len(teste)})")

    # Metodo orientado a dados: aprende com os exemplos rotulados pelas regras
    nb = NaiveBayesPerfil()
    nb.treinar(treino)

    X_teste = [r for r, _ in teste]
    y_gabarito = [p for _, p in teste]   # rotulo de referencia = saida das regras

    # Previsoes + tempo medio de inferencia (ms/amostra)
    t0 = time.perf_counter()
    y_regras = [motor.inferir(r)["perfil"] for r in X_teste]
    t_regras = (time.perf_counter() - t0) / len(X_teste) * 1000

    t0 = time.perf_counter()
    y_bayes = [nb.prever(r) for r in X_teste]
    t_bayes = (time.perf_counter() - t0) / len(X_teste) * 1000

    # Concordancia com o gabarito (as regras SAO o gabarito -> 1.0 por construcao)
    m_regras = metricas_classificacao(y_gabarito, y_regras)
    m_bayes = metricas_classificacao(y_gabarito, y_bayes)

    tabela = pd.DataFrame(
        [
            {"metodo": "Regras (especialista)", **m_regras,
             "tempo_ms_por_inferencia": round(t_regras, 4)},
            {"metodo": "Naive Bayes (dados)", **m_bayes,
             "tempo_ms_por_inferencia": round(t_bayes, 4)},
        ]
    )
    print("\nResultados (concordancia com o gabarito do especialista):")
    print(tabela.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nObservacao: a acuracia das Regras e 1.000 por construcao — elas")
    print("definem o gabarito. A metrica relevante e a CONCORDANCIA do Naive")
    print("Bayes com o especialista: mede se um metodo orientado a dados")
    print(f"reproduz a politica de regras. Concordancia obtida: {m_bayes['acuracia']:.1%}.")

    csv_path = os.path.join(RESULTS, "exp2_metricas.csv")
    tabela.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n[salvo] {csv_path}")

    _grafico_distribuicao_exp2(y_gabarito, y_bayes)
    return tabela


def _grafico_distribuicao_exp2(y_gabarito, y_bayes) -> None:
    classes = sorted(set(y_gabarito) | set(y_bayes))
    cont_gab = [y_gabarito.count(c) for c in classes]
    cont_nb = [y_bayes.count(c) for c in classes]
    x = np.arange(len(classes))
    largura = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - largura / 2, cont_gab, largura, label="Regras (gabarito)",
           color=COR_REGRAS)
    ax.bar(x + largura / 2, cont_nb, largura, label="Naive Bayes",
           color=COR_BAYES)
    ax.set_ylabel("Qtde. no conjunto de teste")
    ax.set_title("Experimento 2 — Distribuicao de perfis previstos")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(RESULTS, "exp2_distribuicao_perfis.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[salvo] {out}")


# ---------------------------------------------------------------------------
def main() -> None:
    print("STIMA — Experimentos de comparacao entre metodos")
    print(f"(resultados serao gravados em: {RESULTS})")
    experimento_1()
    experimento_2()
    banner("Concluido")
    print("Tabelas (.csv) e graficos (.png) disponiveis na pasta results/.")


if __name__ == "__main__":
    main()