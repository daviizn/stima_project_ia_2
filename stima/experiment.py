"""
experiment.py — Experimento 1: Comparação entre Baseline e Apriori (STIMA)

Avalia e compara os dois classificadores do Agente de Mineração:

    1. ClassificadorBaseline: voto majoritário por palavras-chave
    2. ClassificadorApriori: regras de associação via mlxtend

Métricas calculadas (via sklearn):
    acurácia, precisão macro, recall macro, F1 macro e F1 por categoria.

Saídas:

    1. Tabela comparativa impressa no terminal
    2. resultado_geral.png: barras com as 4 métricas lado a lado
    3. resultado_por_categoria.png: F1 por categoria para cada método
"""

from __future__ import annotations

import os
import random
import sys

import matplotlib
matplotlib.use("Agg") # Backend sem janela — compatível com qualquer SO

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import train_test_split

# Garante que o diretório do script está no path, independente de onde
# o usuário rodar o comando.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mining import ClassificadorApriori, ClassificadorBaseline

# Pasta de saída para os gráficos — criada automaticamente se não existir
_DIR_OUTPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(_DIR_OUTPUTS, exist_ok=True)

# Reprodução do gerador de SMS sintéticos do seed_data.py
_ESTABELECIMENTOS = {
    "alimentacao": [
        "EXTRA SUPERMERCADO", "PAO DE ACUCAR", "ASSAI ATACADISTA",
        "CARREFOUR HIPER", "HORTIFRUTI CENTRAL", "PADARIA BELA",
    ],
    "transporte": [
        "POSTO IPIRANGA", "POSTO SHELL", "UBER VIAGENS",
        "APP99 CORRIDA", "ESTACIONAMENTO ROTA",
    ],
    "saude": [
        "DROGASIL", "DROGARIA PACHECO", "FARMACIA POPULAR",
        "LABORATORIO SABIN", "CLINICA ODONTO",
    ],
    "moradia": [
        "CEB ENERGIA", "CAESB AGUA", "CLARO INTERNET",
        "ALUGUEL IMOVEL", "CONDOMINIO EDIFICIO",
    ],
    "lazer": [
        "IFOOD PEDIDO", "CINEMARK INGRESSO", "NETFLIX MENSAL",
        "SPOTIFY PREMIUM", "RESTAURANTE OUTBACK",
    ],
    "dividas": [
        "PARCELA EMPRESTIMO", "FATURA CARTAO", "FINANCIAMENTO VEICULO",
        "CREDITO PESSOAL", "ACORDO RENEGOCIACAO",
    ],
}

_TEMPLATES = [
    "CARTAO Compra aprovada R$ {valor} {estab}",
    "Compra aprovada R$ {valor} em {estab}",
    "Voce fez uma compra de R$ {valor} no {estab}",
    "Debito R$ {valor} {estab}",
    "Pagamento R$ {valor} {estab} realizado",
]

CATEGORIAS = sorted(_ESTABELECIMENTOS.keys())

def gerar_sms_dataset(n_por_categoria: int = 60, seed: int = 42):
    # Gera lista de pares (sms, categoria) com dados sintéticos
    rng = random.Random(seed)
    dados = []

    for categoria, estabs in _ESTABELECIMENTOS.items():
        for _ in range(n_por_categoria):
            estab = rng.choice(estabs)
            valor = f"{rng.uniform(10, 1500):.2f}".replace(".", ",")

            template = rng.choice(_TEMPLATES)
            dados.append((template.format(valor=valor, estab=estab), categoria))

    rng.shuffle(dados)
    return dados

# Avaliação
def avaliar(
    nome: str,
    classificador: ClassificadorBaseline | ClassificadorApriori,
    dados_teste: list[tuple[str, str]],
) -> dict:
    """
    Avalia um classificador já treinado sobre o conjunto de teste.
    Coleta y_real e y_pred, depois calcula acurácia, precisão, recall,
    F1 macro e F1 por categoria via sklearn.
    """
    y_real, y_pred = [], []

    for sms, categoria_real in dados_teste:
        categoria_pred, _ = classificador.classificar(sms)
        y_real.append(categoria_real)
        y_pred.append(categoria_pred)

    acuracia = accuracy_score(y_real, y_pred)
    precisao = precision_score(y_real, y_pred, average="macro", zero_division=0)

    recall = recall_score(y_real, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_real, y_pred, average="macro", zero_division=0)

    # F1 por categoria — usado no gráfico detalhado
    f1_cat = f1_score(
        y_real, y_pred,
        labels=CATEGORIAS,
        average=None,
        zero_division=0,
    )

    f1_por_categoria = dict(zip(CATEGORIAS, f1_cat))

    return {
        "nome": nome,
        "acuracia": acuracia,
        "precisao": precisao,
        "recall": recall,
        "f1": f1,
        "f1_por_categoria": f1_por_categoria,
        "y_real": y_real,
        "y_pred": y_pred,
    }

# Impressão da tabela no terminal
def imprimir_tabela(resultados: list[dict]) -> None:
    # Imprime tabela comparativa e relatório detalhado por categoria.
    col = 14
    metricas = ["acuracia", "precisao", "recall", "f1"]
    cabecalho = ["Método", "Acurácia", "Precisão", "Recall", "F1-Score"]

    separador = "+" + "+".join(["-" * (col + 2)] * len(cabecalho)) + "+"
    linha_fmt = "| " + " | ".join([f"{{:<{col}}}"] * len(cabecalho)) + " |"

    print()
    print("  EXPERIMENTO 1 — Comparação Baseline vs Apriori")

    print(separador)
    print(linha_fmt.format(*cabecalho))
    print(separador)

    for r in resultados:
        valores = [r["nome"]] + [f"{r[m]:.4f}" for m in metricas]
        print(linha_fmt.format(*valores))

    print(separador)
    print()

    for r in resultados:
        print(f"  Relatório completo — {r['nome']}")
        print(
            classification_report(
                r["y_real"], r["y_pred"],
                labels=CATEGORIAS,
                zero_division=0,
            )
        )

# Gráfico 1 — Métricas gerais lado a lado
def plotar_comparacao_geral(
    resultados: list[dict],
    caminho: str = os.path.join(_DIR_OUTPUTS, "resultado_geral.png"),
) -> None:
    """
    Gráfico de barras agrupadas com acurácia, precisão, recall e F1
    para cada método. Salva o arquivo em `caminho`.
    """
    metricas = ["acuracia", "precisao", "recall", "f1"]
    rotulos = ["Acurácia", "Precisão", "Recall", "F1-Score"]

    n_metricas = len(metricas)
    n_metodos = len(resultados)

    x = np.arange(n_metricas)
    largura = 0.35
    cores = ["#2196F3", "#FF5722"] # Azul = Baseline, laranja = Apriori

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (res, cor) in enumerate(zip(resultados, cores)):
        valores = [res[m] for m in metricas]
        offset = (i - (n_metodos - 1) / 2) * largura
        barras = ax.bar(x + offset, valores, largura, label=res["nome"], color=cor, alpha=0.85)

        # Valor numérico em cima de cada barra
        for barra, val in zip(barras, valores):
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                barra.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_title("Comparação Geral — Baseline vs Apriori", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Valor da Métrica", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"[✓] Gráfico salvo: {caminho}")


# Gráfico 2 — F1 por categoria
def plotar_comparacao_por_categoria(
    resultados: list[dict],
    caminho: str = os.path.join(_DIR_OUTPUTS, "resultado_por_categoria.png"),
) -> None:
    """
    Gráfico de barras agrupadas com F1-Score de cada categoria
    para cada método. Revela onde cada classificador vai melhor e pior.
    Salva o arquivo em `caminho`.
    """
    n_cats = len(CATEGORIAS)
    n_metodos = len(resultados)
    x = np.arange(n_cats)
    largura = 0.35
    cores = ["#2196F3", "#FF5722"]

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (res, cor) in enumerate(zip(resultados, cores)):
        valores = [res["f1_por_categoria"].get(cat, 0.0) for cat in CATEGORIAS]
        offset = (i - (n_metodos - 1) / 2) * largura
        barras = ax.bar(x + offset, valores, largura, label=res["nome"], color=cor, alpha=0.85)

        for barra, val in zip(barras, valores):
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                barra.get_height() + 0.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
            )

    rotulos_cats = [c.capitalize() for c in CATEGORIAS]

    ax.set_title("F1-Score por Categoria — Baseline vs Apriori", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("F1-Score", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos_cats, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"[✓] Gráfico salvo: {caminho}")

# Orquestrador principal
def executar_experimento() -> None:
    """
    Ponto de entrada do experimento. Orquestra todas as etapas:

        1. Gera o dataset sintético
        2. Faz o split treino/teste (80/20, estratificado por categoria)
        3. Treina e avalia o Baseline
        4. Treina e avalia o Apriori
        5. Imprime a tabela comparativa
        6. Gera os dois gráficos
    """

    print("\n[1/6] Gerando dataset sintético...")

    dados  = gerar_sms_dataset(n_por_categoria=60, seed=42)
    rotulos = [d[1] for d in dados]

    print("[2/6] Dividindo treino/teste (80/20, estratificado)...")

    treino, teste = train_test_split(
        dados, test_size=0.2, random_state=42, stratify=rotulos
    )

    print(f"Total: {len(dados)} | Treino: {len(treino)} | Teste: {len(teste)}")
    print("[3/6] Treinando e avaliando Baseline...")

    baseline = ClassificadorBaseline()
    baseline.treinar(treino)
    res_baseline = avaliar("Baseline", baseline, teste)

    print("[4/6] Treinando e avaliando Apriori...")

    apriori_clf = ClassificadorApriori(min_support=0.03, min_confidence=0.6)
    apriori_clf.treinar(treino)

    res_apriori = avaliar("Apriori", apriori_clf, teste)
    resultados = [res_baseline, res_apriori]

    print("[5/6] Imprimindo tabela comparativa...")
    imprimir_tabela(resultados)

    print("[6/6] Gerando gráficos...")
    plotar_comparacao_geral(resultados)
    plotar_comparacao_por_categoria(resultados)

    print("\n[✓] Experimento concluído.\n")

if __name__ == "__main__":
    executar_experimento()