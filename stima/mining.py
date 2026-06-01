"""
Mineracao de SMS bancarios e classificacao automatica de gastos.
Nucleo do Agente de Mineracao/Recomendacao (Financial Tracker).

- Mineracao de texto: normaliza o SMS e extrai tokens relevantes
  (tipicamente o estabelecimento), removendo valores e palavras vazias.

- Metodo principal: regras de associacao (APRIORI) que ligam tokens do SMS
  a categoria do plano de contas, escolhendo a de MAIOR CONFIANCA, exatamente
  como descrito no projeto (algoritmo Apriori de Agrawal & Srikant, 1994).

- Metodo de comparacao: baseline por palavra-chave (voto majoritario).
"""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Palavras comuns dos SMS bancarios que nao ajudam a discriminar a categoria.
STOPWORDS = {
    "compra", "aprovada", "aprovado", "cartao", "credito", "debito", "no", "na",
    "de", "do", "da", "em", "para", "com", "valor", "transacao", "voce", "fez",
    "uma", "foi", "realizada", "realizado", "pagamento", "final",
}

def extrair_tokens(sms: str) -> list[str]:
    """
    Mineracao de texto simples: minuscula, remove R$/numeros/pontuacao e
    stopwords; devolve os tokens relevantes (geralmente o estabelecimento).
    """
    texto = sms.lower()
    texto = re.sub(r"r\$?\s*\d+[\.,]?\d*", " ", texto) # Remove "R$ 45,90"
    texto = re.sub(r"\d+", " ", texto) # Remove numeros soltos
    texto = re.sub(r"[^a-zà-ú\s]", " ", texto) # Remove pontuacao

    return [t for t in texto.split() if len(t) > 2 and t not in STOPWORDS]

def extrair_valor(sms: str) -> float | None:
    # Extrai o valor monetario do SMS, se houver.
    m = re.search(r"r\$?\s*(\d+[\.,]?\d*)", sms.lower())

    if not m:
        return None

    return float(m.group(1).replace(".", "").replace(",", "."))

class ClassificadorApriori:
    # Classifica gastos via regras de associacao  token -> categoria.

    def __init__(self, suporte_min: float = 0.02, confianca_min: float = 0.3):
        self.suporte_min = suporte_min
        self.confianca_min = confianca_min
        self.regras = pd.DataFrame()
        self.categorias: list[str] = []
        self.cat_mais_comum: str | None = None

    def treinar(self, dados: list[tuple[str, str]]) -> None:
        # dados: lista de pares (sms, categoria).
        categorias = [cat for _, cat in dados]

        self.categorias = sorted(set(categorias))
        self.cat_mais_comum = max(set(categorias), key=categorias.count)

        # Cada transacao = tokens do SMS + um marcador da categoria.
        transacoes = []

        for sms, cat in dados:
            itens = set(extrair_tokens(sms))
            itens.add(f"CAT::{cat}")
            transacoes.append(list(itens))

        # API do mlxtend — exige matriz booleana one-hot
        te = TransactionEncoder()

        matriz = te.fit_transform(transacoes)
        df = pd.DataFrame(matriz, columns=te.columns_)

        # Itemsets frequentes acima do suporte minimo
        freq = apriori(df, min_support=self.suporte_min, use_colnames=True)
        if freq.empty:
            return

        # Regras de associacao acima da confianca minima
        regras = association_rules(
            freq, metric="confidence", min_threshold=self.confianca_min
        )

        # Mantem apenas regras  {tokens} -> {CAT::categoria}
        def consequente_e_categoria(conseq) -> bool:
            return len(conseq) == 1 and next(iter(conseq)).startswith("CAT::")

        def antecedente_sem_categoria(antec) -> bool:
            return all(not it.startswith("CAT::") for it in antec)

        regras = regras[
            regras["consequents"].apply(consequente_e_categoria)
            & regras["antecedents"].apply(antecedente_sem_categoria)
        ]

        # Ordena por confianca (e suporte como desempate) — "maior confianca".
        self.regras = regras.sort_values(
            ["confidence", "support"], ascending=False
        ).reset_index(drop=True)

    def classificar(self, sms: str) -> tuple[str, float]:
        """
        Casa a regra de MAIOR CONFIANCA cujo antecedente esteja contido nos
        tokens do SMS. Retorna (categoria, confianca).
        """
        tokens = set(extrair_tokens(sms))

        for _, regra in self.regras.iterrows():
            if set(regra["antecedents"]).issubset(tokens):
                cat = next(iter(regra["consequents"])).replace("CAT::", "")
                return cat, float(regra["confidence"])

        # fallback: nenhuma regra casou -> categoria mais comum, confianca 0
        return self.cat_mais_comum, 0.0

    def top_regras(self, n: int = 10) -> pd.DataFrame:
        # Tabela legivel das melhores regras (para inspecao/relatorio).
        if self.regras.empty:
            return pd.DataFrame()

        out = self.regras.head(n).copy()
        out["antecedente"] = out["antecedents"].apply(lambda s: ", ".join(sorted(s)))

        out["categoria"] = out["consequents"].apply(
            lambda s: next(iter(s)).replace("CAT::", "")
        )

        return out[["antecedente", "categoria", "support", "confidence", "lift"]]

class ClassificadorBaseline:
    # Baseline por palavra-chave (voto majoritario) — metodo de comparacao
    def __init__(self):
        self.mapa: dict[str, str] = {} # token -> categoria mais frequente
        self.cat_mais_comum: str | None = None

    def treinar(self, dados: list[tuple[str, str]]) -> None:
        contagem: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        categorias = []

        for sms, cat in dados:
            categorias.append(cat)

            for t in extrair_tokens(sms):
                contagem[t][cat] += 1

        self.cat_mais_comum = max(set(categorias), key=categorias.count)

        for token, cats in contagem.items():
            self.mapa[token] = max(cats, key=cats.get)

    def classificar(self, sms: str) -> tuple[str, float]:
        votos: dict[str, int] = defaultdict(int)

        for t in extrair_tokens(sms):
            if t in self.mapa:
                votos[self.mapa[t]] += 1

        if not votos:
            return self.cat_mais_comum, 0.0

        cat = max(votos, key=votos.get)
        return cat, votos[cat] / sum(votos.values())