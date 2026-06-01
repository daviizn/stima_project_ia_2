"""
mining.py — Agente de Mineração/Recomendação (STIMA)

Responsável por classificar automaticamente gastos a partir do texto de
SMS bancários, sem intervenção do usuário.

Contém:

    1. extrair_tokens: pré-processamento de texto (função pura)
    2. ClassificadorBaseline: classificação por voto majoritário de palavras-chave
    3. ClassificadorApriori: classificação por regras de associação (mlxtend)

Ambos os classificadores seguem a mesma interface:

  .treinar(dados: list[tuple[str, str]]) -> None
  .classificar(sms: str) -> tuple[str, float]

A saída de .classificar() pode ser usada diretamente para preencher
os campos `categoria` e `confianca` de uma Transacao (domain.py).
"""

from __future__ import annotations
import pandas as pd

import re
import unicodedata

from collections import Counter
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Constantes internas
# Stopwords em português relevantes para SMS bancários
_STOPWORDS: set[str] = {
    "a", "o", "e", "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "para", "por", "com", "sem", "um", "uma", "uns",
    "umas", "ao", "aos", "as", "se", "que", "ou", "mas", "mais",

    "foi", "seu", "sua", "este", "esta", "isso", "aqui", "ja",
    "voce", "nos", "eles", "dele", "dela", "num", "numa",
    "ate", "apos", "desde", "entre", "sobre",
}

# Prefixo que distingue categorias de tokens comuns nos itemsets do Apriori
_PREFIXO_CAT = "CAT:"

# Dicionário de palavras-chave → categoria (usado pelo Baseline)
# Baseado nos estabelecimentos definidos em seed_data.py
_DICIONARIO_KEYWORDS: dict[str, str] = {
    # Alimentacao
    "extra": "alimentacao",
    "pao": "alimentacao",
    "acucar": "alimentacao",
    "assai": "alimentacao",
    "atacadista": "alimentacao",
    "carrefour": "alimentacao",
    "hortifruti": "alimentacao",
    "padaria": "alimentacao",
    "supermercado": "alimentacao",
    "mercado": "alimentacao",
    "restaurante": "alimentacao",
    "lanchonete": "alimentacao",
    "ifood": "alimentacao",
    "rappi": "alimentacao",
    "hamburgueria": "alimentacao",
    "pizzaria": "alimentacao",
    "acougue": "alimentacao",
    "panificadora": "alimentacao",

    # Transporte
    "posto": "transporte",
    "ipiranga": "transporte",
    "shell": "transporte",
    "uber": "transporte",
    "corrida": "transporte",
    "app99": "transporte",
    "estacionamento": "transporte",
    "combustivel": "transporte",
    "gasolina": "transporte",
    "etanol": "transporte",
    "pedagio": "transporte",
    "mecanica": "transporte",
    "oficina": "transporte",
    "onibus": "transporte",
    "metro": "transporte",

    # Saude
    "drogasil": "saude",
    "drogaria": "saude",
    "pacheco": "saude",
    "farmacia": "saude",
    "popular": "saude",
    "laboratorio": "saude",
    "sabin": "saude",
    "clinica": "saude",
    "odonto": "saude",
    "hospital": "saude",
    "medico": "saude",
    "dentista": "saude",
    "academia": "saude",
    "unimed": "saude",
    "plano": "saude",

    # Moradia
    "ceb": "moradia",
    "energia": "moradia",
    "caesb": "moradia",
    "agua": "moradia",
    "claro": "moradia",
    "internet": "moradia",
    "aluguel": "moradia",
    "imovel": "moradia",
    "condominio": "moradia",
    "edificio": "moradia",
    "gas": "moradia",
    "iptu": "moradia",
    "reforma": "moradia",
    "luz": "moradia",

    # Lazer
    "netflix": "lazer",
    "spotify": "lazer",
    "cinema": "lazer",
    "cinemark": "lazer",
    "ingresso": "lazer",
    "outback": "lazer",
    "teatro": "lazer",
    "bar": "lazer",
    "balada": "lazer",
    "steam": "lazer",
    "amazon": "lazer",
    "disney": "lazer",
    "premium": "lazer",
    "mensal": "lazer",

    # Dividas
    "parcela": "dividas",
    "emprestimo": "dividas",
    "fatura": "dividas",
    "financiamento": "dividas",
    "veiculo": "dividas",
    "credito": "dividas",
    "acordo": "dividas",
    "renegociacao": "dividas",
    "pessoal": "dividas",
}

# Função pura de pré-processamento
def extrair_tokens(sms: str) -> list[str]:
    """
    Recebe o texto bruto de um SMS bancário e devolve uma lista de tokens
    limpos e úteis para classificação.

    Etapas aplicadas em ordem:

        1. Minúsculas
        2. Remoção de acentos (normalização unicode NFKD)

        3. Remoção de valores monetários (R$ 1.234,56)
        4. Remoção de datas (10/05, 2024-05-10)

        5. Remoção de horários (14:30)
        6. Remoção de pontuação e caracteres especiais

        7. Remoção de stopwords
        8. Remoção de tokens com menos de 3 caracteres
    """
    texto = sms

    # 1. Minúsculas
    texto = texto.lower()

    # 2. Remoção de acentos
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))

    # 3. Valores monetários: R$ 1.234,56 ou R$45,90
    texto = re.sub(r"r\$\s*[\d.,]+", " ", texto)

    # 4. Datas: dd/mm, dd/mm/aaaa, aaaa-mm-dd
    texto = re.sub(r"\d{1,2}/\d{1,2}(?:/\d{2,4})?", " ", texto)
    texto = re.sub(r"\d{4}-\d{2}-\d{2}", " ", texto)

    # 5. Horários: hh:mm
    texto = re.sub(r"\d{1,2}:\d{2}", " ", texto)

    # 6. Pontuação, números isolados e caracteres especiais
    texto = re.sub(r"[^a-z\s]", " ", texto)

    # 7. Tokenização por espaço e remoção de stopwords
    tokens = [t for t in texto.split() if t not in _STOPWORDS]

    # 8. Remoção de tokens curtos (menos de 3 caracteres)
    tokens = [t for t in tokens if len(t) >= 3]

    return tokens

# Classificador Baseline — voto majoritário por palavras-chave
class ClassificadorBaseline:
    """
    Classifica um SMS pela categoria com maior número de tokens
    que constam no dicionário de palavras-chave.

    Não aprende com dados — o conhecimento está no dicionário fixo.
    O método .treinar() existe apenas para manter a mesma interface
    do ClassificadorApriori.
    """

    def __init__(self, dicionario: dict[str, str] | None = None) -> None:
        self._dic: dict[str, str] = dicionario if dicionario is not None \
            else _DICIONARIO_KEYWORDS

    def treinar(self, dados: list[tuple[str, str]]) -> None:
        """
        Não faz nada de fato — o Baseline não aprende com exemplos.
        Existe para manter interface uniforme com o ClassificadorApriori.
        """

        # Sem aprendizado; apenas valida o tipo recebido.
        if not isinstance(dados, list):
            raise TypeError("dados deve ser list[tuple[str, str]]")

    def classificar(self, sms: str) -> tuple[str, float]:
        # Classifica o SMS pela categoria com mais votos de tokens.
        tokens = extrair_tokens(sms)
        votos: Counter[str] = Counter()

        for token in tokens:
            if token in self._dic:
                votos[self._dic[token]] += 1

        if not votos:
            return ("outros", 0.0)

        total_votos = sum(votos.values())
        categoria, contagem = votos.most_common(1)[0]

        confianca = round(contagem / total_votos, 4)
        return (categoria, confianca)

# Classificador Apriori — regras de associação via mlxtend
class ClassificadorApriori:
    """
    Aprende regras de associação a partir de SMS rotulados e as usa para
    classificar novos SMS.

    Estratégia de representação: 

    Cada SMS de treino vira um itemset que mistura tokens e a categoria,
    distinguindo-os pelo prefixo "CAT:". Exemplo:
        SMS "Compra Uber R$ 22,50" com categoria "transporte"
        → itemset: ["compra", "uber", "CAT:transporte"]

    O Apriori encontra itemsets frequentes nessa coleção e o
    association_rules extrai regras do tipo:
        {uber} → {CAT:transporte}  (confiança = 1.0)

    Na classificação, os tokens do SMS novo são verificados contra os
    antecedentes das regras; a categoria com maior confiança vence.
    """

    def __init__(
        self,
        min_support: float = 0.05,
        min_confidence: float = 0.6,
        fallback: ClassificadorBaseline | None = None,
    ) -> None:
        self.min_support = min_support
        self.min_confidence = min_confidence

        self._fallback = fallback if fallback is not None \
            else ClassificadorBaseline()
        
        self._regras: pd.DataFrame = pd.DataFrame() # Preenchido em treinar()
        self._n_transacoes: int = 0

    def _montar_transacoes(
        self, dados: list[tuple[str, str]]
    ) -> list[list[str]]:
        """
        Converte pares (sms, categoria) em itemsets para o TransactionEncoder.
        Cada itemset = tokens do SMS + "CAT:<categoria>".
        """

        transacoes = []
        for sms, categoria in dados:
            tokens = extrair_tokens(sms)

            if not tokens:
                continue

            # Adiciona o rótulo da categoria como item especial
            itemset = tokens + [f"{_PREFIXO_CAT}{categoria}"]
            transacoes.append(itemset)

        return transacoes

    def treinar(self, dados: list[tuple[str, str]]) -> None:
        """
        Treina o classificador com SMS rotulados.

        Pipeline:

            1. Monta itemsets (tokens + CAT:<categoria>)
            2. Codifica com TransactionEncoder → DataFrame binário
            3. Encontra itemsets frequentes com apriori()
            4. Extrai regras com association_rules()
            5. Filtra apenas regras cujo consequente é uma categoria
        """

        transacoes = self._montar_transacoes(dados)
        self._n_transacoes = len(transacoes)

        if self._n_transacoes == 0:
            raise ValueError("Nenhuma transação válida encontrada nos dados.")

        # Codificação binária
        te = TransactionEncoder()
        te_array = te.fit_transform(transacoes)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)

        # Itemsets frequentes 
        freq_itemsets = apriori(
            df_encoded,
            min_support=self.min_support,
            use_colnames=True,
        )

        if freq_itemsets.empty:
            raise ValueError(
                f"Nenhum itemset frequente encontrado. "
                f"Tente reduzir min_support (atual: {self.min_support})."
            )

        # Regras de associação 
        # Nota: num_itemsets é obrigatório no mlxtend >= 0.22
        todas_regras = association_rules(
            freq_itemsets,
            num_itemsets=self._n_transacoes,
            metric="confidence",
            min_threshold=self.min_confidence,
        )

        # Filtra só regras que predizem uma categoria
        # O consequente deve conter exatamente um item do tipo "CAT:..."
        def _consequente_eh_categoria(itemset: frozenset) -> bool:
            return (
                len(itemset) == 1
                and next(iter(itemset)).startswith(_PREFIXO_CAT)
            )

        mask = todas_regras["consequents"].apply(_consequente_eh_categoria)
        self._regras = todas_regras[mask].copy()

        self._regras = self._regras.sort_values(
            "confidence", ascending=False
        ).reset_index(drop=True)

        if self._regras.empty:
            raise ValueError(
                "Nenhuma regra de categorização encontrada. "
                "Tente reduzir min_confidence ou aumentar os dados de treino."
            )

    def classificar(self, sms: str) -> tuple[str, float]:
        """
        Classifica o SMS usando as regras aprendidas.

        Processo:

            1. Tokeniza o SMS
            2. Verifica quais regras disparam (antecedente ⊆ tokens do SMS)
            
            3. Agrupa por categoria e guarda a maior confiança de cada uma
            4. Retorna a categoria com maior confiança

        Se nenhuma regra disparar, delega ao Baseline (fallback).
        """

        if self._regras.empty:
            raise RuntimeError(
                "Modelo não treinado. Chame .treinar() antes de .classificar()."
            )

        tokens_sms = set(extrair_tokens(sms))
        melhor_por_cat: dict[str, float] = {}

        for _, linha in self._regras.iterrows():
            antecedente: frozenset = linha["antecedents"]
            consequente: frozenset = linha["consequents"]
            confianca: float = linha["confidence"]

            """
            Regra dispara se todos os tokens do antecedente estão no SMS
            (ignoramos itens do antecedente que sejam CAT:... — raro mas possível em itemsets mistos)
            """

            tokens_antecedente = {
                item for item in antecedente
                if not item.startswith(_PREFIXO_CAT)
            }

            if tokens_antecedente and tokens_antecedente.issubset(tokens_sms):
                categoria = next(iter(consequente)).replace(_PREFIXO_CAT, "")

                # Guarda apenas a confiança máxima para cada categoria
                if categoria not in melhor_por_cat \
                        or confianca > melhor_por_cat[categoria]:
                    melhor_por_cat[categoria] = confianca

        if not melhor_por_cat:
            # Nenhuma regra disparou — usa o Baseline como fallback
            return self._fallback.classificar(sms)

        categoria_final = max(melhor_por_cat, key=lambda c: melhor_por_cat[c])
        return (categoria_final, round(melhor_por_cat[categoria_final], 4))