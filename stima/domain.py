"""
Modelos de dominio do STIMA

Estruturas de dados que circulam entre os agentes: perfis financeiros,
indicadores (perguntas), regras de linguagem do especialista, planos de 
contas e lancamentos financeiros
"""

from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class Indicador: 
    """
    Pergunta apresentada ao etudante.

    As opções sõa ORDINAIS: a posição na lista define a ordem crescente de gravidade,
    usada pelos operadores >, <, >=, <= da linguagem de regras.
    """

    id: str
    pergunta: str
    opcoes: list[str]

    def indice(self, valor: str) -> int:
        return self.opcoes.index(valor)

@dataclass
class Perfil:
    """
    Perfil financeiro cadastrado pelo especialista
    """

    id: str
    nome: str
    descricao: str

@dataclass
class Condicao:
    """
    Condicao atomica de uma regra: <indicador> <operador> <valor>
    """

    indicador: str
    operador: str
    valor: str

@dataclass
class Regra: 
    """
    Regra ja avaliada (saida do analisador do especialista)
    Forma: SE <condicoes> ENTAO <perfil> PESO <peso>
    """

    condicoes: list[Condicao]
    perfil: str
    peso: float
    origem_texto: str = ""  # texto original, para auditoria/depuracao

@dataclass
class PlanoDeContas: 
    """
    Distribuicao percentual recomendada de renda por categoria
    """

    pefil: str
    distribuicao: dict[str, float]  # categoria -> % (soma ~100)

    def valores(self, renda: float) -> dict[str, float]:
        """
        Converter os percentuais em valores monetarios para uma dada renda
        """
        return {c: round(renda * p / 100,2) for c, p in self.distribuicao.items()}

@dataclass
class Transacao:
    """
    Lancamento financeiro (gasto), vindo de SMS ou  manual
    """

    descricao: str
    valor: float
    categoria: str | None = None    # preenchida pela classificação
    fonte: str = "manual"           # "sms" ou "manual"
    confianca: float | None = None  # confianca de classificação 