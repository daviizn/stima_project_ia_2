"""Motor de inferencia por regras — o RuleInterpreter do Agente Tutor.

Avalia as regras ja compiladas contra as respostas do estudante, acumula
os pesos por perfil (a AFINIDADE), normaliza e atribui o perfil de maior
afinidade, como descrito na seccao 3 do projeto.
"""
from __future__ import annotations

from .domain import Condicao, Indicador, Regra


def _compara(op: str, esquerda: float, direita: float) -> bool:
    return {
        "==": esquerda == direita,
        "!=": esquerda != direita,
        ">": esquerda > direita,
        "<": esquerda < direita,
        ">=": esquerda >= direita,
        "<=": esquerda <= direita,
    }[op]


def _condicao_satisfeita(
    cond: Condicao,
    respostas: dict[str, str],
    indicadores: dict[str, Indicador],
) -> bool:
    if cond.indicador not in respostas:
        return False
    resposta = respostas[cond.indicador]
    if cond.operador == "==":
        return resposta == cond.valor
    if cond.operador == "!=":
        return resposta != cond.valor
    ind = indicadores[cond.indicador]
    return _compara(cond.operador, ind.indice(resposta), ind.indice(cond.valor))


class MotorDeRegras:
    """Avaliador de regras com calculo de afinidade por perfil."""

    def __init__(
        self,
        regras: list[Regra],
        indicadores: dict[str, Indicador],
        perfis_ids: list[str],
    ):
        self.regras = regras
        self.indicadores = indicadores
        self.perfis_ids = perfis_ids

    def inferir(self, respostas: dict[str, str]) -> dict:
        """Retorna o perfil escolhido, a afinidade por perfil, os escores
        brutos e as regras que dispararam."""
        escores = {p: 0.0 for p in self.perfis_ids}
        disparadas: list[str] = []
        for regra in self.regras:
            if all(
                _condicao_satisfeita(c, respostas, self.indicadores)
                for c in regra.condicoes
            ):
                escores[regra.perfil] += regra.peso
                disparadas.append(regra.origem_texto)
        total = sum(escores.values())
        if total > 0:
            afinidade = {p: round(s / total, 4) for p, s in escores.items()}
            perfil = max(afinidade, key=afinidade.get)
        else:
            afinidade = {p: 0.0 for p in self.perfis_ids}
            perfil = None
        return {
            "perfil": perfil,
            "afinidade": afinidade,
            "escores": escores,
            "regras_disparadas": disparadas,
        }
