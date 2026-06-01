"""Classificador Naive Bayes para inferencia de perfil financeiro.

Serve como METODO DE COMPARACAO ao motor de regras do Tutor: aprende, a
partir de exemplos (respostas -> perfil), a estimar P(perfil | respostas) e
prever o perfil mais provavel. Usa suavizacao de Laplace para valores raros.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict


class NaiveBayesPerfil:
    def __init__(self):
        self.classes: list[str] = []
        self.prior: dict[str, float] = {}
        # P(valor | classe) por indicador: cond[indicador][classe][valor]
        self.cond: dict[str, dict[str, dict[str, float]]] = {}
        self.indicadores: list[str] = []
        self.opcoes: dict[str, set] = defaultdict(set)

    def treinar(self, exemplos: list[tuple[dict[str, str], str]]) -> None:
        contagem_classe = Counter(rotulo for _, rotulo in exemplos)
        self.classes = sorted(contagem_classe)
        n = len(exemplos)
        self.prior = {c: contagem_classe[c] / n for c in self.classes}
        if exemplos:
            self.indicadores = sorted(exemplos[0][0].keys())

        cont = {ind: {c: Counter() for c in self.classes} for ind in self.indicadores}
        for respostas, rotulo in exemplos:
            for ind, val in respostas.items():
                cont[ind][rotulo][val] += 1
                self.opcoes[ind].add(val)

        # probabilidades condicionais com suavizacao de Laplace
        self.cond = {}
        for ind in self.indicadores:
            self.cond[ind] = {}
            k = len(self.opcoes[ind])
            for c in self.classes:
                total = sum(cont[ind][c].values())
                self.cond[ind][c] = {
                    val: (cont[ind][c][val] + 1) / (total + k)
                    for val in self.opcoes[ind]
                }

    def _logprob_classe(self, respostas: dict[str, str], c: str) -> float:
        lp = math.log(self.prior[c])
        for ind, val in respostas.items():
            k = len(self.opcoes.get(ind, [])) or 1
            tabela = self.cond.get(ind, {}).get(c, {})
            p = tabela.get(val, 1 / (k + 1))   # valor nunca visto
            lp += math.log(p)
        return lp

    def prever(self, respostas: dict[str, str]) -> str:
        return max(self.classes, key=lambda c: self._logprob_classe(respostas, c))

    def afinidade(self, respostas: dict[str, str]) -> dict[str, float]:
        """Distribuicao de probabilidade (softmax dos log-likelihoods)."""
        logs = {c: self._logprob_classe(respostas, c) for c in self.classes}
        m = max(logs.values())
        exps = {c: math.exp(l - m) for c, l in logs.items()}
        s = sum(exps.values())
        return {c: round(v / s, 4) for c, v in exps.items()}
