"""Linguagem propria de regras do Agente Especialista.

Gramatica (analisador por descida recursiva):

    regra    := 'SE' condicao ('E' condicao)* 'ENTAO' IDENT 'PESO' NUMERO
    condicao := IDENT OPERADOR VALOR
    OPERADOR := '==' | '!=' | '>=' | '<=' | '>' | '<'
    VALOR    := STRING | IDENT | NUMERO

O RuleInterpreter do Tutor usa este modulo para as analises LEXICA
(tokenizacao), SINTATICA (parsing) e SEMANTICA (validacao das referencias),
exatamente como descrito no projeto: "Um analisador sintatico valida cada
regra antes de armazena-la".
"""
from __future__ import annotations
from dataclasses import dataclass
from .domain import Condicao, Indicador, Perfil, Regra

class ErroLexico(Exception):
    ...

class ErroSintatico(Exception):
    ...

class ErroSemantico(Exception):
    ...

PALAVRAS_CHAVE = {"SE", "E", "ENTAO", "PESO"}
OPERADORES = {"==", "!=", ">=", "<=", ">", "<"}

@dataclass
class Token:
    tipo: str   # KEYWORD, IDENT, OP, NUMBER, STRING, EOF
    valor: str
    pos: int

def tokenizar(texto: str) -> list[Token]:
    """Analise lexica: transforma o texto da regra em uma lista de tokens."""
    tokens: list[Token] = []
    i, n = 0, len(texto)
    while i < n:
        c = texto[i]
        if c.isspace():
            i += 1
            continue
        if texto[i:i + 2] in OPERADORES:
            tokens.append(Token("OP", texto[i:i + 2], i))
            i += 2
            continue
        if c in "<>":
            tokens.append(Token("OP", c, i))
            i += 1
            continue
        if c == '"':
            j = i + 1
            while j < n and texto[j] != '"':
                j += 1
            if j >= n:
                raise ErroLexico(f"String nao fechada na posicao {i}")
            tokens.append(Token("STRING", texto[i+1:j], i))
            i = j + 1
            continue
        if c.isdigit() or (c == "-" and i + 1 < n and texto[i + 1].isdigit()):
            j = i + 1
            while j < n and (texto[j].isdigit() or texto[j] == "."):
                j += 1
            tokens.append(Token("NUMBER", texto[i:j], i))
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and (texto[j].isalnum() or texto[j] == "_"):
                j += 1
            palavra = texto[i:j]
            tipo = "KEYWORD" if palavra.upper() in PALAVRAS_CHAVE else "IDENT"
            tokens.append(Token(tipo, palavra, i))
            i = j
            continue
        raise ErroLexico(f"Caractere inesperado '{c}' na posicao {i}")
    tokens.append(Token("EOF", "", n))
    return tokens

class _Parser:
    """Analise sintatica por descida recursiva."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.k = 0

    def _atual(self) -> Token:
        return self.tokens[self.k]
    
    def _consome(self, tipo: str, valor: str | None = None) -> Token:
        t = self._atual()
        if t.tipo != tipo or (valor is not None and t.valor.upper() != valor.upper()):
            esperado = valor or tipo
            raise ErroSintatico(
                f"Esperado '{esperado}', encontrado '{t.valor or t.tipo}' (pos {t.pos})"
            )
        self.k += 1
        return t
    
    def parse(self) -> Regra:
        self._consome("KEYWORD", "SE")
        condicoes = [self._condicao()]
        while self._atual().tipo == "KEYWORD" and self._atual().valor.upper() == "E":
            self._consome("KEYWORD", "E")
            condicoes.append(self._condicao())
        self._consome("KEYWORD", "ENTAO")
        perfil = self._consome("IDENT").valor
        self._consome("KEYWORD", "PESO")
        peso = float(self._consome("NUMBER").valor)
        self._consome("EOF")
        return Regra(condicoes=condicoes, perfil=perfil, peso=peso)
    
    def _condicao(self) -> Condicao:
        ind = self._consome("IDENT").valor
        op = self._consome("OP").valor
        t = self._atual()
        if t.tipo not in ("STRING", "IDENT", "NUMBER"):
            raise ErroSintatico(f"Valor invalido em condicao (pos {t.pos})")
        self.k += 1
        return Condicao(indicador=ind, operador=op, valor=t.valor)
    
def validar_semantica(
        regra: Regra,
        indicadores: dict[str, Indicador],
        perfis: dict[str, Perfil],
) -> None:
    """Analise semantica: confere se indicadores/perfis existem e se os
    valores sao opcoes validas dos respectivos indicadores."""
    if regra.perfil not in perfis:
        raise ErroSemantico(f"Perfil desconhecido: '{regra.perfil}'")
    if regra.peso <= 0:
        raise ErroSemantico("PESO deve ser positivo")
    for c in regra.condicoes:
        if c.indicador not in indicadores:
            raise ErroSemantico(f"Indicador desconhecido: '{c.indicador}'")
        if c.operador not in OPERADORES:
            raise ErroSemantico(f"Operador invalido: '{c.operador}'")
        if c.valor not in indicadores[c.indicador].opcoes:
            raise ErroSemantico(
                f"Valor '{c.valor}' nao e opcao valida de '{c.indicador}'. "
                f"Opcoes: {indicadores[c.indicador].opcoes}"
            )
        
def compilar_regra(
        texto: str,
        indicadores: dict[str, Indicador],
        perfis: dict[str, Perfil],
) -> Regra: 
    """Pipeline completo de validacao: lexico -> sintatico -> semantico."""
    tokens = tokenizar(texto)
    regra = _Parser(tokens).parse()
    validar_semantica(regra, indicadores, perfis)
    regra.origem_texto = texto.strip()
    return regra