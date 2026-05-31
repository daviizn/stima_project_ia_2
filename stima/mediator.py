"""Agente Mediador/Integrador (StimaEF).

Ponto unico de orquestracao (coordenacao centralizada, seccao 6.3): autentica
atores, emite e valida tokens de sessao, roteia mensagens entre os agentes e
persiste os dados (aqui, em memoria). Nenhum agente fala diretamente com
outro — tudo passa por aqui, o que simplifica o controle de acesso e a
auditoria das trocas.
"""
from __future__ import annotations

import secrets

from .protocol import Mensagem

class Mediador:
    def __init__(self):
        self._agentes: dict[str, object] = {}  # id -> agente (.receber(msg, mediador))
        self._tokens: dict[str, str] = {}      # token -> ator
        self._credenciais: dict[str, str] = {} # usuario -> senha
        self.log: list[str] = []               # trilha de auditoria
        self.banco = {                         # "persistencia" em memoria
            "perfis": {},
            "indicadores": {},
            "regras": [],
            "planos": {},
            "usuarios": {}
        }

    #Infraestrutura
    def registrar_agente(self, agente) -> None:
        self._agentes[agente.id] = agente

    def cadastrar_credencial(self, usuario: str, senha: str) -> None:
        self._credenciais[usuario] = senha

    # Segurança
    def autenticar(self, usuario: str, senha: str) -> str | None:
        if self._credenciais.get(usuario) == senha:
            token = secrets.token_hex(8)
            self._tokens[token] = usuario
            self.log.append(f"AUTH ok: {usuario} -> token {token}")
            return token
        self.log.append(f"AUTH falhou: {usuario}")
        return None
    
    def _token_valido(self, token: str | None) -> bool:
        return token in self._tokens
    
    # Roteamento
    def enviar(self, msg: Mensagem):
        """Valida o token e roteia a mensagem ao agente de destino."""
        if not self._token_valido(msg.token):
            self.log.append(f"NEGADO (token invalido): {msg.resumo()}")
            raise PermissionError("Token invalido ou ausente")
        if msg.destino not in self._agentes:
            raise KeyError(f"Destino desconhecido: {msg.destino}")
        self.log.append(f"ROTA: {msg.resumo()}")
        return self._agentes[msg.destino].receber(msg, self)