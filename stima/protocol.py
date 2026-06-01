"""Protocolo de mensagens trocadas entre os agentes do STIMA.

Conforme a Figura 1 do projeto, nenhum agente fala diretamente com outro:
toda mensagem segue o formato  msg { origem, destino, token, acao, payload }
e passa pelo Mediador (StimaEF), que autentica, valida o token e roteia.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Mensagem:
    """Unidade de comunicação do barramento do STIMA."""

    origem: str             # id do ator/agente de origem
    destino: str            # id do agente de destino
    acao: str               # ex.: "inferir_perfil"
    payload: dict[str, Any] = field(default_factory=dict)
    token: str | None = None # token de sessao (apos auth)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: float = field(default_factory=time.time)

    def resumo(self) -> str:
        return f"[{self.id}] {self.origem} -> {self.destino} :: {self.acao}"