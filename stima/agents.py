"""Agentes do STIMA (o Mediador esta em mediator.py).

- StimaExpert      : cadastra perfis, indicadores, planos e regras (validadas).
- StimaTutor       : motor de inferencia (RuleInterpreter) + plano de contas.
- StimaEstud       : interface do estudante (respostas, lancamentos, auxilio).
- FinancialTracker : minera SMS e classifica gastos (Apriori).
"""
from __future__ import annotations

from .domain import Indicador, Perfil, PlanoDeContas, Transacao
from .mining import ClassificadorApriori, extrair_valor
from .protocol import Mensagem
from .rule_dsl import ErroLexico, ErroSemantico, ErroSintatico, compilar_regra
from .rule_engine import MotorDeRegras


class Agente:
    """Classe base: um agente reage a mensagens roteadas pelo Mediador."""

    def __init__(self, id: str):
        self.id = id

    def receber(self, msg: Mensagem, mediador):
        raise NotImplementedError
    
    
class StimaExpert(Agente):
    """Agente Especialista - estrutra o conhecimento do dominio."""

    def receber(self, msg, mediador):
        banco = mediador.banco
        if msg.acao == "cadastrar_perfil":
            p = Perfil(**msg.payload)
            banco["perfis"][p.id] = p
            return {"ok": True, "perfil": p.id}
        if msg.acao == "cadastrar_indicador":
            ind = Indicador(**msg.payload)
            banco["indicadores"][ind.id] = ind
            return {"ok": True, "indicador": ind.id}
        if msg.acao == "cadastrar_plano":
            plano = PlanoDeContas(**msg.payload)
            banco["planos"][plano.perfil] = plano
            return {"ok": True, "plano": plano.perfil}
        if msg.acao == "cadastrar_regra":
            # analisador lexico + sintatico + semantico antes de armazenar
            try:
                regra = compilar_regra(
                    msg.payload["texto"], banco["indicadores"], banco["perfis"]
                )
            except (ErroLexico, ErroSintatico, ErroSemantico) as e:
                return {"ok": False, "erro": str(e)}
            banco["regras"].append(regra)
            return {"ok": True, "regra": regra.origem_texto}
        raise ValueError(f"Ação desconhecida no Especialista: {msg.acao}")
    

class StimaTutor(Agente):
    """Agente Tutor — infere o perfil e personaliza o plano de contas."""

    # Base de conhecimento simples para o auxilio (poderia ser expandida).
    _DICAS = {
        "endividado_critico": (
            "Priorize quitar a divida mais cara (rotativo/cheque especial) e "
            "renegocie prazos antes de assumir novas parcelas."
        ),
        "endividado_moderado": (
            "Concentre pagamentos acima do minimo da fatura e evite contratar "
            "novos financiamentos ate reduzir o comprometimento."
        ),
        "equilibrado": (
            "Mantenha o controle dos lancamentos e aumente gradualmente a "
            "reserva de emergencia ate 3-6 meses de despesa."
        ),
        "poupador": (
            "Mantenha a reserva de 6 meses e considere diversificar os "
            "investimentos conforme seus objetivos."
        ),
    }

    def receber(self, msg, mediador):
        banco = mediador.banco
        if msg.acao == "inferir_perfil":
            motor = MotorDeRegras(
                banco["regras"], banco["indicadores"], list(banco["perfis"])
            )
            return motor.inferir(msg.payload["respostas"])
        if msg.acao == "gerar_plano":
            perfil = msg.payload["perfil"]
            renda = float(msg.payload.get("renda", 0.0))
            plano = banco["planos"].get(perfil)
            if not plano:
                return {"ok": False, "erro": "Sem plano para o perfil"}
            return {
                "ok": True,
                "perfil": perfil,
                "distribuicao": plano.distribuicao,
                "valores": plano.valores(renda)
            }
        if msg.acao == "responder_auxilio":
            perfil = msg.payload.get("perfil", "")
            return {
                "ok": True,
                "resposta": self._DICAS.get(
                    perfil, "Mantenha o registro dos gastos em dia.0"
                )
            }
        raise ValueError(f"Ação desconhecida no Tutor: {msg.acao}")
    

class StimaEstud(Agente):
    """Agente Estudante — interface do usuario final."""

    def __init__(self, id: str, usuario: str):
        super().__init__(id)
        self.usuario = usuario
        self.respostas: dict[str, str] = {}
        self.lancamentos: list[Transacao] = []

    def receber(self, msg, mediador):
        if msg.acao == "registrar_resposta":
            self.respostas[msg.payload["indicador"]] = msg.payload["valor"]
            return {"ok": True}
        if msg.acao == "indicadores_pendentes":
            todos = set(mediador.banco["indicadores"])
            return {"pendentes": sorted(todos - set(self.respostas))}
        if msg.acao == "registrar_lancamento":
            self.lancamentos.append(Transacao(**msg.payload))
            return {"ok": True}
        raise ValueError(f"Ação desconhecida no Estudante: {msg.acao}")
    

class FinancialTracker(Agente):
    """Agente de Mineracao/Recomendacao — minera SMS e classifica gastos."""

    def __init__(self, id: str, classificador: ClassificadorApriori):
        super().__init__(id)
        self.classificador = classificador

    def receber(self, msg, mediador):
        if msg.acao == "classificar_sms":
            sms = msg.payload["sms"]
            categoria, confianca = self.classificador.classificar(sms)
            valor = extrair_valor(sms)
            t = Transacao(
                descricao=sms,
                valor=valor or 0.0,
                categoria=categoria,
                fonte="sms"
                confianca=confianca
            )
            return {"ok": True, "transacao": t}
        raise ValueError(f"Ação desconhecida no Tracker: {msg.acao}")