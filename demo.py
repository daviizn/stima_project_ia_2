"""Demonstracao do STIMA - parte de Mineracao (Apriori).

Esta demo usa apenas os modulos atualmente disponiveis no projeto:

    domain.py, mediator.py, mining.py, protocol.py, seed_data.py

Os agentes Especialista, Tutor e Estudante (que viriam de stima.agents)
ainda nao foram implementados pelos colegas, portanto esta demo cobre
a fatia que JA ESTA pronta: o pipeline do agente de Mineracao/Recomendacao
(Financial Tracker), com toda a comunicacao passando pelo Mediador (StimaEF).

Fluxo demonstrado:

  0. Inicializacao do Mediador.
  1. Treinamento do ClassificadorApriori com SMS sinteticos rotulados.
  2. Registro do agente FinancialTracker no Mediador.
  3. Autenticacao do usuario e obtencao de token.
  4. Envio de SMS bancarios via Mediador; o Tracker classifica e devolve
     uma Transacao (categoria + confianca).
  5. Exibicao das regras de associacao aprendidas pelo Apriori.
  6. Trilha de auditoria do Mediador.

Execucao:  python demo.py
"""
from __future__ import annotations

import re

from stima import seed_data
from stima.domain import Transacao
from stima.mediator import Mediador
from stima.mining import ClassificadorApriori
from stima.protocol import Mensagem

LINHA = "=" * 66
PREFIXO_CAT = "CAT:"  # prefixo usado pelo Apriori para distinguir categorias


def banner(titulo: str) -> None:
    print(f"\n{LINHA}\n  {titulo}\n{LINHA}")


# ---------------------------------------------------------------------------
# Agente minimo de Mineracao (inline)
# ---------------------------------------------------------------------------
# Stub do Financial Tracker definido aqui dentro porque o modulo stima.agents
# ainda nao existe. Ele cumpre o contrato esperado pelo Mediador: ter um .id
# e um metodo .receber(msg, mediador). Quando o agente "oficial" estiver
# pronto, este stub pode ser removido e a importacao trocada.
# ---------------------------------------------------------------------------
class FinancialTracker:
    """Agente de mineracao: classifica SMS bancarios em categorias de gasto."""

    def __init__(self, identificador: str, classificador: ClassificadorApriori):
        self.id = identificador
        self._clf = classificador

    def receber(self, msg: Mensagem, mediador: Mediador) -> dict:
        if msg.acao == "classificar_sms":
            sms = msg.payload["sms"]
            categoria, confianca = self._clf.classificar(sms)
            transacao = Transacao(
                descricao=sms,
                valor=_extrair_valor(sms),
                categoria=categoria,
                fonte="sms",
                confianca=confianca,
            )
            return {"transacao": transacao}
        raise ValueError(f"Acao desconhecida para FinancialTracker: {msg.acao}")


def _extrair_valor(sms: str) -> float:
    """Tenta extrair o valor monetario do SMS (R$ 1.234,56 -> 1234.56)."""
    m = re.search(r"R\$\s*([\d.]+,\d{2})", sms)
    if not m:
        return 0.0
    return float(m.group(1).replace(".", "").replace(",", "."))


# ---------------------------------------------------------------------------
def main() -> None:
    # ----------------------------------------------------------------- 0
    banner("0. Inicializacao do Mediador (StimaEF) e do classificador")
    mediador = Mediador()
    mediador.cadastrar_credencial("aluno01", "1234")

    # Treina o classificador do Tracker com SMS sinteticos rotulados.
    clf = ClassificadorApriori(min_support=0.02, min_confidence=0.3)
    dados_sms = seed_data.gerar_sms_dataset()
    print(f"Dataset de treino: {len(dados_sms)} SMS sinteticos rotulados.")
    clf.treinar(dados_sms)
    print(f"Regras de associacao aprendidas: {len(clf._regras)}")

    # ----------------------------------------------------------------- 1
    banner("1. Registro do agente FinancialTracker no Mediador")
    tracker = FinancialTracker("FinancialTracker", clf)
    mediador.registrar_agente(tracker)
    print(f"Agentes registrados: {', '.join(mediador._agentes)}")

    # ----------------------------------------------------------------- 2
    banner("2. Autenticacao do usuario")
    token = mediador.autenticar("aluno01", "1234")
    print(f"Token de sessao obtido: {token}")

    # Tentativa com credencial errada (para demonstrar o controle de acesso).
    fail = mediador.autenticar("aluno01", "senha_errada")
    print(f"Tentativa com senha errada -> token={fail} (negado pelo Mediador)")

    # ----------------------------------------------------------------- 3
    banner("3. Classificacao automatica de SMS via Mediador")
    sms_exemplos = [
        "CARTAO Compra aprovada R$ 187,90 EXTRA SUPERMERCADO",
        "Compra aprovada R$ 62,00 em POSTO IPIRANGA",
        "Debito R$ 45,90 DROGASIL",
        "Pagamento R$ 980,00 PARCELA EMPRESTIMO realizado",
        "Voce fez uma compra de R$ 54,90 no IFOOD PEDIDO",
        "Debito R$ 350,00 CEB ENERGIA",
    ]
    print(f"{'Categoria':<14}{'Conf.':>6}{'Valor':>11}   SMS")
    print("-" * 66)
    transacoes: list[Transacao] = []
    for sms in sms_exemplos:
        resposta = mediador.enviar(Mensagem(
            origem="usuario",
            destino="FinancialTracker",
            acao="classificar_sms",
            payload={"sms": sms},
            token=token,
        ))
        t = resposta["transacao"]
        transacoes.append(t)
        print(f"{t.categoria:<14}{t.confianca:>5.0%} R$ {t.valor:>7.2f}   {sms[:36]}")

    print(f"\nLancamentos classificados: {len(transacoes)}")

    # Tentativa de envio com token invalido (controle de acesso).
    print("\nTentativa com token invalido:")
    try:
        mediador.enviar(Mensagem(
            origem="usuario",
            destino="FinancialTracker",
            acao="classificar_sms",
            payload={"sms": "qualquer coisa"},
            token="token_falso",
        ))
    except PermissionError as exc:
        print(f"  Mediador negou o roteamento -> {exc}")

    # ----------------------------------------------------------------- 4
    banner("4. Regras de associacao aprendidas pelo Apriori")
    # O atributo _regras e um DataFrame ja ordenado por confianca decrescente
    # (mining.py faz a ordenacao no final do .treinar()).
    if clf._regras.empty:
        print("Nenhuma regra aprendida (verifique min_support/min_confidence).")
    else:
        print(f"{'Antecedente':<24}{'Categoria':<14}{'Conf.':>7}{'Sup.':>8}")
        print("-" * 53)
        for _, linha in clf._regras.head(10).iterrows():
            antecedente = sorted(
                item for item in linha["antecedents"]
                if not item.startswith(PREFIXO_CAT)
            )
            ante_str = ", ".join(antecedente) if antecedente else "-"
            categoria = next(iter(linha["consequents"])).replace(PREFIXO_CAT, "")
            print(f"{ante_str[:23]:<24}{categoria:<14}"
                  f"{linha['confidence']:>6.2f} {linha['support']:>7.3f}")

    # ----------------------------------------------------------------- log
    banner("Trilha de auditoria do Mediador (ultimas 8 mensagens)")
    for entrada in mediador.log[-8:]:
        print(" ", entrada)


if __name__ == "__main__":
    main()