"""
Dados de exemplo (sinteticos) do STIMA

perfis financeiros, indicadores, regras de linguagem do Especialista, planos de contas
e um gerador de SMS bancarios rotulados.
Tudo aqui é ficticio, criado apenas para demonstrar o sistema
"""

from __future__ import annotations
import random

#============================================================================
#Perfils Financeiros
#============================================================================

PERFIS = [
    ("endividado_critico", "Endividado Critico",
     "Comprometimento alto da renda com dividas e sem reserva."),
    ("endividado_moderado", "Endividado Moderado",
     "Dividas relevantes, mas ainda administraveis."),
    ("equilibrado", "Equilibrado",
     "Contas em dia e inicio de formacao de reserva."),
    ("poupador", "Poupador / Investidor",
     "Sobra de renda e foco em poupanca/investimento."),
]

#============================================================================
#Indicadores (perguntas). Opcoes em ORDEM CRESCENTE de gravidade financeira.
#============================================================================

INDICADORES = [
    ("comprometimento_renda",
     "Quanto da sua renda mensal esta comprometida com dividas?",
     ["ate_10", "de_10_a_30", "de_30_a_50", "acima_50"]),
    ("reserva_emergencia",
     "De quantos meses de despesa e a sua reserva de emergencia?",
     ["acima_6", "de_3_a_6", "de_1_a_3", "nenhuma"]),
    ("pagamento_fatura",
     "Como voce costuma pagar a fatura do cartao de credito?",
     ["total", "parcial", "minimo", "nao_paga"]),
    ("atraso_contas",
     "Com que frequencia voce atrasa o pagamento de contas?",
     ["nunca", "raramente", "as_vezes", "frequentemente"]),
    ("poupanca_mensal",
     "Voce consegue poupar parte da renda no mes?",
     ["sim_regular", "sim_eventual", "nao"]),
    ("uso_rotativo",
     "Com que frequencia usa cheque especial ou credito rotativo?",
     ["nunca", "raramente", "frequentemente"]),
    ("emprestimos_ativos",
     "Quantos emprestimos/financiamentos ativos voce tem?",
     ["nenhum", "um", "dois", "tres_ou_mais"]),
]


#============================================================================
# Regras na linguagem propria do Especialista
# (validadas pelo analisador lexico/sintatico/semantico antes de armazenar)
#============================================================================

REGRAS = [
    'SE comprometimento_renda == "acima_50" ENTAO endividado_critico PESO 3',
    'SE comprometimento_renda == "de_30_a_50" ENTAO endividado_moderado PESO 2',
    'SE comprometimento_renda == "de_10_a_30" ENTAO equilibrado PESO 1',
    'SE comprometimento_renda == "ate_10" ENTAO poupador PESO 1',
    'SE reserva_emergencia == "nenhuma" ENTAO endividado_critico PESO 1',
    'SE reserva_emergencia >= "de_1_a_3" E comprometimento_renda >= "de_30_a_50" ENTAO endividado_critico PESO 2',
    'SE reserva_emergencia == "de_3_a_6" ENTAO equilibrado PESO 1',
    'SE reserva_emergencia == "acima_6" ENTAO poupador PESO 2',
    'SE pagamento_fatura == "minimo" ENTAO endividado_critico PESO 2',
    'SE pagamento_fatura == "nao_paga" ENTAO endividado_critico PESO 3',
    'SE pagamento_fatura == "parcial" ENTAO endividado_moderado PESO 1',
    'SE pagamento_fatura == "total" ENTAO equilibrado PESO 1',
    'SE uso_rotativo == "frequentemente" ENTAO endividado_critico PESO 2',
    'SE atraso_contas == "frequentemente" ENTAO endividado_critico PESO 1',
    'SE atraso_contas == "nunca" ENTAO equilibrado PESO 1',
    'SE poupanca_mensal == "sim_regular" ENTAO poupador PESO 2',
    'SE poupanca_mensal == "nao" E comprometimento_renda >= "de_30_a_50" ENTAO endividado_moderado PESO 1',
    'SE emprestimos_ativos == "tres_ou_mais" ENTAO endividado_critico PESO 1',
    'SE emprestimos_ativos == "nenhum" ENTAO poupador PESO 1',
]


#============================================================================
#Planos de contas por perfil (% da renda por categoria)
#============================================================================

PLANOS = {
    "endividado_critico": {"essenciais": 55, "dividas": 35, "reserva": 5, "lazer": 5},
    "endividado_moderado": {"essenciais": 50, "dividas": 25, "reserva": 15, "lazer": 10},
    "equilibrado": {"essenciais": 50, "dividas": 10, "reserva": 20, "lazer": 20},
    "poupador": {"essenciais": 45, "dividas": 5, "reserva": 30, "lazer": 20},
}


#============================================================================
#Geracao de SMS bancarios sinteticos rotulados (para Apriori e baseline)
#============================================================================

_ESTABELECIMENTOS = {
    "alimentacao": ["EXTRA SUPERMERCADO", "PAO DE ACUCAR", "ASSAI ATACADISTA",
                    "CARREFOUR HIPER", "HORTIFRUTI CENTRAL", "PADARIA BELA", "RESTAURANTE OUTBACK", "BAR DO ZE"],
    "transporte": ["POSTO IPIRANGA", "POSTO SHELL", "UBER VIAGENS",
                   "APP99 CORRIDA", "ESTACIONAMENTO ROTA"],
    "saude": ["DROGASIL", "DROGARIA PACHECO", "FARMACIA POPULAR",
              "LABORATORIO SABIN", "CLINICA ODONTO"],
    "moradia": ["CEB ENERGIA", "CAESB AGUA", "CLARO INTERNET",
                "ALUGUEL IMOVEL", "CONDOMINIO EDIFICIO"],
    "lazer": ["IFOOD PEDIDO", "CINEMARK INGRESSO", "NETFLIX MENSAL",
              "SPOTIFY PREMIUM", "RESTAURANTE OUTBACK", "BAR DO ZE"],
    "dividas": ["PARCELA EMPRESTIMO", "FATURA CARTAO", "FINANCIAMENTO VEICULO",
                "CREDITO PESSOAL", "ACORDO RENEGOCIACAO"],
}
_TEMPLATES = [
    "CARTAO Compra aprovada R$ {valor} {estab}",
    "Compra aprovada R$ {valor} em {estab}",
    "Voce fez uma compra de R$ {valor} no {estab}",
    "Debito R$ {valor} {estab}",
    "Pagamento R$ {valor} {estab} realizado",
]


def gerar_sms_dataset(n_por_categoria: int = 60, seed: int = 42):
    """
    Gerar uma lista de pares (sms e categoria)
    """

    rng = random.Random(seed)
    dados = []

    for categoria, estabs in _ESTABELECIMENTOS.items():
        for _ in range (n_por_categoria):
            estab = rng.choice(estabs)
            valor = f"{rng.uniform(10, 1500):.2f}".replace(".", ",")
            template = rng.choice(_TEMPLATES)
            dados.append((template.format(valor=valor, estab=estab), categoria))
    rng.shuffle(dados)
    return dados