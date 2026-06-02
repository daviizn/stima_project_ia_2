"""Demonstracao ponta-a-ponta do STIMA.

Simula a jornada completa, com TODA comunicacao passando pelo Mediador
(coordenacao centralizada):

  1. Especialista cadastra perfis, indicadores, planos e regras.
  2. Estudante autentica-se e responde os indicadores.
  3. Tutor infere o perfil e gera o plano de contas personalizado.
  4. Financial Tracker minera SMS bancarios e classifica os gastos (Apriori).
  5. Estudante solicita auxilio ao Tutor.

Execucao:  python demo.py
"""
from stima import seed_data
from stima.agents import FinancialTracker, StimaEstud, StimaExpert, StimaTutor
from stima.mediator import Mediador
from stima.mining import ClassificadorApriori
from stima.protocol import Mensagem

LINHA = "=" * 66


def banner(titulo: str) -> None:
    print(f"\n{LINHA}\n  {titulo}\n{LINHA}")


def main() -> None:
    # ----------------------------------------------------------------- 0
    banner("0. Inicializacao do sistema e do Mediador (StimaEF)")
    mediador = Mediador()
    mediador.cadastrar_credencial("aluno01", "1234")

    # Treina o classificador do Tracker com SMS sinteticos.
    clf = ClassificadorApriori(suporte_min=0.02, confianca_min=0.3)
    clf.treinar(seed_data.gerar_sms_dataset())

    especialista = StimaExpert("StimaExpert")
    tutor = StimaTutor("StimaTutor")
    estudante = StimaEstud("StimaEstud", usuario="aluno01")
    tracker = FinancialTracker("FinancialTracker", clf)
    for ag in (especialista, tutor, estudante, tracker):
        mediador.registrar_agente(ag)
    print("Agentes registrados:", ", ".join(mediador._agentes))

    token_admin = mediador.autenticar("aluno01", "1234")

    # ----------------------------------------------------------------- 1
    banner("1. Especialista cadastra o conhecimento do dominio")
    for pid, nome, desc in seed_data.PERFIS:
        mediador.enviar(Mensagem("admin", "StimaExpert", "cadastrar_perfil",
                                 {"id": pid, "nome": nome, "descricao": desc},
                                 token=token_admin))
    for iid, perg, opcoes in seed_data.INDICADORES:
        mediador.enviar(Mensagem("admin", "StimaExpert", "cadastrar_indicador",
                                 {"id": iid, "pergunta": perg, "opcoes": opcoes},
                                 token=token_admin))
    for perfil, dist in seed_data.PLANOS.items():
        mediador.enviar(Mensagem("admin", "StimaExpert", "cadastrar_plano",
                                 {"perfil": perfil, "distribuicao": dist},
                                 token=token_admin))
    print(f"Perfis: {len(mediador.banco['perfis'])} | "
          f"Indicadores: {len(mediador.banco['indicadores'])} | "
          f"Planos: {len(mediador.banco['planos'])}")

    print("\nValidando e cadastrando regras (analisador lexico/sintatico):")
    ok = err = 0
    for texto in seed_data.REGRAS:
        r = mediador.enviar(Mensagem("admin", "StimaExpert", "cadastrar_regra",
                                     {"texto": texto}, token=token_admin))
        ok += int(r["ok"])
        err += int(not r["ok"])
    print(f"  {ok} regras validas cadastradas, {err} rejeitadas.")

    ruim = mediador.enviar(Mensagem(
        "admin", "StimaExpert", "cadastrar_regra",
        {"texto": 'SE renda == "muito_alta" ENTAO rico PESO 1'}, token=token_admin))
    print(f"  Exemplo de regra invalida -> rejeitada pelo analisador:")
    print(f"    {ruim['erro']}")

    # ----------------------------------------------------------------- 2
    banner("2. Estudante autentica-se e responde os indicadores")
    token = mediador.autenticar("aluno01", "1234")
    print(f"Token de sessao do estudante: {token}")

    respostas_usuario = {
        "comprometimento_renda": "acima_50",
        "reserva_emergencia": "nenhuma",
        "pagamento_fatura": "minimo",
        "atraso_contas": "as_vezes",
        "poupanca_mensal": "nao",
        "uso_rotativo": "frequentemente",
        "emprestimos_ativos": "dois",
    }
    for ind, val in respostas_usuario.items():
        mediador.enviar(Mensagem("StimaEstud", "StimaEstud", "registrar_resposta",
                                 {"indicador": ind, "valor": val}, token=token))
    print(f"Respostas registradas: {len(estudante.respostas)}/"
          f"{len(mediador.banco['indicadores'])}")

    # ----------------------------------------------------------------- 3
    banner("3. Tutor infere o perfil e gera o plano de contas")
    resultado = mediador.enviar(Mensagem("StimaEstud", "StimaTutor",
                                         "inferir_perfil",
                                         {"respostas": estudante.respostas},
                                         token=token))
    print("Afinidade por perfil:")
    for p, a in sorted(resultado["afinidade"].items(), key=lambda x: -x[1]):
        nome = mediador.banco["perfis"][p].nome
        print(f"  {nome:<24} {a:6.1%}  {'#' * int(a * 40)}")
    perfil = resultado["perfil"]
    print(f"\n>> Perfil atribuido: {mediador.banco['perfis'][perfil].nome}")
    print(f"   Regras disparadas: {len(resultado['regras_disparadas'])}")

    renda = 4000.0
    plano = mediador.enviar(Mensagem("StimaEstud", "StimaTutor", "gerar_plano",
                                     {"perfil": perfil, "renda": renda},
                                     token=token))
    print(f"\nPlano de contas personalizado (renda R$ {renda:.2f}):")
    for cat, val in plano["valores"].items():
        print(f"  {cat:<12} {plano['distribuicao'][cat]:>3.0f}%   R$ {val:>8.2f}")

    # ----------------------------------------------------------------- 4
    banner("4. Financial Tracker minera SMS e classifica gastos (Apriori)")
    sms_exemplos = [
        "CARTAO Compra aprovada R$ 187,90 EXTRA SUPERMERCADO",
        "Compra aprovada R$ 62,00 em POSTO IPIRANGA",
        "Debito R$ 45,90 DROGASIL",
        "Pagamento R$ 980,00 PARCELA EMPRESTIMO realizado",
        "Voce fez uma compra de R$ 54,90 no IFOOD PEDIDO",
        "Debito R$ 350,00 CEB ENERGIA",
    ]
    print(f"{'Categoria':<14}{'Conf.':>6}   SMS")
    print("-" * 66)
    for sms in sms_exemplos:
        r = mediador.enviar(Mensagem("FinancialTracker", "FinancialTracker",
                                     "classificar_sms", {"sms": sms}, token=token))
        t = r["transacao"]
        mediador.enviar(Mensagem(
            "StimaEstud", "StimaEstud", "registrar_lancamento",
            {"descricao": t.descricao, "valor": t.valor, "categoria": t.categoria,
             "fonte": "sms", "confianca": t.confianca}, token=token))
        print(f"{t.categoria:<14}{t.confianca:>5.0%}   {sms[:42]}")
    print(f"\nLancamentos automaticos registrados: {len(estudante.lancamentos)}")

    # mostra algumas regras de associacao aprendidas (le _regras direto)
    top = clf.top_regras(6)
    if not top.empty:
        print("\nExemplos de regras de associacao aprendidas (Apriori):")
        for _, row in top.iterrows():
            print(f"  {row['antecedente']:<18} -> {row['categoria']:<12} "
                  f"(conf={row['confidence']:.2f}, sup={row['support']:.3f})")


    # ----------------------------------------------------------------- 5
    banner("5. Estudante solicita auxilio ao Tutor")
    aux = mediador.enviar(Mensagem("StimaEstud", "StimaTutor", "responder_auxilio",
                                   {"perfil": perfil}, token=token))
    print(f"Tutor: {aux['resposta']}")

    # ----------------------------------------------------------------- log
    banner("Trilha de auditoria do Mediador (ultimas 8 mensagens)")
    for linha in mediador.log[-8:]:
        print(" ", linha)


if __name__ == "__main__":
    main()