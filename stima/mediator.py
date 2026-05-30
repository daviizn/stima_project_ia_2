"""Agente Mediador/Integrador (StimaEF).

Ponto unico de orquestracao (coordenacao centralizada, seccao 6.3): autentica
atores, emite e valida tokens de sessao, roteia mensagens entre os agentes e
persiste os dados (aqui, em memoria). Nenhum agente fala diretamente com
outro — tudo passa por aqui, o que simplifica o controle de acesso e a
auditoria das trocas.
"""