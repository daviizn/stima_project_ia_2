# STIMA — Sistema Tutor Inteligente Multiagente

Protótipo funcional de um **Sistema Multiagente (SMA)** para **educação
financeira de adultos endividados**, desenvolvido como trabalho final da
disciplina **Inteligência Artificial II** (Ciência da Computação, IESB).

O sistema é uma adaptação acadêmica do STIMA de Souza (2021): um conjunto de
agentes de software coopera para **diagnosticar a situação financeira do
usuário** e **recomendar ações corretivas** (quitar dívidas e formar reserva de
emergência), sem acompanhamento presencial de um consultor.

> Implementação 100% em Python, com armazenamento em memória e interface por
> linha de comando (CLI). Não requer banco de dados nem serviços externos.

---

## 1. Visão geral da arquitetura

Cinco agentes autônomos, com **coordenação centralizada**: nenhum agente fala
diretamente com outro — toda mensagem passa pelo **Mediador**, que autentica,
valida o token, roteia e persiste os dados (protocolo
`msg {origem, destino, token, acao, payload}`).

```
                        +------------------------+
                        |   Mediador (StimaEF)   |
                        | autentica - roteia - DB|
                        +-----------+------------+
                                    |
        +-----------------+---------+---------+------------------+
        |                 |                   |                  |
+---------------+ +---------------+ +-----------------+ +------------------+
| Especialista  | |    Tutor      | |   Estudante     | | FinancialTracker |
| (StimaExpert) | | (StimaTutor)  | |  (StimaEstud)   | |  (mineracao SMS) |
+---------------+ +---------------+ +-----------------+ +------------------+
 cadastra regras   infere perfil     responde/lanca       classifica gastos
 perfis, planos    e gera plano      pede auxilio          (Apriori)
```

### Os cinco agentes

| Agente | Papel | Técnica principal |
|---|---|---|
| **StimaExpert** (Especialista) | Cadastra perfis, indicadores, planos e regras | Analisador léxico/sintático/semântico de uma DSL de regras |
| **StimaTutor** (Tutor) | Infere o perfil do usuário e gera o plano de contas | **Motor de regras** com cálculo de afinidade |
| **StimaEstud** (Estudante) | Interface do usuário (responde, lança, pede auxílio) | Agente reativo |
| **FinancialTracker** (Mineração) | Minera SMS bancários e classifica os gastos | **Apriori** (regras de associação) |
| **StimaEF** (Mediador) | Autenticação, tokens, roteamento e persistência | Coordenação centralizada |

---

## 2. Técnicas de IA implementadas

O protótipo concentra duas técnicas de IA, cada uma no seu agente, e cada uma
com um **método de comparação** (exigência do trabalho):

1. **Inferência de perfil — motor de regras (Tutor).**
   O Especialista escreve regras em uma **linguagem própria** (DSL), por exemplo:
   ```
   SE comprometimento_renda == "acima_50" ENTAO endividado_critico PESO 3
   SE pagamento_fatura == "nao_paga" ENTAO endividado_critico PESO 3
   ```
   As regras passam por **análise léxica, sintática e semântica** antes de serem
   aceitas. O motor soma os pesos das regras disparadas por perfil, normaliza e
   atribui o perfil de **maior afinidade**.
   *Comparação:* um classificador **Naive Bayes** treinado a partir de exemplos.

2. **Classificação de gastos — Apriori (Financial Tracker).**
   O agente faz **mineração de texto** dos SMS bancários (remove valores,
   pontuação e *stopwords*, restando o estabelecimento) e aprende **regras de
   associação** `tokens -> categoria`, classificando cada gasto na categoria de
   **maior confiança**.
   *Comparação:* um **baseline por palavra-chave** (voto majoritário).

---

## 3. Instalação

Requer **Python 3.10+** (testado em 3.12).

```bash
# (opcional) ambiente virtual
python -m venv .venv && source .venv/Scripts/activate/

pip install -r requirements.txt
```

Dependências: `mlxtend` (Apriori), `scikit-learn` (métricas), `pandas`,
`numpy`, `matplotlib`.

---

## 4. Como executar

### 4.1 Demonstração ponta-a-ponta

```bash
python demo.py
```

Simula a jornada completa, com **toda** a comunicação passando pelo Mediador:

0. Inicialização do sistema e do Mediador
1. O Especialista cadastra o conhecimento do domínio (e o analisador rejeita uma regra inválida, demonstrando a validação)
2. O Estudante autentica-se (`aluno01` / `1234`) e responde os 7 indicadores
3. O Tutor infere o perfil e gera o plano de contas (ex.: renda de R$ 4.000)
4. O Financial Tracker minera SMS e classifica os gastos via Apriori
5. O Estudante solicita auxílio ao Tutor
6. Trilha de auditoria das mensagens roteadas pelo Mediador

### 4.2 Experimentos (comparação entre métodos)

```bash
python experiment.py
```

Gera **tabelas (`.csv`)** e **gráficos (`.png`)** na pasta `results/`:

- **Experimento 1 — Apriori × baseline** (classificação de gastos).
  Avalia dois cenários: *Limpo* (SMS sem ruído) e *Ruidoso* (SMS com tokens
  promocionais/genéricos, simulando dados reais). Métricas: acurácia, precisão,
  revocação e F1 (macro), além da matriz de confusão.
- **Experimento 2 — Regras × Naive Bayes** (inferência de perfil).
  As regras do especialista são o **gabarito**; mede-se a **concordância** do
  Naive Bayes com esse gabarito e o tempo médio de inferência.

---

## 5. Resultados resumidos

> Valores reprodutíveis (sementes fixas) gerados por `experiment.py`.

**Experimento 1 — classificação de gastos (conjunto de teste):**

| Cenário | Método | Acurácia | F1 (macro) |
|---|---|---|---|
| Limpo | Apriori (projeto) | 1.00 | 1.00 |
| Limpo | Baseline (palavra-chave) | 1.00 | 1.00 |
| Ruidoso | Apriori (projeto) | **1.00** | **1.00** |
| Ruidoso | Baseline (palavra-chave) | 0.87 | 0.88 |

Em dados limpos os dois métodos acertam tudo; sob ruído, o baseline é enganado
por votos espúrios de tokens irrelevantes, enquanto o Apriori se mantém robusto
ao casar a **regra de maior confiança** — e ainda entrega regras interpretáveis.

**Experimento 2 — inferência de perfil (concordância com o gabarito):**

| Método | Concordância | F1 (macro) | Tempo (ms/inferência) |
|---|---|---|---|
| Regras (especialista) | 1.00 *(por construção)* | 1.00 | ~0.015 |
| Naive Bayes (dados) | **0.88** | 0.74 | ~0.012 |

A acurácia das regras é 1.00 **por construção** (elas definem o gabarito). O
resultado relevante é que um método puramente orientado a dados (Naive Bayes)
reproduz ~88% das decisões do especialista, acertando bem o perfil majoritário,
mas perdendo revocação nos perfis menos frequentes.

---

## 6. Estrutura do projeto

```
stima_project/
├── demo.py              # demonstracao ponta-a-ponta (CLI)
├── experiment.py        # experimentos + tabelas/graficos (results/)
├── requirements.txt
├── README.md
├── results/             # saidas dos experimentos (.csv e .png)
└── stima/
    ├── __init__.py
    ├── protocol.py      # estrutura de Mensagem do protocolo
    ├── domain.py        # Indicador, Perfil, Condicao, Regra, Plano, Transacao
    ├── rule_dsl.py      # DSL de regras: lexer + parser + validacao semantica
    ├── rule_engine.py   # motor de inferencia por regras (afinidade)
    ├── bayes.py         # Naive Bayes (comparacao do Tutor)
    ├── mining.py        # mineracao de SMS + Apriori + baseline
    ├── mediator.py      # Mediador (auth, tokens, roteamento, persistencia)
    ├── agents.py        # os 5 agentes
    └── seed_data.py     # perfis, indicadores, regras, planos e gerador de SMS
```

---

## 7. Mapeamento com o documento do projeto

| Item do projeto (PDF) | Onde está no código |
|---|---|
| 5 agentes e suas responsabilidades | `stima/agents.py` |
| Coordenação centralizada pelo Mediador | `stima/mediator.py` |
| Protocolo `msg {origem, destino, token, acao, payload}` | `stima/protocol.py` |
| Regras do Especialista validadas por analisador sintático | `stima/rule_dsl.py` |
| Motor de inferência do Tutor (afinidade de perfil) | `stima/rule_engine.py` |
| Classificação de gastos por Apriori | `stima/mining.py` |
| Avaliação/experimentos e comparação entre métodos | `experiment.py` |

---

## 8. Observações

- Todos os dados (perfis, regras, SMS) são **sintéticos**, criados apenas para
  demonstrar o funcionamento do sistema.
- A persistência é **em memória**: cada execução parte de um estado limpo.
- Adaptação acadêmica de seis para **cinco agentes** (a interface de banco foi
  incorporada ao Mediador), conforme descrito no documento do projeto.

### Referências
- AGRAWAL, R.; SRIKANT, R. *Fast algorithms for mining association rules.* VLDB, 1994.
- RUSSELL, S.; NORVIG, P. *Inteligência Artificial.* 3. ed. Elsevier, 2013.
- SOUZA, R. M. M. de. *STIMA: Sistema Tutor Inteligente Multiagente para Educação Financeira de Adultos no Brasil.* Tese (Doutorado) — Universidade Presbiteriana Mackenzie, 2021.
