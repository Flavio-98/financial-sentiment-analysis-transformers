# FinSignal AI — Análise de Sentimento Financeiro com FinBERT

Pipeline ponta a ponta de Processamento de Linguagem Natural (PLN) desenvolvido para quantificar o sentimento do mercado financeiro a partir de dados não estruturados (notícias e relatórios corporativos) utilizando o modelo de aprendizado profundo **FinBERT** (*Transformer*).

---

## 📌 Resumo & Visão de Negócio
O **FinSignal AI** opera como um motor de inteligência quantitativa voltado ao setor FinTech/B2B. No mercado financeiro, a velocidade na tomada de decisão é crítica; contudo, o volume diário de notícias econômicas, decisões regulatórias e relatórios de resultados ultrapassa a capacidade analítica humana em tempo real. 

Este projeto resolve esse gargalo convertendo textos financeiros não estruturados em sinais quantitativos de sentimento (Positivo, Neutro e Negativo), automatizando o *scoring* de risco e acelerando a tomada de decisão em relatórios e carteiras de investimento.

---

## 📊 Conjunto de Dados & Metadados
O projeto utiliza o *dataset* público **FinancialPhraseBank** (`takala/financial_phrasebank`), *benchmark* em PLN financeiro criado pela Hanken School of Economics e publicado no *Journal of the Association for Information Science and Technology* (JASIST).

Para garantir a confiabilidade dos rótulos e eliminar ambiguidades, utiliza-se o recorte **`Sentences_AllAgree`**, composto por sentenças com 100% de consenso entre os avaliadores humanos.

* **Modalidade dos Dados:** Textual (Linguagem Natural Não Estruturada).
* **Volume do Subconjunto:** 2.264 sentenças extraídas de artigos econômicos, relatórios e demonstrações de resultados.
* **Rotulagem Supervisionada:** Classificação por especialistas do setor financeiro em três categorias.

### Dicionário de Metadados
| Nome da Variável | Tipo de Dado | Descrição do Atributo | Função no Projeto |
| :--- | :--- | :--- | :--- |
| `sentence` | Textual (`string`) | Fragmento de texto com jargões, dados numéricos e relatórios financeiros. | **Variável de Entrada:** Processada via tokenização *WordPiece* para inferência do FinBERT. |
| `label` | Categórico (`int64`) | Rótulo de sentimento: `0` (Negativo), `1` (Neutro), `2` (Positivo). | **Variável Alvo:** Rótulo dependente a ser previsto pelo modelo supervisionado. |
| `agreement_level` | Categórico (`string`) | Nível de consenso entre os avaliadores (`allagree`, `75agree`, `66agree`, `50agree`). | **Filtro de Qualidade:** Utilizado para selecionar a amostra `allagree` (100% de consenso). |

### Distribuição das Classes (`Sentences_AllAgree`)
| Classe | Rótulo | Sentenças | Proporção (%) |
| :--- | :---: | :---: | :---: |
| **Neutro** | `1` | 1.391 | 61,44% |
| **Positivo** | `2` | 570 | 25,18% |
| **Negativo** | `0` | 303 | 13,38% |
| **Total** | — | **2.264** | **100,00%** |

* **Comprimento médio das sentenças:** ~22 palavras por observação.
* **Vocabulário dominante:** *EUR, company, profit, sales, net, mn*.

---

## 🎯 Objetivos & Metas

### Objetivos
* Desenvolver, implementar e avaliar um *pipeline* de Inteligência Artificial baseado no modelo *Transformer* pré-treinado (**FinBERT**) para automação da análise de sentimento textual.
* Realizar Análise Exploratória de Dados (EDA), limpeza de ruídos de formatação e aplicação de **amostragem estratificada** (*80% treino / 20% teste*).
* Executar tokenização apropriada via algoritmo *WordPiece* com tokens de controle (`[CLS]` e `[SEP]`), *padding* e truncamento.
* Avaliar a capacidade de generalização do modelo utilizando um subconjunto de teste estritamente isolado (*held-out dataset*).

### Metas vs. Resultados Alcançados
| Métrica / Meta | Meta Estabelecida | Resultado Alcançado |
| :--- | :---: | :---: |
| **Acurácia Global no Teste** | > 90,00% | **96,91%** |
| **F1-Score Ponderado (*Weighted Avg*)** | > 0,90 | **0,97** |
| **F1-Score Macro (Equilíbrio de Classes)** | > 0,85 | **0,96** |
| **Isolamento do Teste (*Held-Out*)** | 100% independente | Concluído |
| **Execução Modular & Reprodutível** | 100% no Google Colab c/ GPU | Concluído |

---

## 📈 Desempenho Detalhado no Teste (453 Amostras Independente)
A inferência foi executada sobre o subconjunto de teste estratificado (20% da base — **453 sentenças**). O conjunto permaneceu estritamente isolado (*held-out*) durante o processo, garantindo a total independência na avaliação e que os resultados reflitam a real capacidade de generalização em dados inéditos.

| Classe | Precisão | Revocação (*Recall*) | F1-Score | Suporte |
| :--- | :---: | :---: | :---: | :---: |
| **Negativo (0)** | 0,95 | 0,97 | 0,96 | 61 |
| **Neutro (1)** | 1,00 | 0,96 | 0,98 | 278 |
| **Positivo (2)** | 0,91 | 0,99 | 0,95 | 114 |
| **Acurácia Global** | — | — | **96,91%** | **453** |
| **Macro Avg** | 0,95 | 0,97 | **0,96** | 453 |
| **Weighted Avg** | 0,97 | 0,97 | **0,97** | 453 |

---

## 💼 Produto Gerado & Modelo de Negócio

### Produto Gerado (`FinSignal Core Engine`)
O produto desenvolvido é o motor de inteligência **FinSignal AI**, uma solução analítica focada em converter textos financeiros não estruturados (notícias corporativas, relatórios de mercado e fatos relevantes) em sinais numéricos estruturados de sentimento (Positivo, Neutro e Negativo). O *pipeline* realiza a ingestão do texto, aplica a tokenização pelo algoritmo *WordPiece* e classifica a mensagem com base no FinBERT, retornando a análise de sentimento e o grau de confiança da predição.

### Modelo de Negócio (B2B / FinTech)
* **Proposta de Valor:** Redução drástica do tempo de leitura de analistas e gestores, transformando fluxos de notícias em métricas quantitativas de risco e sentimento em tempo real, abrindo a possibilidade de considerar mais informações e favorecer a tomada de decisão e os ajustes de portfólio.
* **Público-Alvo:** Gestoras de fundos de investimento (*buy-side*), mesas de operações (*sell-side*), *fintechs*, departamentos de relações com investidores (RI) e investidores individuais.
* **Estratégia de Monetização:**
  * **API por Demanda (*Pay-per-call*):** Cobrança por volume de requisições ou notícias processadas via integração de API.
  * **Assinatura Enterprise (SaaS):** Plano recorrente mensal para acesso a *dashboards* de monitoramento de carteiras e *feeds* de notícias em tempo real.

---

## 📖 Storytelling

1. **O Problema:** No mercado financeiro o tempo é um fator crítico que pode definir ganhos maiores ou prejuízos em uma operação. Assim, o volume diário de notícias econômicas, relatórios corporativos e regulatórios supera a capacidade humana de leitura, gerando gargalos e atrasos na identificação de riscos e de oportunidades.
2. **Os Dados:** Utilizou-se o *dataset* público *FinancialPhraseBank* (recorte `Sentences_AllAgree` com 2.264 sentenças e 100% de consenso humano). A análise exploratória revelou sentenças curtas (média de 22 palavras), com forte assimetria de classes (61,44% neutras, 25,18% positivas e 13,38% negativas) e vocabulário denso em jargões econômicos.
3. **A Solução:** Implementou-se o motor **FinSignal AI** utilizando o modelo *FinBERT* (*Transformer* pré-treinado para o segmento financeiro). Diferente de modelos genéricos ou estatísticos, o FinBERT compreende a dependência contextual e a semântica bidirecional do texto corporativo via mecanismos de autoatenção.
4. **O Impacto:** Validado em um conjunto de teste independente com 453 amostras *held-out*, o modelo alcançou **96,91% de Acurácia Global** e **0,97 de F1-Score Ponderado**. O produto transforma horas de leitura manual em sinais quantitativos instantâneos via API e *dashboards* para que gestoras, *fintechs* e investidores tomem decisões de forma ágil.

---

## 📅 Cronograma de Atividades

| Etapa | Tarefa | Entregável | Data | Status |
| :--- | :--- | :--- | :---: | :---: |
| **Etapa 1** | Apresentação da atuação da empresa, seleção dos dados textuais, metadados, objetivos e repositório no GitHub. | Relatório técnico | 16/09/2026 | Concluído |
| **Etapa 2** | Análise exploratória da base textual, distribuição de classes, verificação de nulos, tokenização e tratamento dos dados. | Script de Pré-processamento | 25/09/2026 | Concluído |
| **Etapa 3** | Configuração da arquitetura Transformer, fine-tuning do FinBERT, produto preliminar, modelo de negócios e storytelling. | Relatório de Treinamento | 23/10/2026 | **Concluído** |
| **Etapa 4** | Avaliação final, consolidação do repositório, gravação de apresentação em vídeo e relatório final. | Relatório Consolidado & Apresentação | 20/11/2026 | A realizar |

---

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python
* **Requisição & Extração:** `urllib.request`
* **Manipulação & Estruturação:** `pandas`, `NumPy`
* **Análise Léxica & Frequência:** `collections.Counter`
* **Visualização Gráfica:** `matplotlib.pyplot`, `seaborn`
* **Amostragem & Avaliação:** `scikit-learn` (`train_test_split`, `accuracy_score`, `classification_report`)
* **Modelagem & Inferência Transformer:** `transformers` (`pipeline` / FinBERT)
* **Ambiente:** Google Colaboratory com aceleração via GPU
