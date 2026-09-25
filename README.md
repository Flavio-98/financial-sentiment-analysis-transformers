# FinSignal AI — Análise de Sentimento Financeiro com FinBERT

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

Pipeline ponta a ponta de Processamento de Linguagem Natural (*PLN*) desenvolvido para quantificar o sentimento do mercado financeiro a partir de dados não estruturados (notícias e relatórios corporativos) utilizando o modelo de aprendizado profundo **FinBERT** (*Transformer*).

---

## 📌 Resumo

O **FinSignal AI** opera como um motor de inteligência quantitativa voltado ao setor FinTech/B2B. No mercado financeiro, o volume diário de notícias econômicas, decisões regulatórias e relatórios de resultados ultrapassa a capacidade analítica humana em tempo real. Este projeto resolve esse gargalo convertendo textos financeiros não estruturados em sinais quantitativos de sentimento (*Positivo*, *Neutro* e *Negativo*), automatizando o *scoring* de risco e acelerando a tomada de decisão em relatórios e carteiras de investimento.

---

## 📊 Conjunto de Dados & Metadados

O projeto utiliza o *dataset* público **FinancialPhraseBank** (`takala/financial_phrasebank`), *benchmark* em *PLN* financeiro criado pela *Hanken School of Economics* e publicado no *Journal of the Association for Information Science and Technology* (JASIST). 

Para garantir a confiabilidade dos rótulos e eliminar ambiguidades, o projeto utiliza o subconjunto **`Sentences_AllAgree`**, composto por sentenças com 100% de consenso entre os avaliadores humanos.

* **Modalidade dos Dados:** Textual (Linguagem Natural Não Estruturada).
* **Volume do Subconjunto:** 2.264 sentenças extraídas de artigos econômicos, relatórios e demonstrações de resultados.
* **Rotulagem Supervisionada:** Classificação por especialistas do setor financeiro em três categorias.

### Dicionário de Metadados

| Nome da Variável | Tipo de Dado | Descrição do Atributo | Função no Projeto |
| :--- | :--- | :--- | :--- |
| `sentence` | Textual (`string`) | Fragmento de texto com jargões, dados numéricos e relatórios financeiros. | **Variável de Entrada:** Processada via tokenização *WordPiece* para inferência do *FinBERT*[cite: 1]. |
| `label` | Categórico (`int64` / `string`) | Rótulo de sentimento: 0 (*Negative*), 1 (*Neutral*), 2 (*Positive*). | **Variável Alvo:** Rótulo dependente a ser previsto pelo modelo supervisionado. |
| `agreement_level` | Categórico (`string`) | Nível de consenso entre os avaliadores (`allagree`, `75agree`, `66agree`, `50agree`). | **Filtro de Qualidade:** Utilizado para selecionar a amostra `allagree` (100% de consenso). |

### Distribuição das Classes (`Sentences_AllAgree`)

| Classe | Rótulo | Sentenças | Proporção (%) |
| :--- | :---: | :---: | :---: |
| **Neutro** | 1 | 1.391 | 61,44% |
| **Positivo** | 2 | 570 | 25,18% |
| **Negativo** | 0 | 303 | 13,38% |
| **Total** | — | **2.264** | **100,00%** |

* **Comprimento médio das sentenças:** ~22 palavras por observação.
* **Vocabulário dominante:** *EUR*, *company*, *profit*, *sales*, *net*, *mn*.

---

## 🎯 Objetivos & Metas

### Objetivos
* Implementar e avaliar um pipeline de inteligência artificial baseado no modelo *Transformer* pré-treinado (**FinBERT**) para automação da análise de sentimento textual.
* Realizar análise exploratória detalhada, tratar ruídos e aplicar amostragem estratificada (80% treino / 20% teste).
* Executar tokenização apropriada com tokens de controle (`[CLS]` e `[SEP]`), *padding* e truncamento.

### Metas vs. Resultados Alcançados

| Métrica / Meta | Meta Estabelecida | Resultado Alcançado |
| :--- | :---: | :---: |
| **Acurácia Global no Teste** | > 80,00%[cite: 1] | **96,91%** |
| **F1-Score Ponderado (*Weighted Avg*)** | > 0,85 | **0,97** |
| **F1-Score Macro** | Balanço entre classes | **0,96** |
| **Execução Modular** | 100% em Google Colab c/ GPU | **Concluído** |

---

## 📈 Desempenho Detalhado no Teste (453 Amostras)

A inferência foi executada sobre o subconjunto de teste estratificado (20% da base):

| Classe | Precisão | Revocação (*Recall*) | F1-Score | Suporte |
| :--- | :---: | :---: | :---: | :---: |
| **Negativo (0)** | 0,95 | 0,97 | 0,96 | 61 |
| **Neutro (1)** | 1,00 | 0,96 | 0,98 | 278 |
| **Positivo (2)** | 0,91 | 0,99 | 0,95 | 114 |
| **Acurácia Global** | — | — | **0,97 (96,91%)** | **453** |
| **Macro Avg** | 0,95 | 0,97 | **0,96** | 453 |
| **Weighted Avg** | 0,97 | 0,97 | **0,97** | 453 |

---

## 📅 Cronograma de Atividades

| Etapa | Tarefa | Entregável | Data |
| :--- | :--- | :--- | :--- |
| **Etapa 1** | Apresentação da atuação da empresa, seleção dos dados textuais, metadados, objetivos e repositório no GitHub[cite: 1]. | Relatório técnico[cite: 1] | 16/09/2026 |
| **Etapa 2** | Análise exploratória da base textual, distribuição de classes, verificação de nulos, tokenização e tratamento dos dados[cite: 1]. | Script de Pré-processamento | 25/09/2026 |
| **Etapa 3** | Configuração da arquitetura Transformer, fine-tuning do modelo FinBERT e ajuste de hiperparâmetros[cite: 1]. | Relatório de Treinamento[cite: 1] | 23/10/2026 |
| **Etapa 4** | Avaliação final, cálculo de métricas finais, documentação no GitHub e preparação do relatório final[cite: 1]. | Relatório Consolidado, Código Final e Apresentação | 20/11/2026 |

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

---
