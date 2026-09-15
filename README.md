# 📈 FinSignal AI — Análise de Sentimento Financeiro com Transformers

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-Hugging%20Face-orange)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/Licen%C3%A7a-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pipeline ponta a ponta de Processamento de Linguagem Natural (PLN / NLP) desenvolvido para quantificar o sentimento do mercado financeiro a partir de dados não estruturados (notícias e relatórios corporativos) utilizando algorito de aprendizado profundo (Transformers / FinBERT / BERTimbau).

---

## 📌 Resumo

O **FinSignal AI** simula um motor de inteligência para o setor FinTech/B2B. No mercado financeiro, o volume diário de notícias econômicas, decisões regulatórias e relatórios de resultados ultrapassa a capacidade analítica humana. Este projeto aborda esse gargalo ao converter textos financeiros não estruturados em sinais quantitativos de sentimento (Positivo, Neutro, Negativo), permitindo a automação do scoring de risco e o alinhamento ágil de carteiras de investimento.

---

## 📊 Conjunto de Dados & Metadados

O projeto utiliza o dataset FinancialPhraseBank ([takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank)), um benchmark utilizado em PLN financeiro, criado pela Hanken School of Economics e publicado no Journal of the Association for Information Science and Technology (JASIST).

* Modalidade dos Dados: Textual (Linguagem Natural Não Estruturada)
* Volume: 4.840 sentenças extraídas de notícias financeiras e relatórios corporativos
* Anotação: Rotulagem humana realizada por 5 a 8 especialistas do mercado financeiro

### Dicionário de Metadados

| Nome da Variável | Tipo de Dado | Descrição do Atributo | Função no Projeto |
| :--- | :--- | :--- | :--- |
| **`sentence`** | Textual (`string`) | Fragmento de texto com jargões, dados numéricos ou relatórios do mercado financeiro. | **Variável de Entrada:** Processada via tokenização para inferência da rede Transformer. |
| **`label`** | Categórico (`int64` / `string`) | Rótulo de sentimento atribuído: `0` (Negative), `1` (Neutral), `2` (Positive). | **Variável Alvo:** Variável dependente a ser prevista pelos modelos supervisionados. |
| **`agreement_level`** | Categórico (`string`) | Nível de consenso entre os avaliadores humanos (`allagree`, `75agree`, `66agree`, `50agree`). | **Filtro de Qualidade:** Usado no pré-processamento para garantir a confiabilidade do conjunto de treino. |

---

## 🎯 Objetivos do Projeto & Métricas Meta

* Objetivo Principal: Implementar e avaliar um modelo PLN baseado na arquitetura Transformer (FinBERT / BERTimbau) para classificação automática de sentimento em notícias financeiras.
* Comparação de Baseline: Comparar a acurácia e o F1-Score do Transformer contra baselines clássicos de PLN (TF-IDF + Regressão Logística / XGBoost).

### Métricas Meta de Desempenho
* Acurácia: maior que 80% no conjunto de teste
* Demonstrar um ganho superior em pontos percentuais no F1-Score em relação ao modelo clássico
* Pipeline 100% executável e modular em ambiente Google Colaboratory utilizando aceleração via GPU.

---

## 📅 Cronograma de Atividades

| Etapa | Tarefa | Entregável | Data de Entrega |
| :--- | :--- | :--- | :--- |
| **Etapa 1** | Apresentação da atuação da empresa, seleção dos dados textuais, metadados, objetivos e repositório no GitHub. | Relatório técnico | 16/09/2026 |
| **Etapa 2** | Análise exploratória da base textual, distribuição de classes, verificação de nulos, tokenização e tratamento dos dados. | Script de Pré-processamento | 25/09/2026 |
| **Etapa 3** | Configuração da arquitetura Transformer, fine-tuning do modelo FinBERT e ajuste de hiperparâmetros. | Relatório de Treinamento | 23/10/2026 |
| **Etapa 4** | Avaliação final, cálculo de métricas finais, documentação no GitHub e preparação do relatório final. | Relatório Consolidado, Código Final e Apresentação | 20/11/2026 |

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **PLN e Deep Learning:** Hugging Face `transformers`, `datasets`, PyTorch
* **Processamento e Visualização:** Pandas, NumPy, Matplotlib, Seaborn
* **Ambiente de Desenvolvimento:** Google Colaboratory (GPU Runtime)

---
