# FinSignal AI — Análise de Sentimento Financeiro com FinBERT


Pipeline de Processamento de Linguagem Natural (PLN) que classifica textos financeiros em **Positivo, Neutro ou Negativo** com o FinBERT (`ProsusAI/finbert`), modelo Transformer pré-treinado em textos financeiros. O notebook compara três modelos em tweets financeiros: uma referência (classe majoritária), o FinBERT original e o FinBERT com ajuste fino supervisionado.

---

## 📌 Resumo & Visão de Negócio
A FinSignal AI é uma startup de inteligência quantitativa voltada ao setor financeiro (B2B FinTech / SaaS), que propõe fornecer APIs e dashboards para gestoras de investimentos, corretoras e fintechs. O volume diário de notícias econômicas, decisões regulatórias e fatos relevantes excede o que os analistas conseguem ler, e a empresa propõe resolver esse gargalo com sinais estruturados de sentimento extraídos de textos não estruturados.

Este projeto desenvolve e avalia o modelo de sentimento que serve de base a esse produto.

---

## 📊 Conjunto de Dados & Metadados

| Base | Uso no projeto | Tamanho |
| :--- | :--- | :--- |
| FinancialPhraseBank, recorte `Sentences_AllAgree` (Malo et al., 2014, Aalto University School of Business) | Verificação em domínio do FinBERT original | 2.264 de 4.846 sentenças (consenso de 5 a 8 avaliadores). Teste estratificado de 453 sentenças |
| Twitter Financial News Sentiment (`zeroshot/twitter-financial-news-sentiment`) | Ajuste fino e avaliação | 11.931 tweets (treino e validação originais unidos). 9.544 para ajuste fino e 2.387 para teste |

- **Modalidade dos dados:** textual (linguagem natural não estruturada), em inglês.
- **Rotulagem supervisionada:** três polaridades (Positivo, Neutro e Negativo) nas duas bases.
- **FinancialPhraseBank:** arquivo lido da pasta `FinancialPhraseBank-v1.0/` deste repositório.
- **Divisões:** os dois splits são estratificados, com `random_state=42`.

### Dicionário de metadados — FinancialPhraseBank
| Variável | Tipo | Descrição | Função no projeto |
| :--- | :--- | :--- | :--- |
| `texto` | Textual (`string`) | Sentença financeira em inglês, com jargões, valores numéricos e dados corporativos. | Entrada do modelo, para tokenização e inferência. |
| `rotulo_orig` | Categórico (`string`) | Rótulo original do arquivo: negative, neutral ou positive. | Rótulo de referência, lido do arquivo. |
| `sentimento` | Categórico (`string`) | Rótulo traduzido: Negativo, Neutro ou Positivo. | Variável-alvo usada na avaliação. |

### Dicionário de metadados — Twitter Financial News Sentiment
| Variável | Tipo | Descrição | Função no projeto |
| :--- | :--- | :--- | :--- |
| `text` | Textual (`string`) | Manchete ou tweet financeiro com tickers (ex.: $BYND), jargões e URLs. | Entrada submetida à limpeza leve, à tokenização e ao ajuste fino. |
| `label` | Categórico (`int64`) | 0 (Bearish/Negativo), 1 (Bullish/Positivo), 2 (Neutral/Neutro). | Variável-alvo no ajuste fino e na matriz de confusão. |

### Distribuição das classes
| Classe | FinancialPhraseBank (`Sentences_AllAgree`) | Twitter Financial News |
| :--- | :---: | :---: |
| Neutro | 1.391 (61,44%) | 7.744 (64,91%) |
| Positivo | 570 (25,18%) | 2.398 (20,10%) |
| Negativo | 303 (13,38%) | 1.789 (14,99%) |
| **Total** | **2.264** | **11.931** |

- **FinancialPhraseBank:** média de 22 palavras por sentença. Palavras mais frequentes: EUR, company, profit, net, million e sales.
- **Twitter:** textos curtos, com tickers, nomes de bancos e links, que passaram por limpeza leve (remoção de links e de espaços duplicados), mantendo tickers e pontuação.

---

## 🎯 Objetivos & Metas

### Objetivos
- Desenvolver, implementar e avaliar um pipeline de IA baseado no FinBERT para a análise de sentimento de textos financeiros.
- Realizar a Análise Exploratória de Dados (EDA) e a limpeza leve das duas bases.
- Aplicar a tokenização WordPiece própria do FinBERT.
- Estabelecer um modelo de referência (baseline) como piso comparativo.
- Avaliar o FinBERT original, sem ajuste fino, nas duas bases.
- Realizar o ajuste fino (fine-tuning) supervisionado do FinBERT.
- Avaliar os modelos com Acurácia Global, F1-Score Ponderado, F1-Score Macro e Matriz de Confusão, em conjuntos de teste separados do treino.

### Metas vs. resultados
| Meta | Resultado |
| :--- | :--- |
| Superar a referência em acurácia e F1 ponderado (Twitter) | 86,22% contra 64,89% e 0,8621 contra 0,5108 |
| Superar o FinBERT original em acurácia e F1 macro (Twitter) | 86,22% contra 68,79% e 0,8147 contra 0,6386 |
| F1 macro acima da referência, com métricas por classe | 0,8147 contra 0,2624 |

---

## 🔬 Etapas do notebook
1. Leitura e análise exploratória do FinancialPhraseBank.
2. FinBERT pré-treinado aplicado às 453 sentenças de teste (verificação em domínio).
3. Carga, análise exploratória e limpeza do Twitter Financial News Sentiment, com split estratificado.
4. Avaliação nos 2.387 tweets de teste do modelo de referência (`DummyClassifier`, classe majoritária) e do FinBERT original (truncamento em 128 tokens, lotes de 32).
5. Ajuste fino com `Trainer`: 3 épocas, taxa de aprendizado de 2e-5, lote de 32, weight decay de 0,01 e `max_length` de 128. O checkpoint de maior F1 macro é restaurado ao final.
6. Avaliação final do modelo ajustado: relatório de classificação e matriz de confusão.

---

## 📈 Resultados

### FinancialPhraseBank — 453 sentenças (verificação em domínio)
Acurácia de 96,91%, F1 macro de 0,96 e F1 ponderado de 0,97. O FinBERT foi desenvolvido com essa base.

| Classe | Precisão | Revocação | F1-Score | Suporte |
| :--- | :---: | :---: | :---: | :---: |
| Negativo | 0,95 | 0,97 | 0,96 | 61 |
| Neutro | 1,00 | 0,96 | 0,98 | 278 |
| Positivo | 0,91 | 0,99 | 0,95 | 114 |

### Twitter — 2.387 tweets de teste
| Modelo | Acurácia | F1 macro | F1 ponderado |
| :--- | :---: | :---: | :---: |
| Referência (classe majoritária) | 64,89% | 0,2624 | 0,5108 |
| FinBERT original (sem ajuste fino) | 68,79% | 0,6386 | 0,6983 |
| FinBERT ajustado (Época 2) | 86,22% | 0,8147 | 0,8621 |

**Por classe (precisão / revocação / F1):**

| Classe (suporte) | FinBERT original | FinBERT ajustado |
| :--- | :---: | :---: |
| Negativo (358) | 0,4490 / 0,7626 / 0,5652 | 0,7290 / 0,7514 / 0,7400 |
| Positivo (480) | 0,5863 / 0,5875 / 0,5869 | 0,8074 / 0,7771 / 0,7919 |
| Neutro (1.549) | 0,8374 / 0,7017 / 0,7636 | 0,9100 / 0,9141 / 0,9121 |

**Ajuste fino por época:**

| Época | Loss de treino | Loss de validação | Acurácia | F1 macro |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0,5266 | 0,4199 | 0,8454 | 0,7883 |
| 2 | 0,3020 | 0,3990 | 0,8622 | 0,8147 |
| 3 | 0,1773 | 0,4369 | 0,8584 | 0,8116 |

**Matriz de confusão do modelo ajustado (linhas: classe real; colunas: classe prevista):**

| | Negativo | Positivo | Neutro |
| :--- | :---: | :---: | :---: |
| Negativo | 269 | 21 | 68 |
| Positivo | 35 | 373 | 72 |
| Neutro | 65 | 68 | 1.416 |

---

## 💼 Produto Gerado & Modelo de Negócio

### Produto Gerado
O produto é o modelo de análise de sentimento da FinSignal AI, construído sobre o FinBERT com ajuste fino supervisionado. Ele recebe textos financeiros curtos em inglês, como manchetes e posts em redes sociais, e os classifica em Positivo, Neutro ou Negativo. O texto passa por uma limpeza leve (remoção de links e de espaços duplicados), é tokenizado com o WordPiece do FinBERT, com limite de 128 tokens, e segue para o classificador ajustado sobre a base Twitter Financial News Sentiment.

### Modelo de Negócio (B2B / FinTech)
- **Proposta de valor:** reduzir o tempo que analistas e gestores gastam lendo notícias, transformando o fluxo diário em indicadores de sentimento que permitem acompanhar mais fontes e apoiar decisões de alocação.
- **Público-alvo:** gestoras de fundos (*buy-side*), mesas de operação, *fintechs*, áreas de relações com investidores e investidores individuais.
- **Monetização:** uma API com cobrança por volume de requisições processadas e uma assinatura mensal (SaaS) com *dashboards* de monitoramento de carteiras e *feed* de notícias.

---

## 📖 Storytelling
1. **O Problema:** no mercado financeiro, o tempo pode definir ganho ou prejuízo em uma operação. O volume diário de notícias, relatórios corporativos, fatos relevantes e postagens em redes sociais é maior do que um analista consegue ler, o que atrasa a identificação de riscos e oportunidades.
2. **Os Dados:** duas bases públicas. O FinancialPhraseBank (subconjunto `Sentences_AllAgree`, com 2.264 sentenças de consenso humano total e média de 22 palavras) serviu como referência em domínio. O Twitter Financial News Sentiment (11.931 registros, com tickers de ativos e linguagem informal de mercado) foi usado no ajuste fino, com 9.544 registros, e na avaliação, com 2.387. Nas duas bases a classe neutra predomina (61,44% e 64,91%).
3. **A Solução:** o modelo de sentimento da FinSignal AI, baseado no FinBERT, de arquitetura Transformer, com ajuste fino supervisionado de 3 épocas sobre tweets financeiros. Diferente das abordagens por contagem de palavras (Bag-of-Words e TF-IDF), o FinBERT usa autoatenção para considerar o contexto de cada palavra na frase.
4. **O Impacto:** no FinancialPhraseBank, o FinBERT pré-treinado alcançou 96,91% de acurácia na verificação em domínio. Nos 2.387 tweets de teste, a referência obteve 64,89% de acurácia, o FinBERT original 68,79% e o FinBERT ajustado 86,22% (F1 macro de 0,8147). O ajuste fino elevou a acurácia em 17,43 pontos percentuais em relação ao FinBERT original. O modelo pode ser oferecido por API e dashboards a gestoras e fintechs.

---

## 📅 Cronograma de Atividades

| Etapa | Tarefa | Entregável | Data | Status |
| :--- | :--- | :--- | :---: | :---: |
| Etapa 1 | Apresentação da atuação da empresa, seleção dos dados textuais, metadados, objetivos e repositório no GitHub. | Relatório técnico | 16/09/2026 | Concluído |
| Etapa 2 | Análise exploratória da base textual, distribuição de classes, verificação de nulos, tokenização e tratamento dos dados. | Script de Pré-processamento | 25/09/2026 | Concluído |
| Etapa 3 | Configuração da arquitetura Transformer, ajuste fino (fine-tuning) do FinBERT e configuração dos hiperparâmetros. | Relatório de Treinamento | 23/10/2026 | Concluído |
| Etapa 4 | Avaliação final (com a comparação entre o FinBERT original e o ajustado), consolidação do repositório, gravação da apresentação em vídeo e relatório final. | Relatório Consolidado, Código Final e Apresentação | 20/11/2026 | Concluído |

---

## 🛠️ Tecnologias Utilizadas
- **Linguagem:** Python
- **Ambiente:** Google Colab, com aceleração via GPU
- **Requisição & extração:** `urllib.request`; `datasets` (`load_dataset`, `Dataset`)
- **Manipulação & estruturação:** pandas, NumPy, módulo `re`
- **Análise léxica:** `collections.Counter`
- **Visualização:** matplotlib.pyplot, seaborn
- **Amostragem & avaliação:** scikit-learn (`train_test_split`, `DummyClassifier`, `accuracy_score`, `f1_score`, `classification_report`, `confusion_matrix`)
- **Modelagem Transformer:** `transformers` (`pipeline`, `AutoTokenizer`, `AutoModelForSequenceClassification`, `TrainingArguments`, `Trainer`) e `torch` (verificação de GPU)

---

## ▶️ Como executar
1. Abra o notebook `Projeto2_AnaliseSentimento_MercadoFinanceiro.ipynb` no Google Colab, com GPU.
2. Execute as células em ordem. A primeira instala `datasets` e `transformers`.
3. O arquivo `Sentences_AllAgree.txt` é lido deste repositório, pela URL indicada na célula 2.
4. O ajuste fino leva cerca de 11 minutos em GPU (897 passos). Executar o treinamento de novo pode gerar pequenas variações nas métricas.

## ⚠️ Observação metodológica
Os 2.387 tweets de teste foram usados ao final de cada época para escolher o melhor checkpoint (Época 2) e na avaliação final. A Época 3 teria dado 85,84% de acurácia e 0,8116 de F1 macro.

## 📚 Referências
- ARACI, D. FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv:1908.10063, 2019.
- DEVLIN, J. et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT, 2019.
- MALO, P. et al. Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts. JASIST, v. 65, n. 4, 2014.

---

## 🎥 Apresentação do Projeto

Assista à apresentação completa do **FinSignal AI** (contextualização do problema, arquitetura FinBERT, boas práticas com benchmark, ajuste fino e resultados):

[![Assistir no YouTube](https://img.shields.io/badge/YouTube-Apresenta%C3%A7%C3%A3o%20FinSignal%20AI%20(11min)-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/rlrWYxjMBf0)

🔗 **Link direto:** https://youtu.be/rlrWYxjMBf0
