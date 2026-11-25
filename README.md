# Redes Neurais - Residência IA
### Previsão de Churn Bancário com Redes Neurais Artificiais (MLP)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas)
![Status](https://img.shields.io/badge/Status-Concluído-success)

Projeto desenvolvido no âmbito do programa **Residência IA**, com foco na aplicação prática de **Deep Learning / Redes Neurais Artificiais (Multilayer Perceptron - MLP)** em **PyTorch** para predição preventiva de evasão de clientes (*churn*) em uma instituição bancária europeia (`Churn_Modelling.csv`).

---

##  Visão Geral do Projeto

A retenção de clientes é um dos maiores desafios estratégicos do setor bancário, uma vez que o custo de atração de um novo correntista é substancialmente superior ao de retenção de um cliente fidelizado.

Neste projeto:
1. Realizei uma **Análise Exploratória de Dados (EDA)** completa com formulação de hipóteses de negócio.
2. Construí um pipeline de pré-processamento via `ColumnTransformer` (One-Hot, codificação binária e padronização com `StandardScaler` ajustado estritamente no treino).
3. Desenvolvi e treinamos um modelo de referência (**Regressão Logística**) e uma **MLP inicial**.
4. Executei um grid de sintonia com **6 configurações de hiperparâmetros** (variando camadas, neurônios, ativações, taxas de aprendizado, Dropout e regularização L2) utilizando **Early Stopping**.
5. Tratei o desbalanceamento de classes com `class_weight` / `pos_weight` e `SMOTE`, calibrando o limiar operacional de decisão (*threshold tuning*).
6. Avaliei o modelo final **uma única vez no conjunto de teste** (1.500 amostras intocadas) e extraímos a importância dos atributos com **Permutation Feature Importance**.
7. Respondi detalhadamente às questões conceituais teóricas de arquitetura de redes neurais.

---

## Principais Resultados

### Comparativo Final no Conjunto de Teste (1.500 clientes)

| Métrica | Regressão Logística (Baseline) | Rede Neural Final (MLP Regularizada) | Ganho Relativo |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | `0.7710` | **`0.8657`** | **+12,3%** |
| **Recall (Sensibilidade)** | `20.33%` | **`73.11%`** | **+259,6%** (captura quase 4x mais evadidos) |
| **F1-Score** | `0.3054` | **`0.6046`** | **+98,0%** |
| **Precisão** | `59.62%` | **`51.50%`** | Meta $\ge 50\%$ cumprida |
| **Acurácia** | `81.13%` | **`80.47%`** | Estável e representativa |

---

## Arquitetura da Rede Neural Campeã

- **Camada de Entrada:** 12 atributos após pré-processamento.
- **Camada Oculta 1:** 64 neurônios + ativação `ReLU` + `Dropout(0.3)`.
- **Camada Oculta 2:** 32 neurônios + ativação `ReLU` + `Dropout(0.3)`.
- **Camada Oculta 3:** 16 neurônios + ativação `ReLU` + `Dropout(0.3)`.
- **Camada de Saída:** 1 neurônio com função de ativação `Sigmoide` e perda ponderada (`pos_weight = 3.91`).
- **Otimizador:** Adam (`lr = 0.001`, `weight_decay = 1e-4`).
- **Early Stopping:** Monitoramento da perda de validação com restauração dos melhores pesos.

---

## Principais Fatores de Churn (Permutation Importance)

1. **Idade (`Age`):** Fator mais determinante da rede. Clientes acima de 40 anos evadem significativamente mais em busca de serviços patrimoniais.
2. **Número de Produtos (`NumOfProducts`):** Clientes com 2 produtos possuem a menor taxa de churn (7,6%). Clientes com 1 produto (27,7%) ou 3+ produtos (82% a 100%) apresentam altíssimo risco.
3. **Geografia (`Geography_Germany`) e Saldo (`Balance`):** Correntistas na Alemanha possuem o dobro da taxa de evasão em relação a França e Espanha, concentrando clientes com saldos elevados.

---

## Estrutura do Repositório

```text
├── churn_rna.ipynb             # Notebook executado com todos os gráficos e células de texto
├── Churn_Modelling.csv         # Base de dados (10.000 clientes, 14 colunas)
├── modelo_final.pt             # Pesos do modelo campeão treinado em PyTorch
├── requirements.txt            # Dependências e versões exatas utilizadas
├── churn_rna_poxaj.zip         # Pacote zip dos entregáveis oficiais da atividade
├── Tarefa - Redes Neurais.pdf  # Especificação original da atividade acadêmica
└── README.md                   # Documentação completa do projeto
```

---

## Como Executar o Projeto

### Opção 1: Google Colab
1. Acesse o [Google Colab](https://colab.research.google.com/).
2. Faça o upload do notebook `churn_rna.ipynb` e do arquivo `Churn_Modelling.csv`.
3. Execute as células sequencialmente (`Ctrl + F9` ou *Ambiente de Execução > Executar tudo*).

### Opção 2: Localmente
Clone o repositório e configure o ambiente:
```bash
# Clone o repositório
git clone https://github.com/jhessevelyn/redes-neurais-residencia-ia.git
cd redes-neurais-residencia-ia

# Crie e ative um ambiente virtual (opcional, recomendado)
python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt

# Inicie o Jupyter Notebook
jupyter notebook churn_rna.ipynb
```

---

## Respostas às Questões Teóricas

O notebook inclui respostas conceituais aprofundadas sobre:
1. **Sigmoide na saída vs ReLU nas ocultas:** Mapeamento probabilístico contínuo $(0, 1)$ versus mitigação do *vanishing gradient* e aceleração computacional.
2. **Backpropagation e Gradiente Descendente:** Aplicação analítica da regra da cadeia para cálculo de derivadas parciais e atualização orientada de pesos.
3. **Equivalência da rede sem camadas ocultas:** Correspondência matemática com a Regressão Logística e comprovação do limite linear.
4. **Armadilha da acurácia em bases desbalanceadas:** O erro de utilizar modelos nulos (acurácia ilusória de 79,6% com Recall zero).
5. **Custo de Falso Negativo vs Falso Positivo no setor bancário:** Impacto assimétrico e justificativa para redução do limiar de decisão.

---

## Autor
Desenvolvido como atividade prática da disciplina de Redes Neurais Artificiais.
