# Tema 7 — Automação Inteligente e Deteção de Falhas

UC 04914 — Introdução à Inteligência Artificial (2026/27)

**Grupo:** nº 32
**Elementos:** André Henriques (nº 148827), Alexandre Alves (nº 148511), Rodrigo Machado (nº 149614), Tiago Lages (nº 149611)

## Descrição

Sistema híbrido (regras + machine learning) para detetar falhas em 12 máquinas de uma linha de produção
simulada. Comparamos três abordagens:

1. **Regras** — o «manual do técnico» (K01–K08) como sistema baseado em conhecimento, com explicação do raciocínio;
2. **Machine learning** — Regressão Logística, Random Forest e SVM, com validação por máquina (sem fuga de informação);
3. **Híbrido** — alarme = regra crítica OU probabilidade do ML acima de um limiar, com os parâmetros
   otimizados por procura heurística (subida da encosta e algoritmo genético) sobre uma função de custo.

Duas tarefas: `falha` (falha no momento) e `falha_nas_proximas_24h` (antecipação).

## Estrutura

```
data/        dados sintéticos fornecidos (leituras, conhecimento do domínio, LEIA-ME)
src/         código reutilizável: dados.py, regras.py, features.py, modelos.py, hibrido.py
notebooks/   01_exploracao, 02_regras, 03_ml, 04_hibrido
figuras/     gráficos usados no relatório
resultados/  previsoes_ml.csv e comparacao_abordagens.csv (gerados pelos notebooks)
```

## Como reproduzir

Testado com Python 3.13 (versões muito recentes, como a 3.15, podem não ter pacotes prontos para algumas bibliotecas).

```bash
pip install -r requirements.txt
jupyter notebook
```

Os comandos correm na pasta raiz do projeto (a que contém `requirements.txt`). No Windows, se o `pip` não for
reconhecido, usar `py -3.13 -m pip install -r requirements.txt` e `py -3.13 -m jupyter notebook`.

**Correr os notebooks por esta ordem** (o 04 usa ficheiros criados pelo 03):

1. `notebooks/01_exploracao.ipynb`
2. `notebooks/02_regras.ipynb`
3. `notebooks/03_ml.ipynb` — cria `resultados/previsoes_ml.csv` (demora 1–2 min)
4. `notebooks/04_hibrido.ipynb` — cria `resultados/comparacao_abordagens.csv`

Os resultados usam sementes fixas (`random_state=42` e sementes fixas na procura). Mesmo assim, podem variar
ligeiramente com as versões das bibliotecas. Os resultados deste repositório foram obtidos com: Python 3.13,
pandas 3.1.0, numpy 2.5.4, scikit-learn 1.9.1 e matplotlib 3.11.2.

## Resultados principais

Poupança de custo face a não ter sistema (parâmetros otimizados em 3/4 das máquinas e testados nas restantes;
híbrido = algoritmo genético):

| Cenário | Regras | ML | Híbrido |
|---|---|---|---|
| falha, custo FN:FP = 5:1 | 29,6% | 31,4% | 50,0% |
| falha, 10:1 | 61,6% | 51,8% | 64,0% |
| falha, 20:1 | 77,6% | 67,6% | 80,1% |
| falha nas 24 h, 1:1 | 13,8% | 71,5% | 71,4% |

No relatório final estão a discussão completa, as limitações e a reflexão ética.

## Dados

Dados sintéticos fornecidos no âmbito da UC (ficheiros `Tema07_...`), em `data/`. Os CSV usam separador `;`,
decimal `,` e codificação cp1252.

## Uso de IA generativa

Documentado no Anexo A do relatório.
