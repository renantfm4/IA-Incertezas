# Tratamento de Incerteza em Inteligência Artificial

Três projetos práticos desenvolvidos na disciplina **FGA0221 – Inteligência Artificial** (Engenharia de Software, Universidade de Brasília) para o terceiro portifólio da disciplina. Cada projeto escolhe um problema concreto, implementa em Python 3 os algoritmos correspondentes, compara-os de forma experimental e gera tabelas, figuras e arquivos de resultados.

**Autor:** Renan Araújo de Souza  
**Professor:** Fabiano Araujo Soares

## Projetos

| # | Projeto | Código | O que faz |
|---|---------|--------|-----------|
| 1 | Redes Bayesianas | `rede_bayesiana.py` | Modela a rede do Alarme (Pearl / Russell & Norvig, Cap. 14) com tabelas de probabilidade condicional e calcula P(Burglary \| JohnCalls, MaryCalls) por inferência exata (enumeração de variáveis) e por inferência aproximada (Rejection Sampling e Gibbs Sampling / MCMC), com 100 a 50.000 amostras, medindo erro absoluto, taxa de aceitação e tempo de execução. |
| 2 | Modelos Ocultos de Markov (HMM) | `modelo_hmm.py` | Resolve o "Mundo do Guarda-Chuva" com filtragem progressiva, suavização retrospectiva (Forward-Backward) e decodificação da trajetória ótima (Viterbi); também aprende os parâmetros de transição e emissão a partir de uma sequência simulada de 200 passos com o algoritmo Baum-Welch (EM). |
| 3 | Filtros de Kalman | `filtro_kalman.py` | Rastreia um móvel em 2D (posição e velocidade nos eixos X e Y) sob ruído gaussiano, comparando a estimativa do Filtro de Kalman linear com o sensor bruto via RMSE, acompanha a convergência da covariância P e do ganho K e faz um estudo de sensibilidade variando a razão entre os ruídos de medição e de processo (R/Q). |

## Estrutura do repositório

```
.
├── README.md
├── rede_bayesiana.py
├── modelo_hmm.py
├── filtro_kalman.py
└── resultados/
    ├── 1-fig1_convergencia_amostragem.png
    ├── 1-fig2_distribuicao_posteriori.png
    ├── 1-fig3_tempo_vs_amostras.png
    ├── 2-fig1_filtragem_vs_suavizacao.png
    ├── 2-fig2_trellis_viterbi.png
    ├── 2-fig3_convergencia_baum_welch.png
    ├── 3-fig1_trajetoria_rastreamento.png
    ├── 3-fig2_convergencia_covariancia_P.png
    ├── 3-fig3_sensibilidade_ruido_QR.png
    ├── resultados-bayesiana.json / .txt
    ├── resultados-hmm.json / .txt
    └── resultados-kalman.json / .txt
```

A pasta `resultados/` já contém as saídas geradas pela última execução de cada projeto. Os nomes das figuras começam com o número do projeto (`1-` a `3-`).

## Requisitos

- Python 3.8 ou superior
- [numpy](https://numpy.org/) e [matplotlib](https://matplotlib.org/)

Todo o resto (`itertools`, `json`, `os`, `random`, `time`...) faz parte da biblioteca padrão do Python.

## Instalação

```powershell
# (opcional) criar e ativar um ambiente virtual
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # Linux / macOS

pip install numpy matplotlib
```

## Como executar

Cada projeto é um script independente. Rode-o a partir da pasta raiz do repositório:

```powershell
python rede_bayesiana.py
python modelo_hmm.py
python filtro_kalman.py
```

Cada execução imprime um resumo no terminal e grava, na pasta `resultados/` (criada automaticamente), as figuras `.png` do projeto e os arquivos `resultados-*.json` e `resultados-*.txt` com os números. O projeto 1 é o mais demorado, pois o Gibbs Sampling recalcula probabilidades por enumeração a cada passo; os outros dois terminam em poucos segundos.

| Projeto | Arquivos gerados em `resultados/` |
|---------|-----------------------------------|
| 1 | `1-fig*.png`, `resultados-bayesiana.json/.txt` |
| 2 | `2-fig*.png`, `resultados-hmm.json/.txt` |
| 3 | `3-fig*.png`, `resultados-kalman.json/.txt` |

## Figuras geradas

| Projeto | Figura | Conteúdo |
|---------|--------|----------|
| 1 | `1-fig1` | Convergência do erro absoluto de Rejection Sampling e Gibbs Sampling em função de N (escala log-log) |
| 1 | `1-fig2` | Distribuição a posteriori de Burglary: exata vs. aproximada (N = 50.000) |
| 1 | `1-fig3` | Tempo de execução vs. número de amostras, com a enumeração exata como referência |
| 2 | `2-fig1` | Probabilidade de chuva ao longo de 5 dias: filtragem vs. suavização |
| 2 | `2-fig2` | Treliça do Viterbi e caminho escolhido |
| 2 | `2-fig3` | Evolução da log-verossimilhança nas iterações do Baum-Welch |
| 3 | `3-fig1` | Trajetória real, medições ruidosas do radar e estimativa do Filtro de Kalman |
| 3 | `3-fig2` | Traço da matriz P e ganho de Kalman ao longo do tempo |
| 3 | `3-fig3` | RMSE da posição estimada para diferentes fatores de ruído de medição R |

## Parâmetros principais

- **Projeto 1:** evidência `JohnCalls = True` e `MaryCalls = True`; consulta sobre `Burglary`; N ∈ {100, 500, 1.000, 5.000, 10.000, 50.000}; *burn-in* do Gibbs de 10% de N (mínimo de 100).
- **Projeto 2:** observações `[T, T, F, T, T]` (guarda-chuva); modelo verdadeiro com transição 0,7/0,3 e emissão 0,9 (chuva) / 0,2 (sem chuva) para guarda-chuva; Baum-Welch iniciado com parâmetros quase uniformes, até 40 iterações.
- **Projeto 3:** 60 passos com dt = 0,5 s; desvio-padrão do ruído de processo 0,1 e do sensor 5,0 m; estimativa inicial com alta incerteza (P₀ = 100·I); fatores de R testados: 0,01, 0,1, 1, 10 e 100.

## Reprodutibilidade

Os códigos usam sementes aleatórias fixas (`random.seed(42)` e `np.random.seed(42)`), então as instâncias geradas e os valores relatados podem ser reproduzidos. Os tempos de execução, porém, dependem da máquina e da versão do Python.

## Relatório

Os problemas, os algoritmos, as tabelas e a análise dos resultados de cada projeto estão descritos no relatório da disciplina.