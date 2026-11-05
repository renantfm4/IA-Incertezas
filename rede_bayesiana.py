"""PROJETO 1 - REDES BAYESIANAS (Tratamento de Incerteza em IA)
=====================================================================
Problema: Rede de Alarme de Pearl / Russell & Norvig resolvida com
Inferência Exata (Enumeração de Variáveis) e Inferência Aproximada
(Rejection Sampling e Gibbs Sampling / MCMC) sob diferentes volumes de amostras.

Como executar:
    python rede_bayesiana.py

Saídas (pasta ./resultados):
  - 1-fig1_convergencia_amostragem.png
  - 1-fig2_distribuicao_posteriori.png
  - 1-fig3_tempo_vs_amostras.png
  - resultados-bayesiana.json
  - resultados-bayesiana.txt

Dependências: numpy, matplotlib (o restante é biblioteca padrão do Python).
"""

import itertools
import json
import os
import random
import time

import matplotlib

matplotlib.use("Agg")  # backend sem janela (salva direto em arquivo)
import matplotlib.pyplot as plt
import numpy as np

SAIDA = "resultados"
os.makedirs(SAIDA, exist_ok=True)
random.seed(42)
np.random.seed(42)


# =====================================================================
# PARTE 1 - MODELO DE REDE BAYESIANA TABULAR
# =====================================================================
class RedeBayesiana:
    """Representação de Rede Bayesiana em Grafo Acíclico Dirigido (DAG) com CPTs."""

    def __init__(self):
        # Topologia clássica do alarme (Russell & Norvig, Cap. 14)
        self.variaveis = ["Burglary", "Earthquake", "Alarm", "JohnCalls", "MaryCalls"]
        self.pais = {
            "Burglary": [],
            "Earthquake": [],
            "Alarm": ["Burglary", "Earthquake"],
            "JohnCalls": ["Alarm"],
            "MaryCalls": ["Alarm"],
        }
        # Tabelas de Probabilidade Condicional P(X=True | Pais(X))
        self.cpt = {
            "Burglary": {(): 0.001},
            "Earthquake": {(): 0.002},
            "Alarm": {
                (True, True): 0.95,
                (True, False): 0.94,
                (False, True): 0.29,
                (False, False): 0.001,
            },
            "JohnCalls": {
                (True,): 0.90,
                (False,): 0.05,
            },
            "MaryCalls": {
                (True,): 0.70,
                (False,): 0.01,
            },
        }

    def p_var(self, var, val, atribuicao):
        """Calcula P(var = val | pais) com base na atribuição atual dos pais."""
        valores_pais = tuple(atribuicao[p] for p in self.pais[var])
        p_true = self.cpt[var][valores_pais]
        return p_true if val else (1.0 - p_true)


# =====================================================================
# PARTE 2 - MOTORES DE INFERÊNCIA PROBABILÍSTICA
# =====================================================================
def enumerar_tudo(rede, variaveis, atribuicao):
    """Soma recursivamente sobre todas as variáveis não observadas."""
    if len(variaveis) == 0:
        return 1.0
    Y = variaveis[0]
    resto = variaveis[1:]
    if Y in atribuicao:
        p = rede.p_var(Y, atribuicao[Y], atribuicao)
        return p * enumerar_tudo(rede, resto, atribuicao)
    else:
        total = 0.0
        for val in (True, False):
            atribuicao_copia = dict(atribuicao)
            atribuicao_copia[Y] = val
            p = rede.p_var(Y, val, atribuicao_copia)
            total += p * enumerar_tudo(rede, resto, atribuicao_copia)
        return total


def inferencia_exata(rede, query_var, evidencia):
    """Calcula a distribuição exata P(query_var | evidencia) por enumeração."""
    dist = {}
    for val in (True, False):
        atribuicao = dict(evidencia)
        atribuicao[query_var] = val
        dist[val] = enumerar_tudo(rede, rede.variaveis, atribuicao)
    fator_norm = sum(dist.values())
    return {val: dist[val] / fator_norm for val in dist}


def amostragem_priori(rede):
    """Gera uma amostra completa do mundo a partir das probabilidades a priori."""
    amostra = {}
    for var in rede.variaveis:
        valores_pais = tuple(amostra[p] for p in rede.pais[var])
        prob = rede.cpt[var][valores_pais]
        amostra[var] = bool(random.random() < prob)
    return amostra


def rejection_sampling(rede, query_var, evidencia, n_amostras=10000):
    """Inferência aproximada por amostragem com rejeição de inconsistências."""
    contagens = {True: 0, False: 0}
    aceitas = 0
    for _ in range(n_amostras):
        s = amostragem_priori(rede)
        if all(s[k] == evidencia[k] for k in evidencia):
            contagens[s[query_var]] += 1
            aceitas += 1
    if aceitas == 0:
        return {True: 0.0, False: 0.0}, 0
    return {True: contagens[True] / aceitas, False: contagens[False] / aceitas}, aceitas


def gibbs_sampling(rede, query_var, evidencia, n_amostras=10000, burn_in=1000):
    """Inferência aproximada MCMC via Gibbs Sampling condicionando no Markov Blanket."""
    nao_evidencia = [n for n in rede.variaveis if n not in evidencia]
    estado = dict(evidencia)
    for var in nao_evidencia:
        estado[var] = bool(random.random() < 0.5)

    contagens = {True: 0, False: 0}
    for passo in range(n_amostras + burn_in):
        for Z in nao_evidencia:
            estado_t = dict(estado)
            estado_t[Z] = True
            estado_f = dict(estado)
            estado_f[Z] = False

            p_t = enumerar_tudo(rede, rede.variaveis, estado_t)
            p_f = enumerar_tudo(rede, rede.variaveis, estado_f)
            prob_z_true = p_t / (p_t + p_f)

            estado[Z] = bool(random.random() < prob_z_true)

        if passo >= burn_in:
            contagens[estado[query_var]] += 1

    total = sum(contagens.values())
    return {True: contagens[True] / total, False: contagens[False] / total}


# =====================================================================
# PARTE 3 - EXPERIMENTOS, GRÁFICOS E ARQUIVOS DE SAÍDA
# =====================================================================
def main():
    todos = {}
    rede = RedeBayesiana()
    evidencia = {"JohnCalls": True, "MaryCalls": True}
    query = "Burglary"

    cabecalho = "=== Rede Bayesiana: Inferência Exata vs Amostragem ==="
    print(cabecalho)
    linhas_saida = [cabecalho]

    # 1. Inferência Exata
    t0 = time.perf_counter()
    sol_exata = inferencia_exata(rede, query, evidencia)
    t_exata = (time.perf_counter() - t0) * 1000.0

    msg_exata = (
        f"Evidência: {evidencia}\n"
        f"Solução Exata P({query}=True | e): {sol_exata[True]:.6f} (tempo: {t_exata:.3f} ms)"
    )
    print(msg_exata)
    linhas_saida.append(msg_exata)

    # 2. Avaliação de convergência empírica
    amostras_teste = [100, 500, 1000, 5000, 10000, 50000]
    erros_rej, erros_gibbs = [], []
    tempos_rej, tempos_gibbs = [], []
    taxas_aceite = []
    res_rej_list, res_gibbs_list = [], []

    for n in amostras_teste:
        # Rejection
        t0 = time.perf_counter()
        dist_r, aceitas = rejection_sampling(rede, query, evidencia, n_amostras=n)
        t_r = (time.perf_counter() - t0) * 1000.0
        erros_rej.append(abs(dist_r[True] - sol_exata[True]))
        tempos_rej.append(t_r)
        taxas_aceite.append((aceitas / n) * 100.0)
        res_rej_list.append(dist_r[True])

        # Gibbs (MCMC)
        t0 = time.perf_counter()
        dist_g = gibbs_sampling(
            rede, query, evidencia, n_amostras=n, burn_in=max(100, n // 10)
        )
        t_g = (time.perf_counter() - t0) * 1000.0
        erros_gibbs.append(abs(dist_g[True] - sol_exata[True]))
        tempos_gibbs.append(t_g)
        res_gibbs_list.append(dist_g[True])

    todos["inferencia_exata"] = dict(
        prob_true=sol_exata[True], prob_false=sol_exata[False], tempo_ms=t_exata
    )
    todos["experimentos"] = {
        "n_amostras": amostras_teste,
        "rejection_sampling": {
            "estimativas": res_rej_list,
            "erros": erros_rej,
            "tempos_ms": tempos_rej,
            "taxas_aceite_pct": taxas_aceite,
        },
        "gibbs_sampling": {
            "estimativas": res_gibbs_list,
            "erros": erros_gibbs,
            "tempos_ms": tempos_gibbs,
        },
    }

    # --- Figura 1: Convergência do Erro Absoluto (escala log-log)
    plt.figure(figsize=(7, 4.2))
    plt.plot(
        amostras_teste,
        erros_rej,
        "o-",
        lw=2,
        label="Rejection Sampling",
        color="#F58518",
    )
    plt.plot(
        amostras_teste,
        erros_gibbs,
        "s-",
        lw=2,
        label="Gibbs Sampling (MCMC)",
        color="#4C78A8",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Número de Amostras (N)")
    plt.ylabel("Erro Absoluto |P_est - P_exata|")
    plt.title("1-Fig 1: Convergência do Erro das Abordagens de Amostragem")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/1-fig1_convergencia_amostragem.png", dpi=170)
    plt.close()

    # --- Figura 2: Distribuição a Posteriori Comparada
    labels = [
        "Exata",
        f"Rejection\n(N={amostras_teste[-1]})",
        f"Gibbs\n(N={amostras_teste[-1]})",
    ]
    vals_t = [sol_exata[True], res_rej_list[-1], res_gibbs_list[-1]]
    vals_f = [sol_exata[False], 1.0 - res_rej_list[-1], 1.0 - res_gibbs_list[-1]]
    x = np.arange(len(labels))
    w = 0.35

    plt.figure(figsize=(6.5, 4.0))
    r1 = plt.bar(
        x - w / 2, vals_t, w, label="P(Burglary = True)", color="#E45756"
    )
    r2 = plt.bar(
        x + w / 2, vals_f, w, label="P(Burglary = False)", color="#54A24B"
    )
    plt.xticks(x, labels, fontsize=9)
    plt.ylabel("Probabilidade")
    plt.ylim(0, 1.05)
    plt.title("1-Fig 2: Comparação da Distribuição a Posteriori Obtida")
    for rect in list(r1) + list(r2):
        plt.annotate(
            f"{rect.get_height():.3f}",
            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.legend(fontsize=9)
    plt.grid(True, axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/1-fig2_distribuicao_posteriori.png", dpi=170)
    plt.close()

    # --- Figura 3: Tempo de Execução vs Amostras
    plt.figure(figsize=(7, 4.2))
    plt.plot(
        amostras_teste,
        tempos_rej,
        "o-",
        lw=2,
        label="Rejection Sampling",
        color="#F58518",
    )
    plt.plot(
        amostras_teste,
        tempos_gibbs,
        "s-",
        lw=2,
        label="Gibbs Sampling",
        color="#4C78A8",
    )
    plt.axhline(
        t_exata,
        color="#E45756",
        linestyle="--",
        label=f"Enumeração Exata ({t_exata:.2f} ms)",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Número de Amostras (N)")
    plt.ylabel("Tempo de Execução (ms)")
    plt.title("1-Fig 3: Custo Computacional vs Tamanho Amostral")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/1-fig3_tempo_vs_amostras.png", dpi=170)
    plt.close()

    # 1. Salvar JSON solicitado
    with open(f"{SAIDA}/resultados-bayesiana.json", "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)

    # 2. Salvar TXT com a saída formatada do console e dados tabulares
    with open(f"{SAIDA}/resultados-bayesiana.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RELATÓRIO DE RESULTADOS - PROJETO 1: REDES BAYESIANAS\n")
        f.write("=" * 80 + "\n\n")
        f.write("\n".join(linhas_saida))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("DETALHAMENTO POR MÉTODO DE INFERÊNCIA E NÚMERO DE AMOSTRAS:\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"{'Amostras':<10} | {'P(Rej)':<10} | {'Erro Rej':<10} | {'Aceite (%)':<10} | {'Tempo Rej':<10} | {'P(Gibbs)':<10} | {'Erro Gibbs':<10} | {'Tempo Gibbs':<10}\n"
        )
        f.write("-" * 92 + "\n")
        for i, n in enumerate(amostras_teste):
            f.write(
                f"{n:<10} | {res_rej_list[i]:<10.4f} | {erros_rej[i]:<10.4f} | {taxas_aceite[i]:<10.2f} | {tempos_rej[i]:<7.2f} ms | {res_gibbs_list[i]:<10.4f} | {erros_gibbs[i]:<10.4f} | {tempos_gibbs[i]:<7.2f} ms\n"
            )
        f.write("-" * 92 + "\n")

    print(
        f"\nArquivos salvos com sucesso na pasta ./{SAIDA}:\n"
        f"  - 1-fig1_convergencia_amostragem.png\n"
        f"  - 1-fig2_distribuicao_posteriori.png\n"
        f"  - 1-fig3_tempo_vs_amostras.png\n"
        f"  - resultados-bayesiana.json\n"
        f"  - resultados-bayesiana.txt\n"
    )


if __name__ == "__main__":
    main()