"""PROJETO 2 - MODELOS OCULTOS DE MARKOV (HMM) (Tratamento de Incerteza em IA)
=============================================================================
Problema: Raciocínio probabilístico temporal no "Mundo do Guarda-Chuva"
(Russell & Norvig, Cap. 14), avaliando:
  - Filtragem progressiva P(X_t | e_1:t)
  - Suavização retrospectiva P(X_k | e_1:t) via Forward-Backward
  - Decodificação de trajetória ótima por Programação Dinâmica (Viterbi)
  - Aprendizado de parâmetros de transição e emissão via Baum-Welch (EM)

Como executar:
    python modelo_hmm.py

Saídas (pasta ./resultados):
  - 2-fig1_filtragem_vs_suavizacao.png
  - 2-fig2_trellis_viterbi.png
  - 2-fig3_convergencia_baum_welch.png
  - resultados-hmm.json
  - resultados-hmm.txt

Dependências: numpy, matplotlib (o restante é biblioteca padrão do Python).
"""

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
# PARTE 1 - ESTRUTURA DO MODELO OCULTO DE MARKOV (HMM)
# =====================================================================
class ModeloOcultoMarkov:
    """Implementação matricial de Processo HMM Discreto de 1ª Ordem."""

    def __init__(self, pi, T, O):
        self.pi = np.array(pi, dtype=float)  # prior inicial [S]
        self.T = np.array(T, dtype=float)  # transição T[i,j] = P(X_t=j | X_{t-1}=i)
        self.O = np.array(O, dtype=float)  # emissão O[i,k] = P(E_t=k | X_t=i)
        self.num_estados = len(pi)

    # ---- 1. Filtragem (Forward Recursion) -----------------------------
    def filtragem(self, observacoes):
        """Calcula P(X_t | e_{1:t}) recursivamente para cada instante t."""
        T_len = len(observacoes)
        f = np.zeros((T_len, self.num_estados))

        # Passo t = 0
        o_0 = observacoes[0]
        f_curr = self.pi * self.O[:, o_0]
        f[0] = f_curr / np.sum(f_curr)

        for t in range(1, T_len):
            o_t = observacoes[t]
            pred = np.dot(f[t - 1], self.T)  # predição
            update = pred * self.O[:, o_t]  # atualização sensorial
            f[t] = update / np.sum(update)
        return f

    # ---- 2. Suavização (Forward-Backward) -----------------------------
    def suavizacao(self, observacoes):
        """Calcula P(X_k | e_{1:t}) combinando passadas progressiva e regressiva."""
        T_len = len(observacoes)
        f = self.filtragem(observacoes)

        # Mensagem reversa beta
        beta = np.ones((T_len, self.num_estados))
        for t in range(T_len - 2, -1, -1):
            o_prox = observacoes[t + 1]
            beta[t] = np.dot(self.T, (self.O[:, o_prox] * beta[t + 1]))
            beta[t] /= np.sum(beta[t])

        suavizados = f * beta
        suavizados /= np.sum(suavizados, axis=1, keepdims=True)
        return suavizados

    # ---- 3. Decodificação Ótima (Viterbi) -----------------------------
    def viterbi(self, observacoes):
        """Determina a trajetória mais provável argmax P(x_{1:t} | e_{1:t})."""
        T_len = len(observacoes)
        delta = np.zeros((T_len, self.num_estados))
        ponteiros = np.zeros((T_len, self.num_estados), dtype=int)

        delta[0] = self.pi * self.O[:, observacoes[0]]

        for t in range(1, T_len):
            o_t = observacoes[t]
            for s in range(self.num_estados):
                seq_probs = delta[t - 1] * self.T[:, s] * self.O[s, o_t]
                melhor_prev = np.argmax(seq_probs)
                delta[t, s] = seq_probs[melhor_prev]
                ponteiros[t, s] = melhor_prev

        # Reconstrução (Traceback)
        melhor_caminho = [int(np.argmax(delta[-1]))]
        for t in range(T_len - 1, 0, -1):
            melhor_caminho.append(int(ponteiros[t, melhor_caminho[-1]]))
        melhor_caminho.reverse()
        return melhor_caminho, delta

    # ---- 4. Treinamento EM (Baum-Welch) ------------------------------
    def baum_welch(self, observacoes, max_iter=40, tol=1e-5):
        """Reestima as matrizes T e O a partir de uma sequência observada."""
        T_len = len(observacoes)
        K = self.O.shape[1]
        log_verossimilhanças = []

        pi_curr = self.pi.copy()
        T_curr = self.T.copy()
        O_curr = self.O.copy()

        for iteracao in range(max_iter):
            # Passo E: Forward
            alpha = np.zeros((T_len, self.num_estados))
            escalas = np.zeros(T_len)
            alpha[0] = pi_curr * O_curr[:, observacoes[0]]
            escalas[0] = np.sum(alpha[0])
            alpha[0] /= escalas[0]

            for t in range(1, T_len):
                alpha[t] = np.dot(alpha[t - 1], T_curr) * O_curr[:, observacoes[t]]
                escalas[t] = np.sum(alpha[t])
                alpha[t] /= escalas[t]

            # Passo E: Backward
            beta = np.zeros((T_len, self.num_estados))
            beta[-1] = 1.0 / escalas[-1]
            for t in range(T_len - 2, -1, -1):
                beta[t] = (
                    np.dot(T_curr, (O_curr[:, observacoes[t + 1]] * beta[t + 1]))
                    / escalas[t]
                )

            # Xi e Gamma
            xi = np.zeros((T_len - 1, self.num_estados, self.num_estados))
            for t in range(T_len - 1):
                o_prox = observacoes[t + 1]
                num = (
                    np.outer(alpha[t], (O_curr[:, o_prox] * beta[t + 1]))
                    * T_curr
                )
                xi[t] = num / np.sum(num)

            gamma = alpha * beta * escalas[:, np.newaxis]
            gamma /= np.sum(gamma, axis=1, keepdims=True)

            # Passo M: Atualização dos parâmetros
            pi_curr = gamma[0]
            T_curr = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, np.newaxis]

            for k in range(K):
                mask = observacoes == k
                if np.sum(mask) > 0:
                    O_curr[:, k] = np.sum(gamma[mask], axis=0) / np.sum(
                        gamma, axis=0
                    )

            ll = float(np.sum(np.log(escalas)))
            log_verossimilhanças.append(ll)
            if (
                iteracao > 0
                and abs(log_verossimilhanças[-1] - log_verossimilhanças[-2])
                < tol
            ):
                break

        return log_verossimilhanças, pi_curr, T_curr, O_curr


# =====================================================================
# PARTE 2 - CONFIGURAÇÃO DO CENÁRIO DO GUARDA-CHUVA
# =====================================================================
# Estados: 0: Chuva (R), 1: Sem Chuva (¬R)
# Observações: 0: Sem Guarda-chuva (F), 1: Com Guarda-chuva (T)
PI_INICIAL = [0.5, 0.5]
T_VERDADE = [[0.7, 0.3], [0.3, 0.7]]
O_VERDADE = [[0.1, 0.9], [0.8, 0.2]]


# =====================================================================
# PARTE 3 - EXECUÇÃO, GRÁFICOS E ARQUIVOS DE SAÍDA
# =====================================================================
def main():
    todos = {}
    cabecalho = "=== Modelo Oculto de Markov: Filtragem, Suavização e Viterbi ==="
    print(cabecalho)
    linhas_saida = [cabecalho]

    hmm = ModeloOcultoMarkov(PI_INICIAL, T_VERDADE, O_VERDADE)
    # Sequência clássica de 5 dias: [T, T, F, T, T]
    observacoes = np.array([1, 1, 0, 1, 1])

    # 1. Inferência Temporal
    t0 = time.perf_counter()
    filt = hmm.filtragem(observacoes)
    t_filt = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    suav = hmm.suavizacao(observacoes)
    t_suav = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    rota_viterbi, delta = hmm.viterbi(observacoes)
    t_vit = (time.perf_counter() - t0) * 1000.0

    msg_inf = (
        f"Observações: [T, T, F, T, T]\n"
        f"P(Chuva) Filtragem: {np.round(filt[:, 0], 4)} (tempo: {t_filt:.3f} ms)\n"
        f"P(Chuva) Suavização: {np.round(suav[:, 0], 4)} (tempo: {t_suav:.3f} ms)\n"
        f"Trajetória Viterbi: {rota_viterbi} (tempo: {t_vit:.3f} ms)"
    )
    print(msg_inf)
    linhas_saida.append(msg_inf)

    # 2. Experimento de Aprendizado com Baum-Welch (EM)
    n_passos_treino = 200
    st_atual = 0 if random.random() < 0.5 else 1
    seq_treino = []
    for _ in range(n_passos_treino):
        obs_k = 1 if random.random() < O_VERDADE[st_atual][1] else 0
        seq_treino.append(obs_k)
        st_atual = 1 if random.random() < T_VERDADE[st_atual][1] else 0
    seq_treino = np.array(seq_treino)

    hmm_treino = ModeloOcultoMarkov(
        [0.5, 0.5], [[0.5, 0.5], [0.5, 0.5]], [[0.4, 0.6], [0.6, 0.4]]
    )
    t0 = time.perf_counter()
    historico_ll, pi_apr, T_apr, O_apr = hmm_treino.baum_welch(
        seq_treino, max_iter=40
    )
    t_em = (time.perf_counter() - t0) * 1000.0

    msg_em = (
        f"Baum-Welch ({len(historico_ll)} iterações): LL Inicial={historico_ll[0]:.2f} "
        f"-> LL Final={historico_ll[-1]:.2f} (tempo: {t_em:.2f} ms)"
    )
    print(msg_em)
    linhas_saida.append(msg_em)

    todos["inferencia_temporal"] = {
        "observacoes": observacoes.tolist(),
        "prob_chuva_filtragem": filt[:, 0].tolist(),
        "prob_chuva_suavizacao": suav[:, 0].tolist(),
        "rota_viterbi": rota_viterbi,
        "tempos_ms": {"filtragem": t_filt, "suavizacao": t_suav, "viterbi": t_vit},
    }
    todos["treinamento_baum_welch"] = {
        "iteracoes": len(historico_ll),
        "historico_log_verossimilhanca": historico_ll,
        "tempo_ms": t_em,
        "T_aprendido": T_apr.tolist(),
        "O_aprendido": O_apr.tolist(),
    }

    # --- Figura 1: Comparativo Temporal Filtragem vs Suavização
    dias = np.arange(1, len(observacoes) + 1)
    plt.figure(figsize=(7, 4.0))
    plt.plot(
        dias,
        filt[:, 0],
        "o-",
        lw=2,
        label=r"Filtragem $P(R_t \mid e_{1:t})$",
        color="#54A24B",
    )
    plt.plot(
        dias,
        suav[:, 0],
        "s-",
        lw=2,
        label=r"Suavização $P(R_t \mid e_{1:5})$",
        color="#4C78A8",
    )
    plt.xticks(dias)
    plt.xlabel("Dia (Passo de Tempo t)")
    plt.ylabel("Probabilidade de Chuva")
    plt.title("2-Fig 1: Comparativo Temporal: Filtragem vs Suavização")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/2-fig1_filtragem_vs_suavizacao.png", dpi=170)
    plt.close()

    # --- Figura 2: Trellis do Algoritmo de Viterbi
    plt.figure(figsize=(7.5, 4.0))
    rotulos_estados = ["Chuva (R)", "Sem Chuva (¬R)"]
    for s_idx in range(2):
        plt.plot(
            dias,
            delta[:, s_idx],
            "o--",
            alpha=0.6,
            label=f"Prob. Máx {rotulos_estados[s_idx]}",
        )
    prob_vit = [delta[t, rota_viterbi[t]] for t in range(len(observacoes))]
    plt.plot(
        dias,
        prob_vit,
        "D-",
        color="#F58518",
        lw=2.5,
        label="Caminho Escolhido (Viterbi)",
    )
    plt.xticks(dias)
    plt.xlabel("Dia (t)")
    plt.ylabel("Probabilidade do Melhor Caminho")
    plt.title("2-Fig 2: Treliça e Seleção de Trajetória do Algoritmo de Viterbi")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/2-fig2_trellis_viterbi.png", dpi=170)
    plt.close()

    # --- Figura 3: Convergência Monótona do Algoritmo Baum-Welch
    plt.figure(figsize=(7, 4.0))
    plt.plot(
        range(1, len(historico_ll) + 1),
        historico_ll,
        "o-",
        color="#B279A2",
        lw=2,
    )
    plt.xlabel("Iteração do Algoritmo EM")
    plt.ylabel("Log-Verossimilhança")
    plt.title("2-Fig 3: Convergência Monótona do Algoritmo Baum-Welch")
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/2-fig3_convergencia_baum_welch.png", dpi=170)
    plt.close()

    # 1. Salvar JSON solicitado
    with open(f"{SAIDA}/resultados-hmm.json", "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)

    # 2. Salvar TXT formatado
    with open(f"{SAIDA}/resultados-hmm.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RELATÓRIO DE RESULTADOS - PROJETO 2: MODELOS OCULTOS DE MARKOV\n")
        f.write("=" * 80 + "\n\n")
        f.write("\n".join(linhas_saida))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("DETALHAMENTO DA INFERÊNCIA TEMPORAL (MUNDO DO GUARDA-CHUVA):\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"{'Dia (t)':<8} | {'Observação':<18} | {'P(Chuva) Filtr.':<18} | {'P(Chuva) Suav.':<18} | {'Viterbi':<10}\n"
        )
        f.write("-" * 80 + "\n")
        obs_nomes = ["Sem Guarda-Chuva", "Com Guarda-Chuva"]
        est_nomes = ["Chuva", "Sem Chuva"]
        for t in range(len(observacoes)):
            f.write(
                f"{t+1:<8} | {obs_nomes[observacoes[t]]:<18} | {filt[t, 0]:<18.4f} | {suav[t, 0]:<18.4f} | {est_nomes[rota_viterbi[t]]:<10}\n"
            )
        f.write("-" * 80 + "\n\n")
        f.write("DESEMPENHO DO ALGORITMO BAUM-WELCH (EM):\n")
        f.write(f" - Iterações executadas: {len(historico_ll)}\n")
        f.write(
            f" - Log-Verossimilhança: {historico_ll[0]:.4f} -> {historico_ll[-1]:.4f}\n"
        )
        f.write(
            f" - Matriz de Transição Estimada:\n{np.array2string(T_apr, precision=4)}\n"
        )
        f.write(
            f" - Matriz de Emissão Estimada:\n{np.array2string(O_apr, precision=4)}\n"
        )

    print(
        f"\nArquivos salvos com sucesso na pasta ./{SAIDA}:\n"
        f"  - 2-fig1_filtragem_vs_suavizacao.png\n"
        f"  - 2-fig2_trellis_viterbi.png\n"
        f"  - 2-fig3_convergencia_baum_welch.png\n"
        f"  - resultados-hmm.json\n"
        f"  - resultados-hmm.txt\n"
    )


if __name__ == "__main__":
    main()