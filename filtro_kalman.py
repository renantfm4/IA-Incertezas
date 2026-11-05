"""PROJETO 3 - FILTROS DE KALMAN (Tratamento de Incerteza em IA)
===================================================================
Problema: Rastreamento Cinemático 2D de um móvel sob ruído gaussiano
(posição e velocidade nos eixos X e Y), avaliando:
  - Ciclo recursivo ótimo de Predição e Atualização
  - Redução de erro quadrático médio (RMSE) em relação ao sensor bruto
  - Convergência da matriz de covariância de erro P e Ganho de Kalman K
  - Estudo de sensibilidade frente a variações na razão de ruídos R/Q

Como executar:
    python filtro_kalman.py

Saídas (pasta ./resultados):
  - 3-fig1_trajetoria_rastreamento.png
  - 3-fig2_convergencia_covariancia_P.png
  - 3-fig3_sensibilidade_ruido_QR.png
  - resultados-kalman.json
  - resultados-kalman.txt

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
# PARTE 1 - FILTRO DE KALMAN LINEAR DISCRETO
# =====================================================================
class FiltroKalman:
    """Estimador linear ótimo para sistemas dinâmicos sob ruído gaussiano."""

    def __init__(self, A, B, H, Q, R, x0, P0):
        self.A = np.array(A, dtype=float)  # matriz de transição de estados
        self.B = np.array(B, dtype=float) if B is not None else None  # controle
        self.H = np.array(H, dtype=float)  # matriz de observação (sensor)
        self.Q = np.array(Q, dtype=float)  # covariância do ruído de processo
        self.R = np.array(R, dtype=float)  # covariância do ruído de medição
        self.x = np.array(x0, dtype=float)  # estimativa do estado atual
        self.P = np.array(P0, dtype=float)  # covariância do erro de estimativa

    def predizer(self, u=None):
        """Etapa de Predição a priori: propaga o estado e extrapola incerteza."""
        if u is not None and self.B is not None:
            self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        else:
            self.x = np.dot(self.A, self.x)
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q
        return self.x, self.P

    def atualizar(self, z):
        """Etapa de Correção a posteriori: calcula Ganho K e absorve medição."""
        z = np.array(z, dtype=float)
        y = z - np.dot(self.H, self.x)  # inovação (resíduo de medição)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # covariância da inovação
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))  # Ganho ótimo de Kalman

        # Atualização do estado e da covariância
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        return self.x, self.P, K


# =====================================================================
# PARTE 2 - CONFIGURAÇÃO DO SISTEMA CINEMÁTICO 2D
# =====================================================================
def configurar_modelo(dt=0.5, sigma_proc=0.1, sigma_meas=5.0):
    """Gera matrizes do espaço de estados para velocidade constante em 2D."""
    # Vetor de estado: x = [pos_x, pos_y, vel_x, vel_y]^T
    A = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    )
    B = np.zeros((4, 1))
    # Sensor mede diretamente apenas a posição 2D
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    Q = np.eye(4) * (sigma_proc**2)
    R = np.eye(2) * (sigma_meas**2)
    return A, B, H, Q, R


# =====================================================================
# PARTE 3 - EXECUÇÃO, GRÁFICOS E ARQUIVOS DE SAÍDA
# =====================================================================
def main():
    todos = {}
    cabecalho = "=== Filtro de Kalman: Rastreamento Cinemático 2D ==="
    print(cabecalho)
    linhas_saida = [cabecalho]

    dt = 0.5
    passos = 60
    sigma_q = 0.1
    sigma_r = 5.0
    A, B, H, Q, R = configurar_modelo(dt, sigma_q, sigma_r)

    # Condição real e estimativa inicial com alta incerteza
    x_real = np.array([0.0, 0.0, 12.0, 8.0])
    x0_chute = np.array([0.0, 0.0, 0.0, 0.0])
    P0_chute = np.eye(4) * 100.0

    kf = FiltroKalman(A, B, H, Q, R, x0_chute, P0_chute)

    hist_real = []
    hist_medicoes = []
    hist_estimado = []
    hist_traco_p = []
    hist_ganho_k = []

    curr_real = x_real.copy()

    t0 = time.perf_counter()
    for _ in range(passos):
        # 1. Simulação do processo físico com ruído térmico/vento
        ruido_w = np.random.multivariate_normal(np.zeros(4), Q)
        curr_real = np.dot(A, curr_real) + ruido_w
        hist_real.append(curr_real.copy())

        # 2. Leitura ruidosa do sensor radar
        ruido_v = np.random.multivariate_normal(np.zeros(2), R)
        z = np.dot(H, curr_real) + ruido_v
        hist_medicoes.append(z.copy())

        # 3. Ciclo de estimação de Kalman
        kf.predizer()
        est_x, est_P, K = kf.atualizar(z)

        hist_estimado.append(est_x.copy())
        hist_traco_p.append(float(np.trace(est_P)))
        hist_ganho_k.append(float(K[0, 0]))
    t_ciclo = (time.perf_counter() - t0) * 1000.0

    hist_real = np.array(hist_real)
    hist_medicoes = np.array(hist_medicoes)
    hist_estimado = np.array(hist_estimado)

    # Avaliação do Erro Médio Quadrático (RMSE)
    rmse_sensor = float(
        np.sqrt(np.mean((hist_real[:, :2] - hist_medicoes) ** 2))
    )
    rmse_kalman = float(
        np.sqrt(np.mean((hist_real[:, :2] - hist_estimado[:, :2]) ** 2))
    )
    rmse_vel = float(
        np.sqrt(np.mean((hist_real[:, 2:] - hist_estimado[:, 2:]) ** 2))
    )
    reducao_pct = ((rmse_sensor - rmse_kalman) / rmse_sensor) * 100.0

    msg_kalman = (
        f"Simulação: {passos} passos (dt={dt}s) | Tempo Total: {t_ciclo:.2f} ms ({t_ciclo/passos:.4f} ms/passo)\n"
        f"RMSE Posição - Sensor Bruto: {rmse_sensor:.4f} m\n"
        f"RMSE Posição - Filtro Kalman: {rmse_kalman:.4f} m (Redução de {reducao_pct:.1f}%)\n"
        f"RMSE Velocidade (Estado Oculto): {rmse_vel:.4f} m/s\n"
        f"Incerteza Inicial (Traço P): {hist_traco_p[0]:.2f} -> Incerteza Final: {hist_traco_p[-1]:.4f}"
    )
    print(msg_kalman)
    linhas_saida.append(msg_kalman)

    # 4. Estudo de Sensibilidade da Razão de Ruído R/Q
    fatores_r = [0.01, 0.1, 1.0, 10.0, 100.0]
    sens_rmse = []
    for f in fatores_r:
        R_mod = R * f
        kf_teste = FiltroKalman(A, B, H, Q, R_mod, x0_chute, P0_chute)
        est_teste = []
        for step_idx in range(passos):
            kf_teste.predizer()
            ex, _, _ = kf_teste.atualizar(hist_medicoes[step_idx])
            est_teste.append(ex.copy())
        est_teste = np.array(est_teste)
        err = float(
            np.sqrt(np.mean((hist_real[:, :2] - est_teste[:, :2]) ** 2))
        )
        sens_rmse.append(err)

    todos["parametros"] = {
        "dt": dt,
        "passos": passos,
        "sigma_proc": sigma_q,
        "sigma_meas": sigma_r,
    }
    todos["metricas_desempenho"] = {
        "rmse_sensor_bruto": rmse_sensor,
        "rmse_filtro_kalman": rmse_kalman,
        "reducao_erro_pct": reducao_pct,
        "rmse_velocidade": rmse_vel,
        "tempo_total_ms": t_ciclo,
        "tempo_por_passo_ms": t_ciclo / passos,
        "traco_P_final": hist_traco_p[-1],
        "ganho_K_final": hist_ganho_k[-1],
    }
    todos["estudo_sensibilidade_R"] = {
        "fatores_multiplicadores": fatores_r,
        "rmse_resultante": sens_rmse,
    }

    # --- Figura 1: Trajetória 2D
    plt.figure(figsize=(7.5, 4.4))
    plt.plot(
        hist_real[:, 0],
        hist_real[:, 1],
        "-",
        color="#2ca02c",
        lw=2.5,
        label="Trajetória Real (Ground Truth)",
    )
    plt.scatter(
        hist_medicoes[:, 0],
        hist_medicoes[:, 1],
        color="#d62728",
        s=18,
        alpha=0.45,
        label="Medições Ruidosas (Sensor Radar)",
    )
    plt.plot(
        hist_estimado[:, 0],
        hist_estimado[:, 1],
        "--",
        color="#1f77b4",
        lw=2.0,
        label="Estimativa Suave (Filtro de Kalman)",
    )
    plt.xlabel("Coordenada X (m)")
    plt.ylabel("Coordenada Y (m)")
    plt.title("3-Fig 1: Rastreamento Cinemático 2D no Espaço de Estados")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/3-fig1_trajetoria_rastreamento.png", dpi=170)
    plt.close()

    # --- Figura 2: Convergência da Incerteza (Traço de P) e Ganho K
    fig, ax1 = plt.subplots(figsize=(7.5, 4.0))
    ax1.plot(
        range(passos),
        hist_traco_p,
        color="#c0392b",
        lw=2,
        label="Traço da Matriz P_k (Incerteza)",
    )
    ax1.set_xlabel("Passo de Tempo (k)")
    ax1.set_ylabel(
        r"Traço de P ($m^2$ e $m^2/s^2$)", color="#c0392b", fontsize=9
    )
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.grid(True, ls="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(
        range(passos),
        hist_ganho_k,
        color="#2980b9",
        lw=2,
        ls="--",
        label="Ganho de Kalman K[0,0]",
    )
    ax2.set_ylabel("Ganho K", color="#2980b9", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    plt.title("3-Fig 2: Convergência da Incerteza e Estabilização do Ganho K")
    fig.tight_layout()
    plt.savefig(f"{SAIDA}/3-fig2_convergencia_covariancia_P.png", dpi=170)
    plt.close()

    # --- Figura 3: Sensibilidade ao Erro frente ao Balanço R / Q
    plt.figure(figsize=(7, 4.0))
    plt.plot(
        fatores_r,
        sens_rmse,
        "o-",
        lw=2,
        color="#8e44ad",
        label="Estimador Kalman",
    )
    plt.axhline(
        rmse_sensor,
        color="#e74c3c",
        ls="--",
        label=f"Sensor Bruto ({rmse_sensor:.2f} m)",
    )
    plt.xscale("log")
    plt.xlabel("Fator Multiplicador da Covariância de Medição (R / R_nominal)")
    plt.ylabel("RMSE da Posição Estimada (m)")
    plt.title("3-Fig 3: Sensibilidade do Erro vs Ponderação das Covariâncias R/Q")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{SAIDA}/3-fig3_sensibilidade_ruido_QR.png", dpi=170)
    plt.close()

    # 1. Salvar JSON solicitado
    with open(f"{SAIDA}/resultados-kalman.json", "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)

    # 2. Salvar TXT formatado
    with open(f"{SAIDA}/resultados-kalman.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RELATÓRIO DE RESULTADOS - PROJETO 3: FILTROS DE KALMAN\n")
        f.write("=" * 80 + "\n\n")
        f.write("\n".join(linhas_saida))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("ESTUDO DE SENSIBILIDADE DA MATRIZ DE COVARIÂNCIA R:\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Fator Multiplicador':<24} | {'RMSE Kalman (m)':<16}\n")
        f.write("-" * 45 + "\n")
        for m, err in zip(fatores_r, sens_rmse):
            f.write(f"{m:<24.2f} | {err:<16.4f}\n")
        f.write("-" * 45 + "\n")

    print(
        f"\nArquivos salvos com sucesso na pasta ./{SAIDA}:\n"
        f"  - 3-fig1_trajetoria_rastreamento.png\n"
        f"  - 3-fig2_convergencia_covariancia_P.png\n"
        f"  - 3-fig3_sensibilidade_ruido_QR.png\n"
        f"  - resultados-kalman.json\n"
        f"  - resultados-kalman.txt\n"
    )


if __name__ == "__main__":
    main()