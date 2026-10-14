#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Projeto de custo mínimo de um tanque cilíndrico de armazenamento
 Método do Gradiente + Derivadas Numéricas (Diferenças Finitas Adiantadas)
 + Função de Penalização Exterior
-------------------------------------------------------------------------------
 Universidade do Estado do Rio de Janeiro - Instituto Politécnico (IPRJ)
 Disciplina: Modelagem Computacional (IPRJ01-10760 / IPRJ01-11873)
 Professor : Gustavo Barbosa Libotte
 Aluno     : Augusto Ricardo Bonow Moraes dos Santos
-------------------------------------------------------------------------------
 Problema:
     minimizar   f(r, h) = 350*pi*r^2 + 200*pi*r*h          (custo, R$)
     sujeito a   g(r, h) = 500 - pi*r^2*h <= 0                (V >= 500 m^3)

 Estratégia:
     1) Função auxiliar (penalização exterior quadrática):
            P(r, h) = f(r, h) + mu * [max(0, g(r, h))]^2
     2) Gradiente de P aproximado por diferenças finitas adiantadas:
            dP/dr ~ [P(r + D, h) - P(r, h)] / D
            dP/dh ~ [P(r, h + D) - P(r, h)] / D
     3) Método do gradiente com passo fixo alpha:
            x^(k+1) = x^(k) - alpha * grad P(x^(k))

 Uso:
     python otimizacao_tanque.py            -> executa todos os experimentos
     python otimizacao_tanque.py --rapido   -> versão reduzida (menos iterações)
     No Google Colab/Jupyter: cole o código em uma célula e execute, ou use
     !python otimizacao_tanque.py

 Saídas (pasta ./resultados):
     figuras .png, tabelas .csv e um resumo em resultados.json
===============================================================================
"""

import argparse
import csv
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")                     # gera figuras sem abrir janelas
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# 1. DADOS DO PROBLEMA
# -----------------------------------------------------------------------------
PI = math.pi
C_BASE = 150.0      # R$/m^2 - base do tanque
C_LATERAL = 100.0   # R$/m^2 - parede lateral
C_TETO = 200.0      # R$/m^2 - teto do tanque
V_MIN = 500.0       # m^3    - volume mínimo exigido

PASTA_SAIDA = "resultados"


def custo(r, h):
    """Função objetivo f(r,h): custo total de fabricação do tanque [R$].

    f = C_base*(pi r^2) + C_lateral*(2 pi r h) + C_teto*(pi r^2)
      = 350 pi r^2 + 200 pi r h
    """
    return C_BASE * PI * r * r + C_LATERAL * 2.0 * PI * r * h + C_TETO * PI * r * r


def volume(r, h):
    """Volume do cilindro V = pi r^2 h [m^3]."""
    return PI * r * r * h


def restricao(r, h):
    """Restrição de desigualdade no formato padrão g(r,h) <= 0."""
    return V_MIN - volume(r, h)


def cria_funcao_penalizada(mu):
    """Retorna a função auxiliar P(r,h) = f + mu*[max(0,g)]^2 para um dado mu.

    Usa-se uma 'closure' para que P dependa apenas de (r, h), o que
    deixa o laço do método do gradiente independente do valor de mu.
    """
    def P(r, h):
        g = restricao(r, h)
        multa = mu * g * g if g > 0.0 else 0.0   # max(0, g)^2 * mu
        return custo(r, h) + multa
    return P


# -----------------------------------------------------------------------------
# 2. DERIVADAS PARCIAIS POR DIFERENÇAS FINITAS ADIANTADAS
# -----------------------------------------------------------------------------
def gradiente_numerico(P, r, h, delta):
    """Aproxima o vetor gradiente de P em (r,h) por diferenças adiantadas.

    Retorna (dP/dr, dP/dh, P(r,h)). O valor P(r,h) é devolvido para ser
    reaproveitado, evitando uma avaliação extra da função por iteração
    (custo: 3 avaliações de P por iteração).
    """
    p0 = P(r, h)
    dPdr = (P(r + delta, h) - p0) / delta
    dPdh = (P(r, h + delta) - p0) / delta
    return dPdr, dPdh, p0


# -----------------------------------------------------------------------------
# 3. MÉTODO DO GRADIENTE (DESCIDA MÁXIMA COM PASSO FIXO)
# -----------------------------------------------------------------------------
def metodo_gradiente(mu, alpha, x0=(1.0, 1.0), delta=1e-5,
                     tol_grad=1e-3, tol_x=1e-12, max_iter=1_000_000,
                     n_hist=4000):
    """Minimiza P(r,h) pelo método do gradiente com passo fixo.

    Parâmetros
    ----------
    mu       : constante de penalidade
    alpha    : tamanho do passo (taxa de aprendizado)
    x0       : chute inicial (r0, h0)
    delta    : incremento das diferenças finitas
    tol_grad : critério de parada -> ||grad P|| < tol_grad
    tol_x    : critério de parada -> ||x(k+1) - x(k)|| < tol_x
    max_iter : número máximo de iterações
    n_hist   : quantidade aproximada de pontos guardados no histórico

    Retorna
    -------
    dicionário com o ponto final, indicadores e o histórico de convergência.
    """
    P = cria_funcao_penalizada(mu)
    r, h = float(x0[0]), float(x0[1])

    # Para não guardar milhões de pontos, o histórico é registrado em todas
    # as iterações iniciais e depois a cada 'passo_hist' iterações.
    passo_hist = max(1, max_iter // n_hist)
    hist = {"k": [], "r": [], "h": [], "P": [], "f": [], "V": [], "norma_grad": []}

    status = "max_iter"
    t0 = time.perf_counter()
    k = 0
    norma = float("nan")
    for k in range(max_iter):
        dPdr, dPdh, p = gradiente_numerico(P, r, h, delta)
        norma = math.hypot(dPdr, dPdh)

        if k < 200 or k % passo_hist == 0:
            hist["k"].append(k)
            hist["r"].append(r)
            hist["h"].append(h)
            hist["P"].append(p)
            hist["f"].append(custo(r, h))
            hist["V"].append(volume(r, h))
            hist["norma_grad"].append(norma)

        # Critério 1: derivadas próximas de zero
        if norma < tol_grad:
            status = "convergiu (grad)"
            break

        # Atualização: caminhar no sentido contrário ao gradiente
        r_novo = r - alpha * dPdr
        h_novo = h - alpha * dPdh

        # Detecção de divergência (instabilidade numérica)
        if (not math.isfinite(r_novo) or not math.isfinite(h_novo)
                or abs(r_novo) > 1e6 or abs(h_novo) > 1e6):
            status = "divergiu"
            break

        passo = math.hypot(r_novo - r, h_novo - h)
        r, h = r_novo, h_novo

        # Critério 2: as variáveis pararam de mudar
        if passo < tol_x:
            status = "convergiu (passo)"
            break

    tempo = time.perf_counter() - t0
    return {
        "mu": mu, "alpha": alpha, "delta": delta, "x0": list(x0),
        "status": status, "iteracoes": k + 1, "tempo_s": tempo,
        "r": r, "h": h, "custo": custo(r, h), "volume": volume(r, h),
        "g": restricao(r, h), "norma_grad": norma, "hist": hist,
    }


# -----------------------------------------------------------------------------
# 4. FERRAMENTAS DE ANÁLISE
# -----------------------------------------------------------------------------
def solucao_analitica():
    """Solução exata (restrição ativa) usada como referência.

    Com V = 500 -> h = 500/(pi r^2). Substituindo em f:
        f(r) = 350 pi r^2 + 100000 / r
        f'(r) = 700 pi r - 100000 / r^2 = 0  ->  r^3 = 100000 / (700 pi)
    O multiplicador de Lagrange vale lambda = 200/r.
    """
    r = (100000.0 / (700.0 * PI)) ** (1.0 / 3.0)
    h = V_MIN / (PI * r * r)
    return {"r": r, "h": h, "custo": custo(r, h), "volume": volume(r, h),
            "h_sobre_r": h / r, "lambda": 200.0 / r}


def hessiana_penalidade(r, h, mu, g=None):
    """Hessiana de mu*g^2 (lado inviável) + Hessiana de f, avaliada em (r,h).

    Usada apenas para análise de estabilidade/condicionamento (não entra no
    algoritmo). Derivadas analíticas:
        f_rr = 700 pi, f_rh = 200 pi, f_hh = 0
        g_r = -2 pi r h, g_h = -pi r^2
        (mu g^2)'' = 2 mu (grad g grad g^T + g * Hess g)
    Se 'g' não for informado, usa-se g(r,h).
    """
    if g is None:
        g = restricao(r, h)
    gr, gh = -2 * PI * r * h, -PI * r * r
    grr, grh, ghh = -2 * PI * h, -2 * PI * r, 0.0
    Hf = np.array([[700 * PI, 200 * PI], [200 * PI, 0.0]])
    Hg = 2 * mu * (np.outer([gr, gh], [gr, gh]) + g * np.array([[grr, grh], [grh, ghh]]))
    return Hf + Hg


def minimo_exato_penalizado(mu, ref, tol=1e-12, max_iter=100):
    """Mínimo 'exato' de P(r,h) para um dado mu, obtido pelo método de Newton
    com gradiente e Hessiana ANALÍTICOS (usado só como referência para medir
    o erro introduzido pelas diferenças finitas).

        dP/dr = 700 pi r + 200 pi h - 2 mu g (2 pi r h)
        dP/dh = 200 pi r            - 2 mu g (pi r^2)
    """
    x = np.array([ref["r"], ref["h"]])
    for _ in range(max_iter):
        r, h = x
        g = max(0.0, restricao(r, h))
        grad = np.array([700 * PI * r + 200 * PI * h - 2 * mu * g * 2 * PI * r * h,
                         200 * PI * r - 2 * mu * g * PI * r * r])
        dx = np.linalg.solve(hessiana_penalidade(r, h, mu), -grad)
        x = x + dx
        if np.linalg.norm(dx) < tol:
            break
    r, h = x
    return {"r": r, "h": h, "custo": custo(r, h), "volume": volume(r, h)}


def alpha_critico(mu, ref):
    """Estimativa do maior passo estável: alpha < 2 / lambda_max(H).

    A Hessiana é avaliada próximo do mínimo de P, onde a violação da
    restrição vale aproximadamente g ~ lambda/(2 mu) (penalização exterior).
    """
    H = hessiana_penalidade(ref["r"], ref["h"], mu, g=ref["lambda"] / (2 * mu))
    lam = np.linalg.eigvalsh(H)
    return 2.0 / lam.max(), lam


# -----------------------------------------------------------------------------
# 5. EXPERIMENTOS
# -----------------------------------------------------------------------------
def salvar_csv(nome, linhas, campos):
    with open(os.path.join(PASTA_SAIDA, nome), "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for ln in linhas:
            w.writerow(ln)


def resumo(res):
    """Remove o histórico (grande) para gravar apenas os indicadores."""
    return {k: v for k, v in res.items() if k != "hist"}


def imprime(res, ref):
    erro = abs(res["custo"] - ref["custo"]) / ref["custo"] * 100
    print(f"  mu={res['mu']:.0e} alpha={res['alpha']:.1e} delta={res['delta']:.0e} "
          f"| {res['status']:<18} it={res['iteracoes']:>8d} t={res['tempo_s']:6.2f}s "
          f"| r={res['r']:.4f} h={res['h']:.4f} f={res['custo']:.2f} "
          f"V={res['volume']:.3f} erro_f={erro:.3f}%")


def experimentos(rapido=False):
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    escala = 0.2 if rapido else 1.0
    ref = solucao_analitica()
    x0 = (1.0, 1.0)
    print("Solução analítica de referência:", {k: round(v, 5) for k, v in ref.items()})

    saida = {"referencia": ref, "x0": list(x0)}

    # --- 5.1 Estabilidade: alpha crítico para cada mu ------------------------
    tab_estab = []
    for mu in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
        a_c, lam = alpha_critico(mu, ref)
        ex = minimo_exato_penalizado(mu, ref)
        tab_estab.append({"mu": mu, "r_exato_P": ex["r"], "h_exato_P": ex["h"],
                          "custo_exato_P": ex["custo"], "volume_exato_P": ex["volume"], "lambda_min": lam.min(), "lambda_max": lam.max(),
                          "condicionamento": lam.max() / abs(lam.min()),
                          "alpha_critico": a_c,
                          "violacao_teorica": ref["lambda"] / (2 * mu)})
    salvar_csv("estabilidade.csv", tab_estab, list(tab_estab[0].keys()))
    saida["estabilidade"] = tab_estab

    # --- 5.2 Caso base ---------------------------------------------------------
    print("\n[1] Caso base")
    base = metodo_gradiente(mu=1e2, alpha=1e-7, x0=x0, delta=1e-5,
                            max_iter=int(3_000_000 * escala))
    imprime(base, ref)
    saida["caso_base"] = resumo(base)

    # --- 5.3 Teste do tamanho do passo alpha (mu fixo = 1e2) ---------------------
    print("\n[2] Variação de alpha (mu = 1e2)")
    tab_alpha, hist_alpha = [], {}
    for a in [1e-3, 1e-5, 1e-6, 2e-7, 1e-7, 5e-8, 1e-8]:
        res = metodo_gradiente(mu=1e2, alpha=a, x0=x0, delta=1e-5,
                               max_iter=int(3_000_000 * escala))
        imprime(res, ref)
        tab_alpha.append(resumo(res))
        hist_alpha[a] = res["hist"]
    salvar_csv("teste_alpha.csv", tab_alpha, list(tab_alpha[0].keys()))
    saida["teste_alpha"] = tab_alpha

    # --- 5.4 Teste da constante de penalidade mu (alpha = 0,5*alpha_crit) -------
    print("\n[3] Variação de mu (alpha ajustado a cada mu)")
    tab_mu, hist_mu = [], {}
    for mu in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
        a = 1e-5 / mu    # ~0,8 * alpha_crítico (ver tabela de estabilidade)
        res = metodo_gradiente(mu=mu, alpha=a, x0=x0, delta=1e-5,
                               max_iter=int(3_000_000 * escala))
        imprime(res, ref)
        tab_mu.append(resumo(res))
        hist_mu[mu] = res["hist"]
    salvar_csv("teste_mu.csv", tab_mu, list(tab_mu[0].keys()))
    saida["teste_mu"] = tab_mu

    # --- 5.5 mu grande com alpha fixo (mostra a instabilidade) -----------------
    print("\n[4] Mapa de estabilidade: combinações (mu, alpha)")
    tab_mapa = []
    for mu in [1e2, 1e4, 1e6]:
        for a in [1e-3, 1e-5, 1e-7, 1e-9, 1e-11]:
            res = metodo_gradiente(mu=mu, alpha=a, x0=x0, delta=1e-5,
                                   max_iter=int(300_000 * escala))
            imprime(res, ref)
            tab_mapa.append(resumo(res))
    salvar_csv("mapa_mu_alpha.csv", tab_mapa, list(tab_mapa[0].keys()))
    saida["mapa_mu_alpha"] = tab_mapa

    # --- 5.6 Influência do incremento delta das diferenças finitas -----------------
    print("\n[5] Variação de delta (mu = 1e2, alpha = 1e-7)")
    tab_delta = []
    for d in [1e-3, 1e-5, 1e-7, 1e-9]:
        res = metodo_gradiente(mu=1e2, alpha=1e-7, x0=x0, delta=d,
                               max_iter=int(3_000_000 * escala))
        imprime(res, ref)
        tab_delta.append(resumo(res))
    salvar_csv("teste_delta.csv", tab_delta, list(tab_delta[0].keys()))
    saida["teste_delta"] = tab_delta

    # --- 5.7 Penalização sequencial (mu crescente, partida a quente) -----------
    print("\n[6] Penalização sequencial (mu = 10 ... 1e4, delta = 1e-8)")
    tab_seq, xk, total_it, total_t = [], x0, 0, 0.0
    for mu in [1e1, 1e2, 1e3, 1e4]:
        res = metodo_gradiente(mu=mu, alpha=1e-5 / mu, x0=xk, delta=1e-8,
                               max_iter=int(2_000_000 * escala))
        imprime(res, ref)
        total_it += res["iteracoes"]
        total_t += res["tempo_s"]
        r_ = resumo(res)
        r_["iteracoes_acumuladas"] = total_it
        r_["tempo_acumulado_s"] = total_t
        tab_seq.append(r_)
        xk = (res["r"], res["h"])
    salvar_csv("penalizacao_sequencial.csv", tab_seq, list(tab_seq[0].keys()))
    saida["penalizacao_sequencial"] = tab_seq

    with open(os.path.join(PASTA_SAIDA, "resultados.json"), "w", encoding="utf-8") as fp:
        json.dump(saida, fp, indent=2, ensure_ascii=False, default=float)

    # --- 6. Gráficos -------------------------------------------------------------
    graficos(base, hist_alpha, hist_mu, ref)
    print(f"\nResultados gravados na pasta '{PASTA_SAIDA}/'.")
    return saida


# -----------------------------------------------------------------------------
# 6. GRÁFICOS
# -----------------------------------------------------------------------------
def graficos(base, hist_alpha, hist_mu, ref):
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "axes.grid": True, "grid.alpha": 0.3})
    H = base["hist"]
    k = np.array(H["k"]) + 1

    # (a) Convergência do caso base: custo, volume e norma do gradiente
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    ax[0].semilogx(k, H["f"], color="#1f4e79")
    ax[0].axhline(ref["custo"], ls="--", color="#c00000", lw=1, label="ótimo analítico")
    ax[0].set_ylim(0, 1.2e5)
    ax[0].set_xlabel("iteração k"); ax[0].set_ylabel("custo f(r,h) [R$]")
    ax[0].set_title("(a) Custo"); ax[0].legend()
    ax[1].semilogx(k, H["V"], color="#1f4e79")
    ax[1].axhline(V_MIN, ls="--", color="#c00000", lw=1, label="V mín = 500 m³")
    ax[1].set_xlabel("iteração k"); ax[1].set_ylabel("volume [m³]")
    ax[1].set_title("(b) Volume"); ax[1].legend()
    ax[2].loglog(k, H["norma_grad"], color="#1f4e79")
    ax[2].set_xlabel("iteração k"); ax[2].set_ylabel("||∇P||")
    ax[2].set_title("(c) Norma do gradiente")
    fig.tight_layout(); fig.savefig(os.path.join(PASTA_SAIDA, "fig_convergencia_base.png"), dpi=200)
    plt.close(fig)

    # (b) Evolução de r e h
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.semilogx(k, H["r"], label="raio r", color="#1f4e79")
    ax.semilogx(k, H["h"], label="altura h", color="#2e7d32")
    ax.axhline(ref["r"], ls="--", color="#1f4e79", lw=0.8)
    ax.axhline(ref["h"], ls="--", color="#2e7d32", lw=0.8)
    ax.set_xlabel("iteração k"); ax.set_ylabel("dimensão [m]"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(PASTA_SAIDA, "fig_evolucao_r_h.png"), dpi=200)
    plt.close(fig)

    # (c) Trajetória sobre as curvas de nível de f e a fronteira V = 500
    rr, hh = np.meshgrid(np.linspace(0.5, 7, 300), np.linspace(0.5, 20, 300))
    ff = 350 * PI * rr ** 2 + 200 * PI * rr * hh
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cs = ax.contour(rr, hh, ff, levels=np.arange(5000, 90000, 5000), cmap="viridis", linewidths=0.8)
    ax.clabel(cs, fmt="%d", fontsize=7)
    rv = np.linspace(0.5, 7, 400)
    ax.plot(rv, V_MIN / (PI * rv ** 2), "r-", lw=1.5, label="g(r,h)=0  (V = 500 m³)")
    ax.fill_between(rv, 0, V_MIN / (PI * rv ** 2), color="red", alpha=0.07, label="região inviável")
    ax.plot(H["r"], H["h"], "k.-", ms=2, lw=0.8, label="trajetória do método")
    ax.plot(H["r"][0], H["h"][0], "go", label="chute inicial")
    ax.plot(ref["r"], ref["h"], "r*", ms=12, label="ótimo analítico")
    ax.set_xlim(0.5, 7); ax.set_ylim(0.5, 20)
    ax.set_xlabel("raio r [m]"); ax.set_ylabel("altura h [m]"); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(PASTA_SAIDA, "fig_trajetoria.png"), dpi=200)
    plt.close(fig)

    # (d) Comparação de alphas
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for a, Hh in hist_alpha.items():
        if len(Hh["k"]) > 5 and np.all(np.isfinite(Hh["f"])) and max(Hh["V"]) < 1e5:
            ax.loglog(np.array(Hh["k"]) + 1, np.abs(np.array(Hh["f"]) - ref["custo"]) + 1e-3,
                      label=f"α = {a:.0e}")
    ax.set_xlabel("iteração k"); ax.set_ylabel("|f − f*|  [R$]")
    ax.set_title("Erro do custo para diferentes passos (μ = 10²)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(PASTA_SAIDA, "fig_teste_alpha.png"), dpi=200)
    plt.close(fig)

    # (e) Comparação de mus
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for mu, Hh in hist_mu.items():
        kk = np.array(Hh["k"]) + 1
        ax[0].loglog(kk, np.abs(np.array(Hh["f"]) - ref["custo"]) + 1e-3, label=f"μ = {mu:.0e}")
        ax[1].semilogx(kk, Hh["V"], label=f"μ = {mu:.0e}")
    ax[0].set_xlabel("iteração k"); ax[0].set_ylabel("|f − f*|  [R$]"); ax[0].legend(fontsize=8)
    ax[0].set_title("(a) Erro do custo")
    ax[1].axhline(V_MIN, ls="--", color="k", lw=0.8)
    ax[1].set_ylim(480, 505)
    ax[1].set_xlabel("iteração k"); ax[1].set_ylabel("volume [m³]"); ax[1].legend(fontsize=8)
    ax[1].set_title("(b) Volume (zoom)")
    fig.tight_layout(); fig.savefig(os.path.join(PASTA_SAIDA, "fig_teste_mu.png"), dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Otimização do custo de um tanque cilíndrico")
    parser.add_argument("--rapido", action="store_true",
                        help="reduz o número máximo de iterações (teste rápido)")
    # parse_known_args ignora argumentos extras (ex.: o "-f kernel.json" que o
    # Jupyter/Google Colab passa automaticamente), evitando o erro SystemExit: 2
    args, _ = parser.parse_known_args()
    experimentos(rapido=args.rapido)
