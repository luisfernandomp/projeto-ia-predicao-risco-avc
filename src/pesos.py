# ---------------------------------------------------------------------------
# Inteligência Artificial - 7º semestre - Ciência da Computação (Mackenzie)
# Projeto: Aplicação de IA para Predição do Risco de AVC (PNS 2019)
#
# Integrantes:
#   Luana Miho Yasuda                  - RA 10396439
#   Luis Fernando de Mesquita Pereira  - RA 10410686
#   Richard Barbosa Sanches            - RA 10420179
#   Totti Ferreira Gomes               - RA 10428490
#   Gabriel Resende Menezes            - RA 10419003
#
# Arquivo: src/pesos.py
# Conteúdo: prevalência ponderada pelo peso amostral da PNS, com intervalo de
#   confiança de 95% que leva em conta os estratos e as UPAs (conglomerados).
#   O erro-padrão é calculado por linearização de Taylor e o IC na escala
#   logit, do mesmo jeito que o svyciprop do pacote survey (R) faz.
#
# Histórico de alterações:
#   25/09/2026 - Luis Fernando - criação do arquivo
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd

Z = 1.96


def erro_padrao(y, d, w, estrato, upa, p):
    # p = soma(w*d*y) / soma(w*d); z é a variável linearizada de cada pessoa.
    # As UPAs fora do subgrupo entram com zero (é assim que se trata subdomínio).
    z = w * d * (y - p) / np.sum(w * d)
    por_upa = pd.DataFrame({"e": estrato, "u": upa, "z": z}).groupby(["e", "u"], observed=True)["z"].sum()
    g = por_upa.groupby(level="e")
    n_h = g.size()
    sq = g.apply(lambda s: np.sum((s - s.mean()) ** 2))
    ok = n_h > 1  # estrato com uma UPA só não contribui
    return float(np.sqrt(np.sum((n_h / (n_h - 1) * sq)[ok])))


def prevalencia(df, y="avc", por=None, peso="peso_amostral", estrato="estrato", upa="upa"):
    """Prevalência ponderada (%) de y, no total ou por categoria da coluna `por`."""
    base = df[df[y].notna()]
    yv = base[y].to_numpy(dtype=float)
    wv = base[peso].to_numpy(dtype=float)
    ev = base[estrato].to_numpy()
    uv = base[upa].to_numpy()

    if por is None:
        grupos = [("Total", np.ones(len(base), dtype=bool))]
    else:
        col = base[por]
        cats = col.cat.categories if hasattr(col, "cat") else sorted(col.dropna().unique())
        grupos = [(c, (col == c).to_numpy()) for c in cats]

    linhas = []
    for nome, mask in grupos:
        d = mask.astype(float)
        if d.sum() == 0:
            continue
        p = np.sum(wv * d * yv) / np.sum(wv * d)
        ep = erro_padrao(yv, d, wv, ev, uv, p)
        if 0 < p < 1:
            lg, ep_lg = np.log(p / (1 - p)), ep / (p * (1 - p))
            li = 1 / (1 + np.exp(-(lg - Z * ep_lg)))
            ls = 1 / (1 + np.exp(-(lg + Z * ep_lg)))
        else:
            li = ls = p
        linhas.append({"categoria": nome, "n": int(d.sum()), "casos": int(np.sum(d * yv)),
                       "populacao_estimada": float(np.sum(wv * d)),
                       "prevalencia_%": 100 * p, "ic95_inf_%": 100 * li, "ic95_sup_%": 100 * ls,
                       "erro_padrao_%": 100 * ep})
    out = pd.DataFrame(linhas)
    if por is not None:
        out.insert(0, "variavel", por)
    return out


def proporcao_ponderada(df, coluna, peso="peso_amostral"):
    """Distribuição (%) ponderada das categorias de uma coluna."""
    base = df[df[coluna].notna()]
    return 100 * base.groupby(coluna, observed=True)[peso].sum() / base[peso].sum()
