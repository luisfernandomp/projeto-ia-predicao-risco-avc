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
# Arquivo: src/graficos.py
# Conteúdo: funções dos gráficos da análise exploratória (prevalência com
#   IC 95%, histogramas por classe e mapa de correlação), com matplotlib e
#   seaborn.
#
# Histórico de alterações:
#   25/09/2026 - Luis Fernando - criação do arquivo
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.pns import DIR_FIG

AZUL = "#2a78d6"
LARANJA = "#eb6834"
CINZA = "#555555"
CORES_CLASSE = {"Sem AVC": AZUL, "Com AVC": LARANJA}


def configurar():
    sns.set_theme(style="whitegrid", font_scale=0.9)
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlelocation"] = "left"


def salvar(fig, nome):
    DIR_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(DIR_FIG / f"{nome}.png")


def barras_prevalencia(ax, tab, titulo, referencia=None, xmax=None):
    """Barras horizontais com a prevalência de cada categoria e o IC 95%."""
    t = tab.iloc[::-1].reset_index(drop=True)  # primeira categoria em cima
    y = np.arange(len(t))
    erro = [t["prevalencia_%"] - t["ic95_inf_%"], t["ic95_sup_%"] - t["prevalencia_%"]]
    ax.barh(y, t["prevalencia_%"], height=0.55, color=AZUL, zorder=2)
    ax.errorbar(t["prevalencia_%"], y, xerr=erro, fmt="none", ecolor=CINZA, elinewidth=1, capsize=2.5, zorder=3)
    for i, (p, sup) in enumerate(zip(t["prevalencia_%"], t["ic95_sup_%"])):
        ax.text(sup + 0.15, i, f"{p:.1f}%", va="center", fontsize=8.5)
    if referencia is not None:
        ax.axvline(referencia, color=CINZA, lw=1, ls="--")
    ax.set_yticks(y, [str(c) for c in t["categoria"]])
    ax.set_title(titulo)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, xmax if xmax else t["ic95_sup_%"].max() * 1.25)
    ax.set_xlabel("Prevalência ponderada de AVC (%)")


def prevalencia_agrupada(tab, titulo, referencia=None):
    """Várias variáveis num gráfico só, uma abaixo da outra."""
    variaveis = list(dict.fromkeys(tab["variavel"]))
    pos, rotulos, titulos = [], [], []
    y = 0.0
    for var in variaveis:
        titulos.append((y, var))
        y += 1
        for _, linha in tab[tab["variavel"] == var].iterrows():
            pos.append(y)
            rotulos.append(str(linha["categoria"]))
            y += 1
        y += 0.6
    t = tab.reset_index(drop=True)
    pos = np.array(pos)

    fig, ax = plt.subplots(figsize=(8.5, 0.26 * y + 0.9))
    erro = [t["prevalencia_%"] - t["ic95_inf_%"], t["ic95_sup_%"] - t["prevalencia_%"]]
    ax.barh(pos, t["prevalencia_%"], height=0.62, color=AZUL, zorder=2)
    ax.errorbar(t["prevalencia_%"], pos, xerr=erro, fmt="none", ecolor=CINZA, elinewidth=1, capsize=2, zorder=3)
    for yi, p, sup in zip(pos, t["prevalencia_%"], t["ic95_sup_%"]):
        ax.text(sup + 0.12, yi, f"{p:.1f}%", va="center", fontsize=8)
    for yc, var in titulos:
        ax.text(-0.01, yc, var, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=9.5, fontweight="bold")
    if referencia is not None:
        ax.axvline(referencia, color=CINZA, lw=1, ls="--")
        ax.text(referencia, -1.2, f"geral: {referencia:.1f}%", ha="center", va="bottom", fontsize=8, color=CINZA)
    ax.set_yticks(pos, rotulos)
    ax.set_ylim(y - 0.6, -1.4)
    ax.set_xlim(0, t["ic95_sup_%"].max() * 1.12)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Prevalência ponderada de AVC (%), com IC 95%")
    ax.set_title(titulo, pad=14)
    fig.tight_layout()
    return fig


def hist_por_classe(ax, df, coluna, titulo, bins=40, faixa=None, log=False, fmt="{:.1f}"):
    """Histograma (densidade) de uma variável numérica, uma linha por classe."""
    d = df[[coluna, "avc"]].dropna().copy()
    d["classe"] = d["avc"].map({0: "Sem AVC", 1: "Com AVC"})
    d["x"] = np.log10(d[coluna].clip(lower=1)) if log else d[coluna]
    sns.histplot(d, x="x", hue="classe", palette=CORES_CLASSE, hue_order=list(CORES_CLASSE),
                 stat="density", common_norm=False, element="step", fill=False,
                 bins=bins, binrange=faixa, linewidth=2, ax=ax)
    medianas = d.groupby("classe")[coluna].median()
    ax.legend_.set_title("")
    for texto in ax.legend_.get_texts():
        texto.set_text(f"{texto.get_text()} (mediana {fmt.format(medianas[texto.get_text()])})")
    ax.set_title(titulo)
    ax.set_xlabel("")
    ax.set_ylabel("Densidade")


def mapa_correlacao(corr, titulo):
    """Heatmap do triângulo inferior da matriz de correlação."""
    mask = np.triu(np.ones_like(corr, dtype=bool))[1:, :-1]
    c = corr.iloc[1:, :-1]  # tira a primeira linha e a última coluna, que ficam vazias
    anot = c.map(lambda v: f"{v:.2f}" if abs(v) >= 0.1 else "")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(c, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=anot, fmt="",
                annot_kws={"size": 7}, square=True, linewidths=0.5, cbar_kws={"shrink": 0.7, "label": "ρ de Spearman"},
                ax=ax)
    ax.set_title(titulo)
    fig.tight_layout()
    return fig
