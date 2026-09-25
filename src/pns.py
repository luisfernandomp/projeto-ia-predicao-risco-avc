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
# Arquivo: src/pns.py
# Conteúdo: download dos microdados da PNS 2019 no site do IBGE, leitura do
#   arquivo de largura fixa (usando o layout do input SAS), seleção das
#   variáveis do projeto e montagem da base analítica (40 anos ou mais)
#   usada nos notebooks.
#
# Histórico de alterações:
#   25/09/2026 - Luis Fernando - criação do arquivo
# ---------------------------------------------------------------------------

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parents[1]
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROC = RAIZ / "data" / "processed"
DIR_FIG = RAIZ / "results" / "figures"
DIR_TAB = RAIZ / "results" / "tables"

URL_DADOS = "https://ftp.ibge.gov.br/PNS/2019/Microdados/Dados/PNS_2019_20220525.zip"
URL_DOC = "https://ftp.ibge.gov.br/PNS/2019/Microdados/Documentacao/Dicionario_e_input_20220530.zip"

ARQ_EXTRATO = DIR_PROC / "pns2019_extrato_bruto.parquet"
ARQ_BASE = DIR_PROC / "pns2019_avc_40mais.parquet"

IDADE_MIN = 40

# variáveis que vamos tirar do arquivo do IBGE (código -> descrição do dicionário)
VARIAVEIS = {
    "V0001": "Unidade da Federação",
    "V0024": "Estrato",
    "UPA_PNS": "Unidade Primária de Amostragem",
    "V0026": "Situação censitária (1 urbano, 2 rural)",
    "M001": "Entrevista do adulto selecionado (1 = realizada)",
    "V00291": "Peso do morador selecionado com calibração",
    "C006": "Sexo",
    "C008": "Idade (anos)",
    "C009": "Cor ou raça",
    "Q00201": "Diagnóstico médico de hipertensão",
    "Q03001": "Diagnóstico médico de diabetes",
    "P00104": "Peso final (kg)",
    "P00404": "Altura final (cm)",
    "P050": "Fuma atualmente",
    "P052": "Fumou no passado",
    "P027": "Frequência de consumo de álcool",
    "P034": "Praticou exercício físico nos últimos 3 meses",
    "VDD004A": "Nível de instrução",
    "VDF003": "Rendimento domiciliar per capita (R$)",
    "VDF004": "Faixa de rendimento domiciliar per capita",
    "I00102": "Possui plano de saúde",
    "Q068": "Diagnóstico médico de AVC (variável-alvo)",
}

# perguntas que só quem teve AVC responde (idade no diagnóstico, cuidados,
# limitação). Se entrassem como atributo, entregariam a resposta pro modelo.
VARIAVEIS_VAZAMENTO = ["Q070", "Q07208", "Q07209", "Q07210", "Q07211", "Q07212", "Q07213", "Q073"]

# rótulos do dicionário
REGIOES = {1: "Norte", 2: "Nordeste", 3: "Sudeste", 4: "Sul", 5: "Centro-Oeste"}
SEXO = {1: "Masculino", 2: "Feminino"}
COR_RACA = {1: "Branca", 2: "Preta", 3: "Amarela", 4: "Parda", 5: "Indígena"}
SITUACAO = {1: "Urbana", 2: "Rural"}
ALCOOL = {1: "Não bebe", 2: "Menos de 1x/mês", 3: "1x/mês ou mais"}
FAIXA_RENDA = {1: "Até 1/4 SM", 2: "1/4 a 1/2 SM", 3: "1/2 a 1 SM", 4: "1 a 2 SM",
               5: "2 a 3 SM", 6: "3 a 5 SM", 7: "Mais de 5 SM"}
ESCOLARIDADE_GRUPO = {1: "Sem instrução/Fund. incompleto", 2: "Sem instrução/Fund. incompleto",
                      3: "Fund. completo/Médio incompleto", 4: "Fund. completo/Médio incompleto",
                      5: "Médio completo/Sup. incompleto", 6: "Médio completo/Sup. incompleto",
                      7: "Superior completo"}

FAIXAS_ETARIAS = ["40-49", "50-59", "60-69", "70-79", "80+"]
FAIXAS_IMC = ["Baixo peso", "Eutrófico", "Sobrepeso", "Obesidade"]
TABAGISMO = ["Nunca fumou", "Ex-fumante", "Fumante atual"]

ATRIBUTOS = ["idade", "sexo", "cor_raca", "regiao", "situacao",
             "hipertensao", "diabetes", "peso", "altura", "imc",
             "tabagismo", "alcool", "atividade_fisica",
             "escolaridade", "renda_per_capita", "plano_saude"]
COLUNAS_DESENHO = ["peso_amostral", "estrato", "upa"]


# ----------------------------- download ------------------------------------

def baixar(url, destino):
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        print(f"{destino.name} já existe, pulando download")
        return destino
    print(f"baixando {url} ...")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(destino, "wb") as f:
        for bloco in r.iter_content(chunk_size=1024 * 1024):
            f.write(bloco)
    print(f"ok ({destino.stat().st_size / 1e6:.1f} MB)")
    return destino


def baixar_pns2019():
    """Baixa e descompacta os microdados e a documentação em data/raw."""
    for url in (URL_DADOS, URL_DOC):
        z = baixar(url, DIR_RAW / url.split("/")[-1])
        with zipfile.ZipFile(z) as zf:
            for nome in zf.namelist():
                if not (DIR_RAW / nome).exists():
                    zf.extract(nome, DIR_RAW)
    return {
        "txt": DIR_RAW / "PNS_2019.txt",
        "sas": DIR_RAW / "input_PNS_2019.sas",
        "dicionario": DIR_RAW / "dicionario_PNS_microdados_2019.xls",
    }


# ----------------------------- leitura -------------------------------------

def ler_layout(caminho_sas):
    """Pega posição e tamanho de cada variável no input SAS do IBGE.

    As linhas são do tipo  @00117  C008  3.   ou  @01426  V00291  14.8
    """
    padrao = re.compile(r"^@(\d+)\s+(\w+)\s+(\$?)(\d+)\.")
    linhas = []
    with open(caminho_sas, encoding="latin-1") as f:
        for linha in f:
            m = padrao.match(linha.strip())
            if m:
                inicio, nome, cifrao, largura = m.groups()
                linhas.append({"variavel": nome, "inicio": int(inicio), "largura": int(largura),
                               "texto": cifrao == "$"})
    return pd.DataFrame(linhas)


def extrair_variaveis(caminho_txt, layout, variaveis):
    """Lê só as colunas pedidas do arquivo de largura fixa (tudo como texto)."""
    sel = layout.set_index("variavel").loc[list(variaveis)]
    colspecs = [(r.inicio - 1, r.inicio - 1 + r.largura) for r in sel.itertuples()]
    df = pd.read_fwf(caminho_txt, colspecs=colspecs, names=list(sel.index),
                     dtype=str, header=None, encoding="latin-1")
    return df.apply(lambda s: s.str.strip().replace("", pd.NA))


# ----------------------------- preparação ----------------------------------

def num(s):
    return pd.to_numeric(s, errors="coerce")


def sim_nao(s, branco_vira_nao=False):
    # 1 = sim, 2 = não, 9 = ignorado (vira NaN)
    x = num(s)
    out = x.map({1: 1.0, 2: 0.0})
    if branco_vira_nao:
        # quem nunca mediu pressão/glicemia pula a pergunta de diagnóstico
        out = out.where(x.notna(), 0.0)
    return out


def fluxo_amostra(bruto):
    """Quantos registros sobram em cada filtro."""
    entrevistado = num(bruto["M001"]) == 1
    idade_ok = num(bruto["C008"]) >= IDADE_MIN
    alvo_ok = num(bruto["Q068"]).isin([1, 2])
    return pd.DataFrame({
        "etapa": ["Moradores no arquivo",
                  "Com entrevista individual realizada (M001 = 1)",
                  f"Com {IDADE_MIN} anos ou mais",
                  "Com resposta válida sobre AVC (Q068 = 1 ou 2)"],
        "registros": [len(bruto), entrevistado.sum(), (entrevistado & idade_ok).sum(),
                      (entrevistado & idade_ok & alvo_ok).sum()],
    })


def contar_ignorados(bruto):
    """Respostas 'ignorado' (código 9 / 999) e brancos, no recorte de 40+."""
    rec = bruto[(num(bruto["M001"]) == 1) & (num(bruto["C008"]) >= IDADE_MIN)]
    pular = {"V0001", "V0024", "UPA_PNS", "V00291", "M001", "C008", "VDF003", "P00104"}
    linhas = []
    for var, desc in VARIAVEIS.items():
        if var in pular:
            continue
        x = num(rec[var])
        cod = 999 if var == "P00404" else 9
        linhas.append({"variavel": var, "descricao": desc,
                       "ignorado_n": int((x == cod).sum()),
                       "branco_n": int(rec[var].isna().sum())})
    t = pd.DataFrame(linhas)
    t["ignorado_%"] = 100 * t["ignorado_n"] / len(rec)
    t["branco_%"] = 100 * t["branco_n"] / len(rec)
    return t


def construir_base(bruto):
    """Filtra 40+ com entrevista feita e recodifica as variáveis."""
    df = bruto[(num(bruto["M001"]) == 1) & (num(bruto["C008"]) >= IDADE_MIN)]
    df = df[num(df["Q068"]).isin([1, 2])].copy()

    b = pd.DataFrame(index=df.index)
    b["avc"] = (num(df["Q068"]) == 1).astype(int)

    # demográficos
    b["idade"] = num(df["C008"])
    b["faixa_etaria"] = pd.cut(b["idade"], [39, 49, 59, 69, 79, np.inf], labels=FAIXAS_ETARIAS)
    b["sexo"] = num(df["C006"]).map(SEXO)
    b["cor_raca"] = num(df["C009"]).map(COR_RACA)
    b["uf"] = df["V0001"]
    b["regiao"] = num(df["V0001"].str[0]).map(REGIOES)
    b["situacao"] = num(df["V0026"]).map(SITUACAO)

    # clínicos
    b["hipertensao"] = sim_nao(df["Q00201"], branco_vira_nao=True)
    b["diabetes"] = sim_nao(df["Q03001"], branco_vira_nao=True)
    b["peso"] = num(df["P00104"])
    b["altura"] = num(df["P00404"]).replace(999, np.nan)
    imc = b["peso"] / (b["altura"] / 100) ** 2
    b["imc"] = imc.where(imc.between(12, 70))  # fora disso é erro de digitação
    b["imc_categoria"] = pd.cut(b["imc"], [0, 18.5, 25, 30, np.inf], right=False, labels=FAIXAS_IMC)

    # comportamentais
    fuma = num(df["P050"])
    fumou = num(df["P052"])
    tab = pd.Series(pd.NA, index=df.index, dtype="object")
    tab[fuma.isin([1, 2])] = "Fumante atual"
    tab[(fuma == 3) & fumou.isin([1, 2])] = "Ex-fumante"
    tab[(fuma == 3) & (fumou == 3)] = "Nunca fumou"
    b["tabagismo"] = tab
    b["alcool"] = num(df["P027"]).map(ALCOOL)
    b["atividade_fisica"] = sim_nao(df["P034"])

    # socioeconômicos
    b["escolaridade"] = num(df["VDD004A"])  # 1 (sem instrução) a 7 (superior completo)
    b["escolaridade_grupo"] = b["escolaridade"].map(ESCOLARIDADE_GRUPO)
    b["renda_per_capita"] = num(df["VDF003"])
    b["faixa_renda"] = num(df["VDF004"]).map(FAIXA_RENDA)
    b["plano_saude"] = sim_nao(df["I00102"])

    # desenho amostral (só para as estimativas ponderadas)
    b["peso_amostral"] = num(df["V00291"])
    b["estrato"] = df["V0024"]
    b["upa"] = df["UPA_PNS"]

    # deixa as categorias na ordem certa pros gráficos
    ordens = {
        "faixa_etaria": FAIXAS_ETARIAS,
        "escolaridade_grupo": list(dict.fromkeys(ESCOLARIDADE_GRUPO.values())),
        "imc_categoria": FAIXAS_IMC,
        "tabagismo": TABAGISMO,
        "alcool": list(ALCOOL.values()),
        "faixa_renda": list(FAIXA_RENDA.values()),
        "regiao": list(REGIOES.values()),
    }
    for col, ordem in ordens.items():
        b[col] = pd.Categorical(b[col], categories=ordem, ordered=True)

    return b.reset_index(drop=True)


if __name__ == "__main__":
    arqs = baixar_pns2019()
    layout = ler_layout(arqs["sas"])
    bruto = extrair_variaveis(arqs["txt"], layout, VARIAVEIS)
    DIR_PROC.mkdir(parents=True, exist_ok=True)
    bruto.to_parquet(ARQ_EXTRATO, index=False)
    base = construir_base(bruto)
    base.to_parquet(ARQ_BASE, index=False)
    print(f"{len(bruto):,} moradores lidos, base analítica com {len(base):,} registros")
