# Dataset: Pesquisa Nacional de Saúde (PNS) 2019

## O que é

A PNS 2019 é um inquérito domiciliar nacional feito pelo IBGE em parceria com o Ministério da Saúde. Ela levanta
informações sobre doenças referidas, hábitos de vida, características socioeconômicas e acesso a serviços de
saúde. Os microdados são públicos e anonimizados.

| | |
|---|---|
| Produtor | IBGE / Ministério da Saúde |
| Coleta | agosto de 2019 a março de 2020 |
| Unidade | morador de 15 anos ou mais selecionado para o questionário individual |
| Amostragem | conglomerada em três estágios (setores, domicílios e morador), com estratos e pesos calibrados |
| Registros | 293.726 moradores; 90.846 entrevistas individuais |
| Formato | texto de largura fixa (`PNS_2019.txt`, ~455 MB) + dicionário (`.xls`) + programa de leitura (`.sas`) |

Links oficiais:

- Microdados: https://ftp.ibge.gov.br/PNS/2019/Microdados/Dados/PNS_2019_20220525.zip
- Dicionário e input SAS: https://ftp.ibge.gov.br/PNS/2019/Microdados/Documentacao/Dicionario_e_input_20220530.zip
- Página da pesquisa: https://www.ibge.gov.br/estatisticas/sociais/saude/9160-pesquisa-nacional-de-saude.html

## Por que os dados não estão no repositório

Como está na Seção 4.1 do artigo, os microdados não são redistribuídos. O repositório tem só o código que baixa
e processa. Para gerar tudo localmente basta rodar o notebook `01_obtencao_e_preparacao.ipynb` ou
`python -m src.pns`. A pasta fica assim (ignorada pelo git):

```
data/
├── raw/          # arquivos do IBGE (zip, PNS_2019.txt, dicionário, input SAS)
└── processed/
    ├── pns2019_extrato_bruto.parquet   # 22 variáveis selecionadas, todos os moradores, códigos originais
    └── pns2019_avc_40mais.parquet      # base analítica: 54.987 adultos de 40+ anos, 25 colunas
```

## Recorte e variáveis

Ficam na base os adultos com entrevista individual realizada (`M001 = 1`), com 40 anos ou mais e com resposta
válida sobre AVC. Dá 54.987 registros, 1.844 com AVC (3,35% da amostra, prevalência ponderada de 3,20%).

Variável-alvo: `Q068` – "Algum médico já lhe deu o diagnóstico de AVC (Acidente Vascular Cerebral) ou derrame?"
(1 = sim, 2 = não), que virou a coluna `avc` (1/0).

| Grupo | Coluna na base | Código PNS | Descrição |
|---|---|---|---|
| Demográficos | `idade`, `faixa_etaria` | C008 | idade em anos e faixas de 10 anos |
| | `sexo` | C006 | masculino / feminino |
| | `cor_raca` | C009 | branca, preta, amarela, parda, indígena |
| | `regiao`, `uf` | V0001 | UF e grande região |
| | `situacao` | V0026 | urbana / rural |
| Clínicos | `hipertensao` | Q00201 | diagnóstico médico de hipertensão |
| | `diabetes` | Q03001 | diagnóstico médico de diabetes |
| | `peso`, `altura`, `imc`, `imc_categoria` | P00104, P00404 | peso (kg) e altura (cm) referidos; IMC calculado |
| Comportamentais | `tabagismo` | P050, P052 | nunca fumou / ex-fumante / fumante atual |
| | `alcool` | P027 | frequência de consumo de álcool |
| | `atividade_fisica` | P034 | exercício físico nos últimos 3 meses |
| Socioeconômicos | `escolaridade`, `escolaridade_grupo` | VDD004A | nível de instrução (1 a 7 e 4 grupos) |
| | `renda_per_capita`, `faixa_renda` | VDF003, VDF004 | rendimento domiciliar per capita (R$ e faixas de SM) |
| | `plano_saude` | I00102 | tem plano de saúde |
| Desenho amostral | `peso_amostral`, `estrato`, `upa` | V00291, V0024, UPA_PNS | só para as estimativas ponderadas |

Ficaram de fora, para não vazar a resposta, as perguntas que só quem teve AVC responde: idade no primeiro
diagnóstico (Q070), cuidados recebidos (Q07208 a Q07213) e grau de limitação (Q073).

As regras de recodificação estão no notebook 1 e no `src/pns.py`.
