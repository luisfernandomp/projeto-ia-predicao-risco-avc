# Aplicação de IA para Predição do Risco de AVC (PNS 2019)

Projeto da disciplina de Inteligência Artificial – 7º semestre de Ciência da Computação
Faculdade de Computação e Informática – Universidade Presbiteriana Mackenzie
Prof. Dr. Ivan Carlos Alcântara de Oliveira

| Integrante | RA |
|---|---|
| Luana Miho Yasuda | 10396439 |
| Luis Fernando de Mesquita Pereira | 10410686 |
| Richard Barbosa Sanches | 10420179 |
| Totti Ferreira Gomes | 10428490 |
| Gabriel Resende Menezes | 10419003 |

## Sobre o projeto

Usamos os microdados da Pesquisa Nacional de Saúde (PNS) 2019, do IBGE, para classificar o diagnóstico médico
referido de AVC em adultos de 40 anos ou mais, a partir de atributos demográficos, clínicos, comportamentais e
socioeconômicos. Vamos comparar Regressão Logística, Random Forest e XGBoost, com tratamento do desbalanceamento
de classes, e usar SHAP para explicar os fatores mais importantes. O projeto está alinhado à ODS 3 (Saúde e
Bem-Estar).

Nesta entrega (1º bimestre) estão prontas a obtenção e preparação dos dados e a análise exploratória.

## Organização

- [docs/artigo/artigo.md](docs/artigo/artigo.md) – artigo do projeto
- [data/README.md](data/README.md) – descrição do dataset (os microdados não ficam no repositório)
- [notebooks/01_obtencao_e_preparacao.ipynb](notebooks/01_obtencao_e_preparacao.ipynb) – download, leitura, recorte da amostra e base analítica
- [notebooks/02_analise_exploratoria.ipynb](notebooks/02_analise_exploratoria.ipynb) – análise exploratória, com os resultados
- [src/pns.py](src/pns.py) – download do IBGE, leitura do arquivo de largura fixa e montagem da base
- [src/pesos.py](src/pesos.py) – prevalência ponderada com IC 95% (desenho amostral da PNS)
- [src/graficos.py](src/graficos.py) – gráficos usados na análise
- [results/figures/](results/figures/) e [results/tables/](results/tables/) – figuras e tabelas geradas pelos notebooks

## Como rodar

Precisa de Python 3.11+ e uns 600 MB livres em disco (os microdados descompactados têm ~455 MB).

```bash
git clone https://github.com/luisfernandomp/projeto-ia-predicao-risco-avc.git
cd projeto-ia-predicao-risco-avc

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt

jupyter lab
```

Rodar primeiro o notebook `01_obtencao_e_preparacao.ipynb` (ele baixa os dados do IBGE, ~30 MB) e depois o
`02_analise_exploratoria.ipynb`. Se preferir, `python -m src.pns` gera a base analítica direto pela linha de
comando.

## Resultados até agora

- 54.987 adultos de 40+ anos, 1.844 com AVC. Prevalência ponderada de 3,20% (IC 95%: 2,96% a 3,46%); ~29
  negativos para cada positivo.
- Idade é o fator mais forte: 1,2% aos 40-49 anos e 8,5% aos 80+.
- Hipertensão (5,7% contra 1,6%) e diabetes (6,4% contra 2,7%) têm associação clara, e há um gradiente por
  escolaridade (4,6% sem instrução contra 1,4% com superior completo).
- Não beber e não fazer exercício aparecem associados a mais AVC, provavelmente por causalidade reversa (a
  pessoa muda o hábito depois do AVC), o que é uma limitação de dados transversais.
- Só 0,16% dos registros têm algum atributo ausente.

![Prevalência de AVC por faixa etária e sexo](results/figures/01_prevalencia_idade_sexo.png)

## Ética

Os dados são públicos e anonimizados. Os microdados não são redistribuídos aqui, só os scripts que baixam e
processam. Os resultados são associações em nível populacional e não servem para diagnóstico individual.

## Licença

MIT (ver [LICENSE](LICENSE)). Os microdados da PNS são do IBGE.
