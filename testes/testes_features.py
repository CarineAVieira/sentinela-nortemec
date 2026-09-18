
# importar os pacotes e o que será usado
import pandas as pd
from sentinela.features import ORDEM_FEATURES, construir
from sentinela.preprocessamento import limpar
import numpy as np

###
ORDEM_FEATURES = (
    "temp_media_6h",
    "temp_max_24h",
    "delta_temp_24h",
    "vib_media_6h",
    "vib_max_24h",
    "corrente_media_6h",
    "pressao",
    "rpm",
    "idade_equipamento_meses",
    "horas_operacao",
    "maquina_risco",
    "operador_senior",
    "turno",
)



## criar dataframe

df = pd.DataFrame({
    "id_maquina": ["M-01"] * 30,
    "timestamp": pd.date_range(
        start="2026-01-01 00:00",
        periods=30,
        freq="h"
    ),
    "temperatura_c": [
        50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 66, 67,
        68, 69, 70, 71, 72, 73,
        74, 75, 76, 77, 78, 79
    ],
    "vibracao_rms": [
        2.0, 2.1, 2.0, 2.2, 2.1, 2.3,
        2.2, 2.4, 2.3, 2.5, 2.4, 2.6,
        2.5, 2.7, 2.6, 2.8, 2.7, 2.9,
        2.8, 3.0, 2.9, 3.1, 3.0, 3.2,
        3.1, 3.3, 3.2, 3.4, 3.3, 3.5
    ],

    "corrente_a": [
        20.0, 20.1, 20.2, 20.3, 20.4, 20.5,
        20.6, 20.7, 20.8, 20.9, 21.0, 21.1,
        21.2, 21.3, 21.4, 21.5, 21.6, 21.7,
        21.8, 21.9, 22.0, 22.1, 22.2, 22.3,
        22.4, 22.5, 22.6, 22.7, 22.8, 22.9
    ],

    "pressao": [4.0] * 30,
    "rpm": [1700] * 30,
    "idade_equipamento_meses": [24] * 30,
    "horas_operacao": list(range(1000, 1030)),
    "id_operador": ["OP-01"] * 30,
    "turno": (
        [1] * 8 +
        [2] * 8 +
        [3] * 8 +
        [1] * 6
    )
})

## usar a função de construção
features_df = construir(df)


# Comparação manual para verificar o efeito da alteração na média de 6 horas
###
original = df.copy()
alterado = df.copy()

alterado.loc[20, "temperatura_c"] = 90

feat_original = construir(original)
feat_alterado = construir(alterado)

comparacao_teste1 = pd.DataFrame({
    "timestamp": df["timestamp"],
    "temperatura_original": original["temperatura_c"],
    "temperatura_alterada": alterado["temperatura_c"],
    "media_6h_original": feat_original["temp_media_6h"],
    "media_6h_alterada": feat_alterado["temp_media_6h"],
})

comparacao_teste1["diferenca"] = (
    comparacao_teste1["media_6h_alterada"]
    - comparacao_teste1["media_6h_original"]
)

comparacao_teste1.loc[15:21]



# Teste 1: teste de vazamento de dados
def test_temp_media_6h_nao_usa_dados_futuros():
    original = df.copy()
    alterado = df.copy()

    # Alterado SOMENTE uma leitura futura:
    # índice 20 = 20:00
    alterado.loc[20, "temperatura_c"] = 90

    # função ORIGINAL do Sentinela
    feat_original = construir(original)
    feat_alterado = construir(alterado)

    # A leitura das 20:00 ainda não aconteceu nesse instante.
    assert (
        feat_original.loc[18, "temp_media_6h"]
        ==
        feat_alterado.loc[18, "temp_media_6h"]
    )
test_temp_media_6h_nao_usa_dados_futuros()


# Teste 2: o Y entra nas features
def test_maquina_risco_nao_usa_target_futuro():

    original = df.copy()
    original["falha_72h"] = 0
    alterado = original.copy()

    # Alterado apenas o Y em uma observação futura
    alterado.loc[20, "falha_72h"] = 1
    # função ORIGINAL
    feat_original = construir(original)
    feat_alterado = construir(alterado)

    # O risco no instante 18 não deveria mudar
    # porque alteramos apenas o target do instante 20.
    assert (
        feat_original.loc[18, "maquina_risco"]
        ==
        feat_alterado.loc[18, "maquina_risco"]
    )

test_maquina_risco_nao_usa_target_futuro()


# Teste 3: a ordem das linhas não pode mudar as features
def test_features_nao_dependem_da_ordem_das_linhas():
    original = df.copy()

    # Mesmas observações, mas em outra ordem.
    embaralhado = df.sample(
        frac=1,
        random_state=42
    )

    feat_original = construir(original)
    feat_embaralhado = construir(embaralhado)

    # Reordenado somente o RESULTADO para fazer a comparação
    # das mesmas observações.
    feat_embaralhado = feat_embaralhado.loc[feat_original.index]

    pd.testing.assert_frame_equal(
        feat_original,
        feat_embaralhado
    )
test_features_nao_dependem_da_ordem_das_linhas()


# Teste 4: verificar quantidade e ordem das features
def test_features_possuem_colunas_na_ordem_esperada():

    resultado = construir(df.copy())

    assert list(resultado.columns) == list(ORDEM_FEATURES)
    assert resultado.shape[1] == len(ORDEM_FEATURES)

test_features_possuem_colunas_na_ordem_esperada()


# Teste 5: preservar o índice das observações
def test_features_preservam_indice():
    original = df.copy()
    resultado = construir(original)

    assert all(resultado.index == original.index)

test_features_preservam_indice()