# importar bibliotecas necessárias
import pandas as pd
import numpy as np
from sentinela.preprocessamento import limpar


# criar df
df_pre = pd.DataFrame({
    "timestamp": [
        "2026-01-01 00:00",
        "2026-01-01 01:00",
        "2026-01-01 02:00",
        "2026-01-01 03:00",
    ],
    "id_maquina": [
        " m-01 ",
        "M-01",
        "m-02",
        " M-03 "
    ],
    "id_operador": [
        " op-01 ",
        "OP-02",
        "op-07",
        " OP-03 "
    ],
    "unidade_pressao": [
        " BAR ",
        "bar",
        " PSI ",
        "bar"
    ],
    "pressao": [
        4.0,
        4.1,
        58.0,
        3.9
    ],
    "rpm": [
        1700,
        0,
        1750,
        1650
    ],
    "vibracao_rms": [
        2.0,
        2.5,
        np.nan,
        3.0
    ],
    "turno": [
        1,
        2,
        3,
        1
    ],
    "idade_equipamento_meses": [
        24,
        30,
        36,
        48
    ]
})


## aplicação da fun~~ao limpar
df_limpo = limpar(df_pre)

# Teste 1: Imputação de 0 em valores ausentes de vibracao_rms
def test_imputacao_vibracao_rms():
    resultado = limpar(df_pre.copy())
    # não pode restar NaN
    assert resultado["vibracao_rms"].isna().sum() == 0
    # o valor ausente deve ter sido substituído por 0
    assert (resultado["vibracao_rms"] == 0).any()

# Teste 2: Criação da coluna operador_cod
def test_criacao_operador_cod():
    assert "operador_cod" in df_limpo.columns
test_criacao_operador_cod()


# Teste 3: Normalização de textos (id_maquina, id_operador, unidade_pressao)
def test_limpar_normaliza_textos():

    resultado = limpar(df_pre.copy())

    assert resultado.loc[0, "id_maquina"] == "M-01"
    assert resultado.loc[0, "id_operador"] == "OP-01"
    assert resultado.loc[0, "unidade_pressao"] == "bar"
test_limpar_normaliza_textos()


# Teste 4: remoção de registros com motor parado
def test_remocao_motor_parado():
    resultado = limpar(df_pre.copy())
    assert (resultado["rpm"] >= 1).all()
test_remocao_motor_parado()


# Teste 5: conversão dos tipos
def test_conversao_tipos():

    resultado = limpar(df_pre.copy())
    assert pd.api.types.is_datetime64_any_dtype(
        resultado["timestamp"]
    )
    assert pd.api.types.is_integer_dtype(
        resultado["turno"]
    )
    assert pd.api.types.is_integer_dtype(
        resultado["idade_equipamento_meses"]
    )
    assert pd.api.types.is_integer_dtype(
        resultado["operador_cod"]
    )
test_conversao_tipos()


# Teste 6: limpar não deve modificar o DataFrame original
def test_limpar_nao_modifica_dataframe_original():
    entrada = df_pre.copy()
    original = entrada.copy(deep=True)
    limpar(entrada)
    pd.testing.assert_frame_equal(
        entrada,
        original
    )
test_limpar_nao_modifica_dataframe_original()



# Teste 7: Conversão de pressão para unidade padrão (BAR)
# Teste 7: valores equivalentes em BAR e PSI
# 4 bar ≈ 58,015 psi

def test_pressao_equivalente_bar_psi():
    leitura_bar = df_pre.iloc[[0]].copy()
    leitura_psi = df_pre.iloc[[0]].copy()

    leitura_bar["pressao"] = 4.0
    leitura_bar["unidade_pressao"] = "bar"

    leitura_psi["pressao"] = 58.015
    leitura_psi["unidade_pressao"] = "psi"

    resultado_bar = limpar(leitura_bar)
    resultado_psi = limpar(leitura_psi)

    assert np.isclose(
        resultado_bar.loc[0, "pressao"],
        resultado_psi.loc[0, "pressao"],
        rtol=0.01
    )
test_pressao_equivalente_bar_psi()