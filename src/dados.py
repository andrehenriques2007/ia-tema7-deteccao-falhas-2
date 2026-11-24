"""Carregamento dos dados do Tema 7."""
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SENSORES = ["temp_ar_K", "temp_processo_K", "velocidade_rpm", "binario_Nm",
            "desgaste_ferramenta_min", "vibracao_mm_s", "pressao_bar",
            "corrente_A", "horas_desde_manutencao"]


def carregar_leituras() -> pd.DataFrame:
    """Lê o CSV (separador ';', decimal ',', codificação cp1252) e ordena por máquina/tempo."""
    df = pd.read_csv(RAIZ / "data" / "leituras_sensores.csv",
                     sep=";", decimal=",", encoding="cp1252",
                     parse_dates=["timestamp"])
    # tipo_falha vem como 'TWF — desgaste da ferramenta': ficamos só com o código
    df["tipo_falha"] = df["tipo_falha"].str.split(" ").str[0].replace({"nenhuma": "OK"})
    df["leitura_anomala_sensor"] = df["leitura_anomala_sensor"].eq("sim")
    return df.sort_values(["id_maquina", "ciclo"]).reset_index(drop=True)
