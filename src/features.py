"""Features para ML. TODAS usam só informação do passado/presente (sem olhar para a frente),
calculadas por máquina, para não haver fuga de informação (data leakage)."""
import numpy as np
import pandas as pd
from src.regras import potencia_W

# Colunas que NUNCA entram como features (são alvos ou só para análise)
PROIBIDAS = ["falha", "tipo_falha", "falha_nas_proximas_24h", "leitura_anomala_sensor",
             "id_leitura", "id_maquina", "timestamp", "ciclo", "tipo_qualidade"]


def _tendencia(x):
    """Declive da reta ajustada aos valores da janela (por ciclo)."""
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        return np.nan
    return np.polyfit(np.arange(len(x)), x, 1)[0]


def construir_features(df):
    d = df.sort_values(["id_maquina", "ciclo"]).copy()
    g = d.groupby("id_maquina", group_keys=False)
    # 1) preencher valores em falta com o último valor conhecido DA MESMA MÁQUINA (só passado)
    base = ["temp_ar_K", "temp_processo_K", "velocidade_rpm", "binario_Nm",
            "desgaste_ferramenta_min", "vibracao_mm_s", "pressao_bar", "corrente_A",
            "horas_desde_manutencao"]
    d[base] = g[base].ffill()
    g = d.groupby("id_maquina", group_keys=False)
    # 2) variáveis físicas derivadas
    d["delta_T"] = d.temp_processo_K - d.temp_ar_K
    d["potencia_W"] = potencia_W(d)
    d["desgaste_x_binario"] = d.desgaste_ferramenta_min * d.binario_Nm
    d["qualidade"] = d.tipo_qualidade.map({"L": 0, "M": 1, "H": 2})
    # 3) features temporais (só passado: rolling termina na leitura atual)
    for col, nome in [("vibracao_mm_s", "vib"), ("pressao_bar", "pres")]:
        d[f"{nome}_media5"] = g[col].transform(lambda s: s.rolling(5, min_periods=1).mean())
        d[f"{nome}_media20"] = g[col].transform(lambda s: s.rolling(20, min_periods=1).mean())
        d[f"{nome}_std5"] = g[col].transform(lambda s: s.rolling(5, min_periods=2).std())
        d[f"{nome}_delta1"] = g[col].diff(1)
        d[f"{nome}_tend10"] = g[col].transform(lambda s: s.rolling(10, min_periods=10).apply(_tendencia, raw=True))
        d[f"{nome}_vs_media20"] = d[col] - d[f"{nome}_media20"]
    cols = base + ["delta_T", "potencia_W", "desgaste_x_binario", "qualidade"] + \
           [c for c in d.columns if c.startswith(("vib_", "pres_"))]
    return d, cols
