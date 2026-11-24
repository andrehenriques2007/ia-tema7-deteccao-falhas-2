"""Módulo baseado em conhecimento (regras K01-K08 do 'manual do técnico').

Cada regra tem: id, descrição, função que devolve uma série booleana e uma
explicação em linguagem natural. Comparações com valores em falta (NaN) dão False:
uma regra não dispara se faltar um dos sensores de que depende.
"""
import numpy as np
import pandas as pd

LIMITE_SOBRECARGA = {"L": 11000, "M": 12000, "H": 13000}


def potencia_W(df):
    return df["binario_Nm"] * df["velocidade_rpm"] * 2 * np.pi / 60


def pico_isolado_vibracao(df):
    """K08: vibração sobe > 4 mm/s numa leitura e volta ao normal na seguinte.
    NOTA: usa a leitura SEGUINTE, logo em tempo real só se confirma com 1 ciclo de atraso."""
    g = df.groupby("id_maquina")["vibracao_mm_s"]
    ant, seg = g.shift(1), g.shift(-1)
    return ((df["vibracao_mm_s"] - ant) > 4) & ((seg - ant).abs() < 2)


REGRAS = {
    "K01": ("Vibração > 6 mm/s",
            "Alarme de vibração: possível degradação de rolamento",
            lambda d: d.vibracao_mm_s > 6.0),
    "K02": ("Vibração > 5 e pressão < 5,5 bar",
            "Rolamento degradado com perda de lubrificação",
            lambda d: (d.vibracao_mm_s > 5.0) & (d.pressao_bar < 5.5)),
    "K03": ("ΔT < 8,6 K e rpm < 1380",
            "Dissipação de calor insuficiente",
            lambda d: ((d.temp_processo_K - d.temp_ar_K) < 8.6) & (d.velocidade_rpm < 1380)),
    "K04": ("Potência < 3500 W ou > 9000 W",
            "Potência fora da janela de operação",
            lambda d: (potencia_W(d) < 3500) | (potencia_W(d) > 9000)),
    "K05": ("Desgaste da ferramenta > 200 min",
            "Ferramenta perto do fim de vida",
            lambda d: d.desgaste_ferramenta_min > 200),
    "K06": ("Desgaste × binário acima do limiar (L/M/H)",
            "Risco de sobrecarga",
            lambda d: d.desgaste_ferramenta_min * d.binario_Nm > d.tipo_qualidade.map(LIMITE_SOBRECARGA)),
    "K07": ("Horas desde manutenção > 240",
            "Manutenção preventiva em atraso",
            lambda d: d.horas_desde_manutencao > 240),
}
REGRAS_VIBRACAO = ["K01", "K02"]  # as que o K08 pode anular quando o pico é isolado


def aplicar_regras(df, usar_k08=True):
    """Devolve um DataFrame de booleanos (uma coluna por regra) + 'k08_pico' + 'alarme'."""
    r = pd.DataFrame({k: f(df).fillna(False).astype(bool) for k, (_, _, f) in REGRAS.items()}, index=df.index)
    r["k08_pico"] = pico_isolado_vibracao(df).fillna(False)
    if usar_k08:
        for k in REGRAS_VIBRACAO:
            r[k] = r[k] & ~r["k08_pico"]
    r["alarme"] = r[list(REGRAS)].any(axis=1)
    return r


def explicar(df, r, i):
    """Explicação em texto do raciocínio do sistema para a leitura de índice i."""
    ativas = [k for k in REGRAS if r.loc[i, k]]
    if not ativas:
        return "Sem alarme: nenhuma regra disparou."
    linhas = [f"ALARME na leitura {df.loc[i, 'id_leitura']} ({df.loc[i, 'id_maquina']}):"]
    linhas += [f"  - {k}: {REGRAS[k][0]} -> {REGRAS[k][1]}" for k in ativas]
    return "\n".join(linhas)
