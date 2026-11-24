"""Sistema híbrido (regras críticas + ML) e procura heurística dos seus parâmetros (CP3).

Alarme híbrido = (alguma REGRA CRÍTICA ativa) OU (probabilidade do ML >= limiar t)
  - as regras tratam os casos evidentes (e explicáveis);
  - o ML trata os casos ambíguos, que as regras não veem.

Parâmetros a otimizar (o 'estado' da procura):
  mask  : quais das regras K01..K06 contam como críticas (6 bits)
  t     : limiar de probabilidade do ML
  vib   : limite da regra K01 (mm/s)
  desg  : limite da regra K05 (min)
"""
import numpy as np
import pandas as pd
from src.regras import potencia_W, pico_isolado_vibracao, LIMITE_SOBRECARGA

NOMES = ["K01", "K02", "K03", "K04", "K05", "K06"]
LIM = dict(t=(0.05, 0.95), vib=(4.0, 8.0), desg=(120.0, 240.0))   # intervalos de pesquisa
PADRAO = dict(mask=(1, 1, 1, 1, 1, 1), t=0.5, vib=6.0, desg=200.0)  # manual do técnico


class Contexto:
    """Pré-calcula tudo o que não depende dos parâmetros, para avaliar muito depressa."""
    def __init__(self, df, p):
        self.n = len(df)
        self.p = np.asarray(p, dtype=float)
        pico = pico_isolado_vibracao(df).fillna(False).to_numpy()
        self.vib = df.vibracao_mm_s.to_numpy(float)
        self.desg = df.desgaste_ferramenta_min.to_numpy(float)
        fixo = {
            "K02": (df.vibracao_mm_s > 5.0) & (df.pressao_bar < 5.5) & ~pico,
            "K03": ((df.temp_processo_K - df.temp_ar_K) < 8.6) & (df.velocidade_rpm < 1380),
            "K04": (potencia_W(df) < 3500) | (potencia_W(df) > 9000),
            "K06": df.desgaste_ferramenta_min * df.binario_Nm > df.tipo_qualidade.map(LIMITE_SOBRECARGA),
        }
        self.pico = pico
        self.fixo = {k: v.fillna(False).to_numpy(bool) for k, v in fixo.items()}

    def regras(self, par):
        """Matriz (n, 6) de booleanos com as 6 regras já com os limites do 'par'."""
        k01 = (self.vib > par["vib"]) & ~self.pico
        k05 = self.desg > par["desg"]
        cols = [k01, self.fixo["K02"], self.fixo["K03"], self.fixo["K04"], k05, self.fixo["K06"]]
        return np.column_stack(cols)

    def alarme(self, par):
        R = self.regras(par) & np.asarray(par["mask"], bool)
        return R.any(axis=1) | (self.p >= par["t"])


def matriz_confusao(y, a):
    y = np.asarray(y, bool)
    return dict(TP=int((a & y).sum()), FP=int((a & ~y).sum()), FN=int((~a & y).sum()), TN=int((~a & ~y).sum()))


def custo(y, a, c_fn, c_fp):
    m = matriz_confusao(y, a)
    return c_fn * m["FN"] + c_fp * m["FP"]


def metricas(y, a):
    m = matriz_confusao(y, a)
    P = m["TP"] / max(m["TP"] + m["FP"], 1)
    R = m["TP"] / max(m["TP"] + m["FN"], 1)
    return dict(precisao=P, recall=R, F1=2 * P * R / max(P + R, 1e-9), **m)


# ----------------------------- procura heurística -----------------------------
def aleatorio(rng):
    return dict(mask=tuple(int(x) for x in rng.integers(0, 2, 6)),
                t=rng.uniform(*LIM["t"]), vib=rng.uniform(*LIM["vib"]), desg=rng.uniform(*LIM["desg"]))


def vizinho(par, rng):
    """Vizinho = muda UMA coisa: inverte um bit da máscara ou perturba um número."""
    novo = dict(par)
    k = rng.integers(0, 4)
    if k == 0:
        m = list(par["mask"]); i = rng.integers(0, 6); m[i] = 1 - m[i]; novo["mask"] = tuple(m)
    else:
        nome = ["t", "vib", "desg"][k - 1]
        lo, hi = LIM[nome]
        novo[nome] = float(np.clip(par[nome] + rng.normal(0, 0.08 * (hi - lo)), lo, hi))
    return novo


def subida_encosta(ctx, y, c_fn, c_fp, rng, reinicios=15, passos=150):
    """Hill-climbing com reinícios aleatórios. Devolve (melhor_par, melhor_custo, nº_avaliações)."""
    melhor, melhor_c, aval = None, np.inf, 0
    for _ in range(reinicios):
        atual = aleatorio(rng); ca = custo(y, ctx.alarme(atual), c_fn, c_fp); aval += 1
        for _ in range(passos):
            v = vizinho(atual, rng); cv = custo(y, ctx.alarme(v), c_fn, c_fp); aval += 1
            if cv <= ca:
                atual, ca = v, cv
        if ca < melhor_c:
            melhor, melhor_c = atual, ca
    return melhor, melhor_c, aval


def algoritmo_genetico(ctx, y, c_fn, c_fp, rng, pop=40, geracoes=50):
    """Algoritmo genético: seleção por torneio, cruzamento uniforme, mutação, elitismo."""
    P = [aleatorio(rng) for _ in range(pop)]
    C = [custo(y, ctx.alarme(x), c_fn, c_fp) for x in P]
    aval = pop
    for _ in range(geracoes):
        nova = [P[int(np.argmin(C))]]                       # elitismo
        while len(nova) < pop:
            def torneio():
                i, j = rng.integers(0, pop, 2); return P[i] if C[i] <= C[j] else P[j]
            a, b = torneio(), torneio()
            filho = dict(mask=tuple(a["mask"][i] if rng.random() < .5 else b["mask"][i] for i in range(6)),
                         **{k: (a[k] if rng.random() < .5 else b[k]) for k in LIM})
            if rng.random() < 0.7:
                filho = vizinho(filho, rng)                 # mutação
            nova.append(filho)
        P = nova
        C = [custo(y, ctx.alarme(x), c_fn, c_fp) for x in P]; aval += pop
    i = int(np.argmin(C))
    return P[i], C[i], aval


def pesquisa_aleatoria(ctx, y, c_fn, c_fp, rng, n=2000):
    """Linha de base: sortear n soluções ao acaso e ficar com a melhor."""
    melhor, melhor_c = None, np.inf
    for _ in range(n):
        x = aleatorio(rng); c = custo(y, ctx.alarme(x), c_fn, c_fp)
        if c < melhor_c:
            melhor, melhor_c = x, c
    return melhor, melhor_c, n


# ----------------------------- referência e validação -----------------------------
def grelha_exaustiva(ctx, y, c_fn, c_fp, passos_t=46):
    """Referência: testa TODAS as 64 máscaras x uma grelha de limiares t (vib/desg do manual)."""
    melhor, melhor_c = None, np.inf
    for m in range(64):
        mask = tuple((m >> i) & 1 for i in range(6))
        for t in np.linspace(*LIM["t"], passos_t):
            par = dict(PADRAO, mask=mask, t=float(t)); c = custo(y, ctx.alarme(par), c_fn, c_fp)
            if c < melhor_c:
                melhor, melhor_c = par, c
    return melhor, melhor_c, 64 * passos_t


def melhor_limiar_ml(ctx, y, c_fn, c_fp):
    """Só ML: escolhe o limiar t que minimiza o custo (sem regras)."""
    nulo = dict(PADRAO, mask=(0,) * 6)
    ts = np.linspace(0.02, 0.98, 97)
    cs = [custo(y, ctx.alarme(dict(nulo, t=t)), c_fn, c_fp) for t in ts]
    return dict(nulo, t=float(ts[int(np.argmin(cs))]))


def validar_por_maquina(df, p, y, c_fn, c_fp, k=4, seed=42):
    """Avaliação justa: os parâmetros são otimizados em (k-1)/k das máquinas e testados nas restantes.
    Devolve alarmes fora-da-amostra de 3 abordagens + parâmetros escolhidos em cada fold."""
    from sklearn.model_selection import GroupKFold
    y = np.asarray(y, bool); p = np.asarray(p, float)
    A = {n: np.zeros(len(y), bool) for n in ["Regras", "ML", "Híbrido (hill-climbing)", "Híbrido (genético)"]}
    pars = []
    for f, (tr, te) in enumerate(GroupKFold(k).split(df, y, df.id_maquina)):
        dtr, dte = df.iloc[tr], df.iloc[te]
        ctr, cte = Contexto(dtr, p[tr]), Contexto(dte, p[te])
        rng = np.random.default_rng(seed + f)
        A["Regras"][te] = cte.alarme(dict(PADRAO, t=2.0))
        par_ml = melhor_limiar_ml(ctr, y[tr], c_fn, c_fp); A["ML"][te] = cte.alarme(par_ml)
        ph, _, _ = subida_encosta(ctr, y[tr], c_fn, c_fp, rng); A["Híbrido (hill-climbing)"][te] = cte.alarme(ph)
        pg, _, _ = algoritmo_genetico(ctr, y[tr], c_fn, c_fp, rng); A["Híbrido (genético)"][te] = cte.alarme(pg)
        pars.append(dict(fold=f + 1, ml_t=par_ml["t"], **{f"hill_{k_}": v for k_, v in ph.items()},
                         **{f"ga_{k_}": v for k_, v in pg.items()}))
    return A, pars


def explicar_hibrido(df, ctx, par, i):
    """Explicação do alarme da leitura de posição i: que regras críticas e/ou ML o causaram."""
    ativas = [NOMES[j] for j in range(6) if par["mask"][j] and ctx.regras(par)[i, j]]
    ml = ctx.p[i] >= par["t"]
    if not ativas and not ml:
        return "Sem alarme."
    partes = [f"regra {k}" for k in ativas] + ([f"ML (prob. {ctx.p[i]:.2f} >= {par['t']:.2f})"] if ml else [])
    return f"ALARME {df.id_leitura.iloc[i]} ({df.id_maquina.iloc[i]}): " + " + ".join(partes)
