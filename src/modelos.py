"""Modelos, validação por máquina e métricas."""
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

SEED = 42


def criar_modelo(nome):
    imp = SimpleImputer(strategy="median")  # preenche NaN restantes com a mediana do TREINO
    if nome == "LogReg":
        return make_pipeline(imp, StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    if nome == "RF":
        return make_pipeline(imp, RandomForestClassifier(300, class_weight="balanced_subsample",
                                                         min_samples_leaf=3, random_state=SEED, n_jobs=-1))
    if nome == "SVM":
        return make_pipeline(imp, StandardScaler(), SVC(class_weight="balanced", probability=True, random_state=SEED))
    raise ValueError(nome)


def proba_por_maquina(nome, X, y, grupos, k=4):
    """Probabilidades 'out-of-fold': cada leitura é prevista por um modelo que NUNCA viu essa máquina."""
    p = np.zeros(len(y))
    for tr, te in GroupKFold(k).split(X, y, grupos):
        m = criar_modelo(nome)
        m.fit(X.iloc[tr], y.iloc[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


def melhor_limiar(y, p):
    """Limiar que maximiza o F1. ATENÇÃO: escolhido nos mesmos dados -> ligeiramente otimista."""
    pr, rc, t = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-9)
    i = f1[:-1].argmax()
    return dict(limiar=float(t[i]), precisao=float(pr[i]), recall=float(rc[i]), F1=float(f1[i]))
