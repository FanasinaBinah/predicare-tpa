
import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from src.inference import preparer

GRAINE = 42


def charger_jeu(proc="data/processed", labels="data/raw/labels/labels_fraude.csv"):
    X = pd.read_parquet(Path(proc) / "features.parquet")
    cles = pd.read_parquet(Path(proc) / "cles.parquet")[["facture_id", "date_soin"]]
    lab = pd.read_csv(labels)[["facture_id", "est_fraude", "type_fraude"]]
    d = (X.merge(cles, on="facture_id", validate="1:1")
          .merge(lab, on="facture_id", validate="1:1"))
    d["date_soin"] = pd.to_datetime(d["date_soin"])
    return d


def split_temporel(d, fin_train="2026-01-01", fin_valid="2026-05-01"):
    """Train = passé, validation = choix du seuil, test = futur jamais vu."""
    tr = d[d.date_soin < fin_train]
    va = d[(d.date_soin >= fin_train) & (d.date_soin < fin_valid)]
    te = d[d.date_soin >= fin_valid]
    return tr, va, te


def choisir_seuil(y, p, recall_cible=0.95):
    """Seuil le plus ÉLEVÉ qui garde un rappel >= cible (donc la meilleure précision
    compatible avec l'objectif de rappel du cahier des charges)."""
    prec, rap, seuils = precision_recall_curve(y, p)
    ok = np.where(rap[:-1] >= recall_cible)[0]
    return float(seuils[ok[-1]]) if len(ok) else float(seuils[0])


def metriques(y, p, seuil):
    pred = p >= seuil
    return {"seuil": round(float(seuil), 4),
            "rappel": round(float(recall_score(y, pred)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "taux_alerte": round(float(pred.mean()), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4)}


def rappel_par_type(d, p, seuil):
    f = d.est_fraude.values == 1
    return (pd.Series(p[f] >= seuil).groupby(d.type_fraude.values[f]).mean()
            .round(3).sort_values().to_dict())


def make_model(nom, spw):
    if nom == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                             enable_categorical=True, scale_pos_weight=spw,
                             eval_metric="aucpr", early_stopping_rounds=30,
                             random_state=GRAINE, n_jobs=-1)
    if nom == "hgb":  # modèle de référence sans dépendance externe
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                              class_weight="balanced",
                                              categorical_features="from_dtype",
                                              random_state=GRAINE)
    raise ValueError(f"Modèle inconnu : {nom}")


def entrainer(d, nom="xgb", recall_cible=0.95, version="v1", models_dir="models"):
    tr, va, te = split_temporel(d)
    colonnes = [c for c in d.columns
                if c not in ("facture_id", "date_soin", "est_fraude", "type_fraude")]
    categories = {c: list(d[c].cat.categories) for c in colonnes
                  if str(d[c].dtype) == "category"}
    prep = lambda df: preparer(df[colonnes], categories, colonnes)  # noqa: E731
    Xtr, Xva, Xte = prep(tr), prep(va), prep(te)
    ytr, yva, yte = tr.est_fraude.values, va.est_fraude.values, te.est_fraude.values

    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    model = make_model(nom, spw)
    if nom == "xgb":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    else:
        model.fit(Xtr, ytr)

    p_va, p_te = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
    seuil = choisir_seuil(yva, p_va, recall_cible)  # choisi sur la validation, jamais sur le test
    res = {"validation": metriques(yva, p_va, seuil), "test": metriques(yte, p_te, seuil),
           "rappel_par_type_test": rappel_par_type(te, p_te, seuil)}

    dossier = Path(models_dir) / version
    dossier.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dossier / "model.joblib")
    meta = {"version": version, "modele": nom, "date": datetime.now().isoformat(timespec="seconds"),
            "colonnes": colonnes, "categories": categories, "seuil": seuil,
            "recall_cible": recall_cible, "scale_pos_weight": spw,
            "split": {"train": len(tr), "validation": len(va), "test": len(te)},
            "resultats": res}
    (dossier / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    return model, meta, (te, p_te)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgb", choices=["xgb", "hgb"])
    ap.add_argument("--version", default="v1")
    ap.add_argument("--recall-cible", type=float, default=0.95)
    a = ap.parse_args()
    d = charger_jeu()
    try:
        import mlflow
        mlflow.set_experiment("predicare-fraude")
        run = mlflow.start_run(run_name=f"{a.model}-{a.version}")
    except ImportError:
        mlflow, run = None, None
    _, meta, _ = entrainer(d, a.model, a.recall_cible, a.version)
    if mlflow:
        mlflow.log_params({"modele": a.model, "recall_cible": a.recall_cible,
                           "seuil": meta["seuil"], "version": a.version})
        for k, v in meta["resultats"]["test"].items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_artifact(f"models/{a.version}/metadata.json")
        mlflow.end_run()
    print(json.dumps(meta["resultats"], indent=2, ensure_ascii=False))
