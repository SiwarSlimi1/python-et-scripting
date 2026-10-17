"""
Nettoyage et préparation des données - HR Analytics : Job change of Data Scientists
Fichiers attendus : aug_train.csv, aug_test.csv (+ sample_submission.csv, non utilisé ici)

Usage :
    python data_preparation.py --input-dir data --output-dir output

Sorties : train_clean.csv, test_clean.csv et visualisations.pdf (tous les graphiques).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas de fenêtre : on enregistre les figures en PNG
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
COLORS = {0: "#4C78A8", 1: "#E4572E"}  # 0 = reste, 1 = cherche à changer d'emploi

# --------------------------------------------------------------------------- #
# 1. Chargement
# --------------------------------------------------------------------------- #

def load_data(path: str) -> pd.DataFrame:
    """Charge un CSV et affiche sa forme."""
    df = pd.read_csv(path)
    print(f"[load] {Path(path).name} -> {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


# --------------------------------------------------------------------------- #
# 2. Inspection
# --------------------------------------------------------------------------- #

def summarize(df: pd.DataFrame, name: str = "df") -> None:
    """Résumé rapide : types, valeurs manquantes (%), doublons."""
    print(f"\n===== Résumé de {name} {df.shape} =====")
    missing = df.isna().mean().mul(100).round(2)
    summary = pd.DataFrame({"dtype": df.dtypes.astype(str),
                            "n_unique": df.nunique(),
                            "%_manquant": missing})
    print(summary.to_string())
    print(f"Doublons (lignes entières) : {df.duplicated().sum()}")


# --------------------------------------------------------------------------- #
# 3. Nettoyage de base
# --------------------------------------------------------------------------- #

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Noms de colonnes en minuscules + correction de la faute 'relevent'."""
    df = df.copy()
    df.columns = [c.strip().lower().replace("relevent", "relevant") for c in df.columns]
    return df


def remove_duplicates(df: pd.DataFrame, id_col: str = "enrollee_id") -> pd.DataFrame:
    """Supprime les doublons exacts et les doublons d'identifiant."""
    before = len(df)
    df = df.drop_duplicates().drop_duplicates(subset=id_col, keep="first")
    print(f"[duplicates] {before - len(df)} ligne(s) supprimée(s)")
    return df.reset_index(drop=True)


def strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Enlève les espaces superflus dans les colonnes texte."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
    return df


# --------------------------------------------------------------------------- #
# 4. Nettoyage / conversion colonne par colonne
# --------------------------------------------------------------------------- #

def clean_city(df: pd.DataFrame) -> pd.DataFrame:
    """'city_103' -> 103 (entier)."""
    df = df.copy()
    df["city"] = df["city"].str.replace("city_", "", regex=False).astype(int)
    return df


def clean_experience(df: pd.DataFrame) -> pd.DataFrame:
    """'<1' -> 0 ; '>20' -> 21 ; sinon conversion numérique."""
    df = df.copy()
    df["experience"] = (df["experience"]
                        .replace({"<1": "0", ">20": "21"})
                        .astype(float))
    return df


def clean_last_new_job(df: pd.DataFrame) -> pd.DataFrame:
    """'never' -> 0 ; '>4' -> 5 ; sinon conversion numérique."""
    df = df.copy()
    df["last_new_job"] = (df["last_new_job"]
                          .replace({"never": "0", ">4": "5"})
                          .astype(float))
    return df


def clean_company_size(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonise les libellés ('10/49' -> '10-49') et crée une version ordinale."""
    df = df.copy()
    df["company_size"] = df["company_size"].replace({"10/49": "10-49"})
    order = ["<10", "10-49", "50-99", "100-500", "500-999",
             "1000-4999", "5000-9999", "10000+"]
    df["company_size_ord"] = df["company_size"].map({v: i for i, v in enumerate(order)})
    return df


def clean_education_level(df: pd.DataFrame) -> pd.DataFrame:
    """Version ordinale du niveau d'étude."""
    df = df.copy()
    order = ["Primary School", "High School", "Graduate", "Masters", "Phd"]
    df["education_level_ord"] = df["education_level"].map({v: i for i, v in enumerate(order)})
    return df


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """relevant_experience -> 1 / 0."""
    df = df.copy()
    df["relevant_experience"] = (df["relevant_experience"]
                                 .str.replace("relevent", "relevant", regex=False)
                                 .map({"Has relevant experience": 1,
                                       "No relevant experience": 0}))
    return df


# --------------------------------------------------------------------------- #
# 5. Valeurs manquantes
# --------------------------------------------------------------------------- #

CAT_UNKNOWN = ["gender", "enrolled_university", "education_level",
               "major_discipline", "company_size", "company_type"]
NUM_MEDIAN = ["experience", "last_new_job", "education_level_ord", "company_size_ord"]


def add_missing_flags(df: pd.DataFrame, cols=("company_size",)) -> pd.DataFrame:
    """Indicateur 0/1 de valeur manquante (l'absence d'information est informative).

    Uniquement pour company_size : cette colonne est transformée en variable ordinale
    puis imputée par la médiane, donc l'information "manquant" serait sinon perdue.
    Les autres colonnes catégorielles reçoivent une modalité "Unknown" (one-hot),
    ce qui porte déjà cette information : un flag serait un doublon exact."""
    df = df.copy()
    for c in cols:
        df[f"{c}_missing"] = df[c].isna().astype(int)
    return df


def fit_imputer(train: pd.DataFrame) -> dict:
    """Calcule les médianes sur le TRAIN uniquement (évite la fuite de données)."""
    return {c: train[c].median() for c in NUM_MEDIAN if c in train}


def impute_missing(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """Catégorielles -> 'Unknown' ; numériques -> médiane du train."""
    df = df.copy()
    for c in CAT_UNKNOWN:
        df[c] = df[c].fillna("Unknown")
    for c, m in medians.items():
        df[c] = df[c].fillna(m)
    return df


# --------------------------------------------------------------------------- #
# 6. Valeurs aberrantes
# --------------------------------------------------------------------------- #

def fit_outlier_bounds(train: pd.DataFrame, col: str = "training_hours", k: float = 1.5):
    """Bornes IQR calculées sur le train."""
    q1, q3 = train[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    return max(0, q1 - k * iqr), q3 + k * iqr  # les heures ne peuvent pas être négatives


def clip_outliers(df: pd.DataFrame, bounds, col: str = "training_hours") -> pd.DataFrame:
    """Winsorisation : ramène les valeurs extrêmes aux bornes."""
    df = df.copy()
    low, high = bounds
    n = ((df[col] < low) | (df[col] > high)).sum()
    df[col] = df[col].clip(low, high)
    print(f"[outliers] {col}: {n} valeur(s) ramenée(s) dans [{low:.0f}, {high:.0f}]")
    return df


# --------------------------------------------------------------------------- #
# 7. Encodage des variables catégorielles
# --------------------------------------------------------------------------- #

def one_hot_encode(df: pd.DataFrame, columns=None, reference_columns=None) -> pd.DataFrame:
    """One-hot encoding. `reference_columns` aligne le test sur les colonnes du train."""
    columns = columns or ["gender", "enrolled_university", "major_discipline", "company_type"]
    out = pd.get_dummies(df, columns=columns, dtype=int)
    if reference_columns is not None:
        out = out.reindex(columns=reference_columns, fill_value=0)
    return out


def frequency_encode_city(df: pd.DataFrame, freq_map: pd.Series) -> pd.DataFrame:
    """Encode la ville par sa fréquence (123 modalités -> évite 123 colonnes)."""
    df = df.copy()
    df["city_freq"] = df["city"].map(freq_map).fillna(0)
    return df


# --------------------------------------------------------------------------- #
# 8. Visualisations
# --------------------------------------------------------------------------- #

_PDF = None  # PDF unique qui reçoit toutes les figures


def _save(fig, out_dir: Path, name: str) -> None:
    """Ajoute la figure au PDF unique (une page par graphique)."""
    _PDF.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {name}")


def plot_target_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """Déséquilibre de la cible."""
    counts = df["target"].value_counts().sort_index()
    labels = ["0 : ne cherche pas à changer", "1 : cherche à changer"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=[COLORS[0], COLORS[1]])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v / counts.sum():.1%})",
                ha="center", va="bottom")
    ax.set_title("Distribution de la variable cible")
    ax.set_ylabel("Nombre de candidats")
    ax.set_ylim(0, counts.max() * 1.18)
    _save(fig, out_dir, "01_distribution_cible")


def plot_missing_values(df: pd.DataFrame, out_dir: Path) -> None:
    """% de valeurs manquantes par colonne (avant imputation)."""
    df = df[[c for c in df.columns if not c.endswith("_ord")]]  # on ignore les colonnes dérivées
    miss = df.isna().mean().mul(100)
    miss = miss[miss > 0].sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(miss.index, miss.values, color="#7A5195")
    for b, v in zip(bars, miss.values):
        ax.text(v + 0.4, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center")
    ax.set_xlabel("% de valeurs manquantes")
    ax.set_title("Valeurs manquantes par colonne (avant imputation)")
    ax.set_xlim(0, miss.max() * 1.15)
    _save(fig, out_dir, "02_valeurs_manquantes")


def plot_missing_pattern(df: pd.DataFrame, out_dir: Path) -> None:
    """Carte des valeurs manquantes + taux de cible selon présence/absence."""
    cols = [c for c in df.columns if df[c].isna().any() and not c.endswith("_ord")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})
    sample = df[cols].sample(min(600, len(df)), random_state=0).isna().astype(int)
    sns.heatmap(sample, cmap=["#E8EEF4", "#7A5195"], cbar=False, yticklabels=False, ax=axes[0])
    axes[0].set_title("Motif des manquants (échantillon de 600 lignes, violet = manquant)")
    axes[0].tick_params(axis="x", rotation=45)

    rows = []
    for c in cols:
        r = df.groupby(df[c].isna())["target"].mean()
        rows.append({"colonne": c, "Renseigné": r.get(False, np.nan), "Manquant": r.get(True, np.nan)})
    rate = pd.DataFrame(rows).set_index("colonne")
    rate.plot.barh(ax=axes[1], color=[COLORS[0], COLORS[1]])
    axes[1].set_xlabel("Taux de target = 1")
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, 0.55)
    axes[1].legend(loc="lower right")
    axes[1].set_title("Le fait d'être manquant est-il lié à la cible ?")
    axes[1].axvline(df["target"].mean(), color="k", ls="--", lw=1)
    _save(fig, out_dir, "03_motif_manquants")


def plot_numeric_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    """Distribution des variables numériques selon la cible."""
    cols = ["city_development_index", "training_hours", "experience", "last_new_job"]
    titles = ["Indice de développement de la ville", "Heures de formation",
              "Années d'expérience (0 = <1, 21 = >20)", "Années depuis le dernier emploi (0 = never, 5 = >4)"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, c, t in zip(axes.ravel(), cols, titles):
        discrete = df[c].nunique() <= 25
        for k in (0, 1):
            sub = df.loc[df["target"] == k, c].dropna()
            if discrete:
                vc = sub.value_counts(normalize=True).sort_index()
                ax.plot(vc.index, vc.values, marker="o", color=COLORS[k], label=f"target = {k}")
            else:
                sns.kdeplot(sub, ax=ax, color=COLORS[k], fill=True, alpha=0.3, label=f"target = {k}",
                            clip=(sub.min(), sub.max()))
        ax.set_title(t)
        ax.set_xlabel("")
        ax.legend()
    fig.suptitle("Variables numériques selon la cible", fontsize=14)
    fig.tight_layout()
    _save(fig, out_dir, "04_distributions_numeriques")


def plot_outliers_before_after(before: pd.DataFrame, after: pd.DataFrame, out_dir: Path,
                               col: str = "training_hours") -> None:
    """Boxplots avant/après traitement des valeurs aberrantes."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, d, t in zip(axes, (before, after), ("Avant (brut)", "Après (winsorisé)")):
        sns.boxplot(y=d[col], ax=ax, color="#9ECAE1")
        ax.set_title(f"{t} - max = {d[col].max():.0f}")
        ax.set_ylabel(col)
    fig.suptitle("Valeurs aberrantes de training_hours (méthode IQR)", y=1.04)
    _save(fig, out_dir, "05_outliers_avant_apres")


def plot_target_rate_by_category(df: pd.DataFrame, out_dir: Path) -> None:
    """Taux de target = 1 par modalité (les manquants sont affichés explicitement)."""
    cols = ["gender", "relevant_experience", "enrolled_university", "education_level",
            "major_discipline", "company_size", "company_type", "last_new_job"]
    order = {
        "education_level": ["Primary School", "High School", "Graduate", "Masters", "Phd"],
        "company_size": ["<10", "10-49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"],
    }
    base = df["target"].mean()
    fig, axes = plt.subplots(4, 2, figsize=(13, 17))
    for ax, c in zip(axes.ravel(), cols):
        tmp = df[[c, "target"]].copy()
        if c == "relevant_experience":
            tmp[c] = tmp[c].map({1: "Oui", 0: "Non"})
        elif c == "last_new_job":
            tmp[c] = tmp[c].map(lambda v: {0: "never", 5: ">4"}.get(v, None if pd.isna(v) else str(int(v))))
        tmp[c] = tmp[c].astype("object").where(tmp[c].notna(), "Manquant").astype(str)
        g = tmp.groupby(c)["target"].agg(["mean", "size"])
        natural = {**order, "last_new_job": ["never", "1", "2", "3", "4", ">4"],
                   "relevant_experience": ["Non", "Oui"]}
        idx = [v for v in natural.get(c, []) if v in g.index]
        idx += [v for v in g.drop(index="Manquant", errors="ignore").sort_values("mean").index if v not in idx]
        if "Manquant" in g.index:
            idx.append("Manquant")  # toujours en dernier
        g = g.loc[idx]
        bars = ax.bar(g.index, g["mean"], color=["#999999" if i == "Manquant" else "#4C78A8" for i in g.index])
        for b, (m, n) in zip(bars, g.values):
            ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.0%}\n(n={int(n)})", ha="center", va="bottom", fontsize=8)
        ax.axhline(base, color="#E4572E", ls="--", lw=1, label=f"moyenne globale {base:.0%}")
        ax.set_title(c)
        ax.set_ylabel("Taux target = 1")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, g["mean"].max() * 1.35)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Taux de candidats cherchant à changer d'emploi, par modalité", fontsize=14, y=1.0)
    fig.tight_layout()
    _save(fig, out_dir, "06_taux_cible_par_categorie")


def plot_target_rate_by_numeric(df: pd.DataFrame, out_dir: Path) -> None:
    """Taux de target = 1 par tranche de variable numérique."""
    specs = [("city_development_index", 8, "Indice de développement (déciles)"),
             ("training_hours", 8, "Heures de formation (déciles)"),
             ("experience", None, "Années d'expérience")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (c, q, t) in zip(axes, specs):
        d = df[[c, "target"]].dropna()
        key = pd.qcut(d[c], q, duplicates="drop") if q else d[c]
        g = d.groupby(key, observed=True)["target"].mean()
        x = [str(i) for i in g.index] if q else g.index.astype(int)
        ax.plot(range(len(g)), g.values, marker="o", color=COLORS[1])
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(x, rotation=45 if q else 0, ha="right" if q else "center", fontsize=8)
        ax.axhline(df["target"].mean(), color="k", ls="--", lw=1)
        ax.set_ylim(0, 0.65)  # même échelle partout : évite d'exagérer de petites variations
        ax.set_title(t)
        ax.set_ylabel("Taux target = 1")
    fig.tight_layout()
    _save(fig, out_dir, "07_taux_cible_par_numerique")


def plot_correlation_matrix(df: pd.DataFrame, out_dir: Path) -> None:
    """Matrice de corrélation des variables numériques du jeu nettoyé."""
    cols = ["target", "city_development_index", "city_freq", "relevant_experience", "experience",
            "last_new_job", "training_hours", "education_level_ord", "company_size_ord",
            "company_size_missing", "company_type_Unknown", "major_discipline_Unknown", "gender_Unknown"]
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Matrice de corrélation (données nettoyées)")
    _save(fig, out_dir, "08_matrice_correlation")


def plot_target_correlations(df: pd.DataFrame, out_dir: Path, top: int = 15) -> None:
    """Variables les plus corrélées à la cible (toutes colonnes finales)."""
    corr = df.drop(columns=["enrollee_id"]).corr()["target"].drop("target")
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index[:top])[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(corr.index, corr.values, color=[COLORS[1] if v > 0 else COLORS[0] for v in corr.values])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"Top {top} des corrélations avec la cible")
    ax.set_xlabel("Corrélation de Pearson")
    _save(fig, out_dir, "09_correlations_avec_cible")


def plot_train_test_consistency(train: pd.DataFrame, test: pd.DataFrame, out_dir: Path) -> None:
    """Le test a-t-il la même distribution que le train ? (contrôle de dérive)"""
    cols = ["city_development_index", "experience", "training_hours", "last_new_job"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, c in zip(axes, cols):
        sns.kdeplot(train[c], ax=ax, label="train", fill=True, alpha=0.3, color="#4C78A8")
        sns.kdeplot(test[c], ax=ax, label="test", fill=True, alpha=0.3, color="#F58518")
        ax.set_title(c)
        ax.set_xlabel("")
        ax.legend()
    fig.suptitle("Cohérence train / test après nettoyage")
    fig.tight_layout()
    _save(fig, out_dir, "10_coherence_train_test")


def generate_all_plots(train_basic: pd.DataFrame, train_clean: pd.DataFrame,
                       test_clean: pd.DataFrame, out_dir: Path) -> None:
    """Génère tous les graphiques dans UN SEUL PDF (visualisations.pdf).
    train_basic = train nettoyé mais AVANT imputation / winsorisation."""
    global _PDF
    out_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_dir / "visualisations.pdf") as pdf:
        _PDF = pdf
        plot_target_distribution(train_basic, out_dir)
        plot_missing_values(train_basic, out_dir)
        plot_missing_pattern(train_basic, out_dir)
        plot_numeric_distributions(train_basic, out_dir)
        plot_outliers_before_after(train_basic, train_clean, out_dir)
        plot_target_rate_by_category(train_basic, out_dir)
        plot_target_rate_by_numeric(train_basic, out_dir)
        plot_correlation_matrix(train_clean, out_dir)
        plot_target_correlations(train_clean, out_dir)
        plot_train_test_consistency(train_clean, test_clean, out_dir)
    _PDF = None


# --------------------------------------------------------------------------- #
# 9. Pipeline complet
# --------------------------------------------------------------------------- #

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Étapes sans apprentissage (identiques pour train et test)."""
    df = standardize_columns(df)
    df = strip_strings(df)
    df = remove_duplicates(df)
    df = clean_city(df)
    df = clean_experience(df)
    df = clean_last_new_job(df)
    df = clean_company_size(df)
    df = clean_education_level(df)
    df = encode_binary(df)
    df = add_missing_flags(df)
    return df


def prepare_datasets(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    """Applique le nettoyage au train et au test avec les paramètres appris sur le train."""
    train = basic_cleaning(train_raw)
    test = basic_cleaning(test_raw)

    # Paramètres appris sur le train uniquement
    medians = fit_imputer(train)
    bounds = fit_outlier_bounds(train)
    city_freq = train["city"].value_counts(normalize=True)

    train, test = impute_missing(train, medians), impute_missing(test, medians)
    train, test = clip_outliers(train, bounds), clip_outliers(test, bounds)
    train, test = (frequency_encode_city(d, city_freq) for d in (train, test))

    # Colonnes catégorielles d'origine devenues redondantes (versions ordinales/numériques gardées)
    drop_cols = ["city", "education_level", "company_size"]
    train, test = train.drop(columns=drop_cols), test.drop(columns=drop_cols)

    train = one_hot_encode(train)
    test = one_hot_encode(test, reference_columns=[c for c in train.columns if c != "target"])
    return train, test


def main(input_dir: str, output_dir: str, plots: bool = True) -> None:
    in_dir, out_dir = Path(input_dir), Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw = load_data(in_dir / "aug_train.csv")
    test_raw = load_data(in_dir / "aug_test.csv")
    summarize(train_raw, "train (brut)")

    train, test = prepare_datasets(train_raw, test_raw)

    summarize(train, "train (nettoyé)")
    assert train.drop(columns="target").isna().sum().sum() == 0, "NaN restants dans le train"
    assert test.isna().sum().sum() == 0, "NaN restants dans le test"

    if plots:
        generate_all_plots(basic_cleaning(train_raw), train, test, out_dir)

    train.to_csv(out_dir / "train_clean.csv", index=False)
    test.to_csv(out_dir / "test_clean.csv", index=False)
    print(f"\n[save] fichiers écrits dans {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Préparation des données HR Analytics")
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--no-plots", action="store_true", help="ne pas générer les figures")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, plots=not args.no_plots)
