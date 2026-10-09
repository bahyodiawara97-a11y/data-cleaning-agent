"""Chargement des fichiers et outils d'inspection appelés par l'agent (tool calling).

Les outils ne modifient JAMAIS les données : ils renvoient seulement du JSON.
"""
import io
import json

import pandas as pd
from langchain_core.tools import tool


# ---------------------------------------------------------------- chargement
def load_file(name: str, content: bytes) -> pd.DataFrame:
    """Lit un CSV (encodage + séparateur détectés) ou un fichier Excel."""
    if name.lower().endswith((".xlsx", ".xls")):
        return read_excel_table(content)
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(content), sep=None, engine="python",
                               encoding=encoding, dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("Impossible de lire le fichier.")


def read_excel_table(content: bytes) -> pd.DataFrame:
    """Lit la 1re feuille Excel en texte et retrouve la vraie ligne d'en-tête.

    Les factures ou exports Excel ont souvent un titre, un logo ou une adresse au-dessus
    du tableau : on prend comme en-tête la première ligne bien remplie.
    """
    raw = pd.read_excel(io.BytesIO(content), header=None, dtype=str)
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.empty:
        raise ValueError("La feuille Excel est vide.")
    filled = raw.notna().sum(axis=1)
    header_pos = int((filled >= 0.6 * filled.max()).to_numpy().argmax())
    names, seen = [], {}
    for i, v in enumerate(raw.iloc[header_pos].tolist()):
        n = str(v).strip() if pd.notna(v) and str(v).strip() else f"colonne_{i + 1}"
        seen[n] = seen.get(n, 0) + 1
        names.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    df = raw.iloc[header_pos + 1:].copy()
    df.columns = names
    return df.dropna(how="all").reset_index(drop=True)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Transforme les cellules vides ou remplies d'espaces en valeurs manquantes."""
    df = df.copy()
    for col in df.columns:
        if is_text(df[col]):
            df[col] = df[col].replace(r"^\s*$", pd.NA, regex=True)
    return df


# ---------------------------------------------------------------- analyses
def is_number(s: pd.Series) -> bool:
    """Colonne numérique, hors booléens (True/False)."""
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def is_text(s: pd.Series) -> bool:
    """Vrai pour une colonne de texte (object ou str selon la version de pandas)."""
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def _numeric_share(s: pd.Series) -> float:
    s = s.dropna().astype(str)
    if s.empty:
        return 0.0
    cleaned = s.str.replace(",", ".", regex=False).str.replace(r"[€$\s]", "", regex=True)
    return float(pd.to_numeric(cleaned, errors="coerce").notna().mean())


def _date_share(s: pd.Series) -> float:
    s = s.dropna().astype(str)
    if s.empty or not s.str.contains(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}").mean() > 0.8:
        return 0.0
    parsed = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=True)
    return float(parsed.notna().mean())


def overview(df: pd.DataFrame) -> dict:
    return {
        "lignes": int(len(df)),
        "colonnes": int(df.shape[1]),
        "doublons": int(df.duplicated().sum()),
        "valeurs_manquantes": {c: int(n) for c, n in df.isna().sum().items() if n},
        "types": {c: str(t) for c, t in df.dtypes.items()},
    }


def column_report(df: pd.DataFrame, column: str) -> dict:
    s = df[column]
    info = {
        "colonne": column,
        "type": str(s.dtype),
        "manquantes": int(s.isna().sum()),
        "valeurs_uniques": int(s.nunique()),
        "top_valeurs": {str(k): int(v) for k, v in s.value_counts().head(8).items()},
    }
    if is_text(s):
        txt = s.dropna().astype(str)
        info["espaces_parasites"] = int((txt != txt.str.strip()).sum())
        norm = txt.str.strip().str.lower()
        variants = txt.str.strip().groupby(norm).nunique()
        info["variantes_de_casse"] = {k: int(v) for k, v in variants[variants > 1].items()}
        info["part_numerique"] = round(_numeric_share(s), 2)
        info["part_dates"] = round(_date_share(s), 2)
    elif is_number(s):
        info["stats"] = {k: round(float(v), 2) for k, v in s.astype(float).describe().items()}
    elif pd.api.types.is_datetime64_any_dtype(s):
        info["min"], info["max"] = str(s.min()), str(s.max())
    return info


def outliers(df: pd.DataFrame, column: str, k: float = 3.0) -> dict:
    """Valeurs extrêmes (méthode IQR, k = 3) : vraisemblablement des erreurs de saisie."""
    s = df[column]
    if is_text(s):
        s = pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    elif not is_number(s):
        return {"colonne": column, "erreur": "colonne non numérique"}
    s = s.dropna().astype(float)
    if s.empty:
        return {"colonne": column, "erreur": "colonne non numérique"}
    q1, q3 = s.quantile([0.25, 0.75])
    low, high = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    out = s[(s < low) | (s > high)]
    return {"colonne": column, "bornes": [round(low, 2), round(high, 2)],
            "nb_valeurs_aberrantes": int(len(out)), "exemples": [round(float(v), 2) for v in pd.unique(out)[:5]]}


# ---------------------------------------------------------------- outils LLM
def build_tools(df: pd.DataFrame):
    """Crée les outils liés au DataFrame chargé."""

    def dump(obj) -> str:
        return json.dumps(obj, ensure_ascii=False, default=str)

    @tool
    def get_overview() -> str:
        """Vue d'ensemble du dataset : taille, types, doublons, valeurs manquantes."""
        return dump(overview(df))

    @tool
    def inspect_column(column: str) -> str:
        """Inspecte une colonne : valeurs fréquentes, espaces parasites, variantes de casse,
        part de valeurs numériques ou de dates stockées en texte."""
        if column not in df.columns:
            return dump({"erreur": f"colonne inconnue. Colonnes : {list(df.columns)}"})
        return dump(column_report(df, column))

    @tool
    def find_duplicates() -> str:
        """Compte les lignes en double et en montre quelques exemples."""
        dup = df[df.duplicated(keep=False)]
        return dump({"doublons": int(df.duplicated().sum()),
                     "exemples": dup.head(4).to_dict("records")})

    @tool
    def detect_outliers(column: str) -> str:
        """Détecte les valeurs extrêmes d'une colonne numérique (méthode IQR, 3 × IQR)."""
        if column not in df.columns:
            return dump({"erreur": "colonne inconnue"})
        return dump(outliers(df, column))

    return [get_overview, inspect_column, find_duplicates, detect_outliers]
