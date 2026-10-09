"""Moteur d'exécution en liste blanche.

Le LLM ne génère aucun code : il choisit parmi les opérations ci-dessous.
Chaque opération appliquée produit aussi sa ligne de code Pandas, pour exporter
un script de nettoyage reproductible.
"""
from typing import List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from agent.tools import _date_share, _numeric_share, is_number, is_text, outliers

Operation = Literal["drop_total_rows", "drop_duplicates", "strip_whitespace", "normalize_case", "convert_type",
                    "fill_missing", "nullify_outliers", "clip_outliers", "drop_column"]


class Action(BaseModel):
    operation: Operation = Field(description="Opération autorisée")
    column: Optional[str] = Field(None, description="Colonne visée (vide pour drop_duplicates)")
    case: Optional[Literal["most_frequent", "lower", "upper", "title"]] = Field(
        None, description="Pour normalize_case : most_frequent garde l'orthographe la plus courante")
    target_type: Optional[Literal["float", "int", "datetime", "string"]] = Field(None, description="Pour convert_type")
    strategy: Optional[Literal["median", "mean", "mode", "constant", "drop_rows"]] = Field(None, description="Pour fill_missing")
    value: Optional[str] = Field(None, description="Valeur si strategy = constant")
    reason: str = Field(description="Justification courte, en français")


class CleaningPlan(BaseModel):
    actions: List[Action]


# Ordre d'exécution sûr : on nettoie le texte, on convertit, puis on complète.
ORDER = ["drop_total_rows", "drop_column", "drop_duplicates", "strip_whitespace", "normalize_case",
         "convert_type", "nullify_outliers", "clip_outliers", "fill_missing"]


def describe(a: dict) -> str:
    col = a.get("column")
    labels = {
        "drop_duplicates": (f"Supprimer les doublons de « {col} » (garder la 1re ligne)" if col
                            else "Supprimer les lignes en double"),
        "drop_total_rows": "Supprimer les lignes de total / sous-total",
        "strip_whitespace": f"Retirer les espaces parasites de « {col} »",
        "normalize_case": f"Harmoniser la casse de « {col} » ({a.get('case')})",
        "convert_type": f"Convertir « {col} » en {a.get('target_type')}",
        "fill_missing": f"Compléter les valeurs manquantes de « {col} » ({a.get('strategy')})",
        "nullify_outliers": f"Marquer comme manquantes les valeurs extrêmes de « {col} »",
        "clip_outliers": f"Plafonner les valeurs extrêmes de « {col} »",
        "drop_column": f"Supprimer la colonne « {col} »",
    }
    return labels[a["operation"]]


def _to_number(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.replace(",", ".", regex=False).str.replace(r"[€$\s]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def parse_dates(s: pd.Series) -> pd.Series:
    """Dates aux formats mélangés : 2026/04/12 = année-mois-jour, 12/04/2026 = jour/mois/année."""
    s = s.astype("string").str.strip()
    year_first = s.str.match(r"^\d{4}[-/]").fillna(False).astype(bool)
    iso = pd.to_datetime(s.where(year_first).str.replace("/", "-", regex=False),
                         errors="coerce", format="ISO8601")
    other = pd.to_datetime(s.where(~year_first), errors="coerce", dayfirst=True, format="mixed")
    return iso.fillna(other)


def best_spelling(variants: pd.Series) -> str:
    """Parmi « Marseille », « marseille », « MARSEILLE » : la variante bien écrite la plus fréquente."""
    counts = variants.value_counts()
    well = [v for v in counts.index if v != v.lower() and v != v.upper()]
    return well[0] if well else counts.index[0]  # rien de « bien écrit » : on garde la plus fréquente telle quelle


BEST_SPELLING_CODE = (
    "def best_spelling(variants):\n"
    "    counts = variants.value_counts()\n"
    "    well = [v for v in counts.index if v != v.lower() and v != v.upper()]\n"
    "    return well[0] if well else counts.index[0]\n")


def apply_action(df: pd.DataFrame, a: dict):
    """Applique UNE opération. Renvoie (df, ligne_de_code)."""
    op, col = a["operation"], a.get("column")
    if (op not in ("drop_duplicates", "drop_total_rows") or col) and col not in df.columns:
        raise ValueError(f"colonne « {col} » introuvable")
    c = repr(col)

    if op == "drop_total_rows":
        mask = total_rows_mask(df)
        return df[~mask].reset_index(drop=True), (
            "is_total = df.astype('string').apply(lambda c: c.str.strip().str.match(r'(?i)^(sous[- ]?)?total')).any(axis=1)\n"
            "df = df[~(is_total & (df.isna().mean(axis=1) >= 0.4))].reset_index(drop=True)")
    if op == "drop_duplicates" and col:
        return (df.drop_duplicates(subset=[col]).reset_index(drop=True),
                f"df = df.drop_duplicates(subset=[{c}]).reset_index(drop=True)")
    if op == "drop_duplicates":
        return df.drop_duplicates().reset_index(drop=True), "df = df.drop_duplicates().reset_index(drop=True)"
    if op == "drop_column":
        return df.drop(columns=[col]), f"df = df.drop(columns=[{c}])"
    if op == "strip_whitespace":
        df[col] = df[col].astype("string").str.strip()
        return df, f"df[{c}] = df[{c}].astype('string').str.strip()"
    if op == "normalize_case" and (a.get("case") or "most_frequent") == "most_frequent":
        txt = df[col].astype("string").str.strip()
        best = txt.groupby(txt.str.lower()).agg(best_spelling)
        df[col] = txt.str.lower().map(best).astype("string")
        return df, (BEST_SPELLING_CODE +
                    f"txt = df[{c}].astype('string').str.strip()\n"
                    f"df[{c}] = txt.str.lower().map(txt.groupby(txt.str.lower()).agg(best_spelling))")
    if op == "normalize_case":
        case = a.get("case") or "title"
        df[col] = getattr(df[col].astype("string").str.strip().str, case)()
        return df, f"df[{c}] = df[{c}].astype('string').str.strip().str.{case}()"
    if op == "convert_type":
        t = a.get("target_type") or "float"
        if t == "datetime":
            df[col] = parse_dates(df[col])
            return df, (f"s = df[{c}].astype('string').str.strip()\n"
                        f"year_first = s.str.match(r'^\\d{{4}}[-/]').fillna(False).astype(bool)\n"
                        f"df[{c}] = pd.to_datetime(s.where(year_first).str.replace('/', '-'), errors='coerce', format='ISO8601')"
                        f".fillna(pd.to_datetime(s.where(~year_first), errors='coerce', dayfirst=True, format='mixed'))")
        if t in ("float", "int"):
            df[col] = _to_number(df[col])
            code = (f"df[{c}] = pd.to_numeric(df[{c}].astype('string').str.replace(',', '.')"
                    f".str.replace(r'[€$\\s]', '', regex=True), errors='coerce')")
            if t == "int":
                df[col] = df[col].round().astype("Int64")
                code += f"\ndf[{c}] = df[{c}].round().astype('Int64')"
            return df, code
        df[col] = df[col].astype("string")
        return df, f"df[{c}] = df[{c}].astype('string')"
    if op == "nullify_outliers":
        low, high = outliers(df, col)["bornes"]
        df[col] = df[col].mask((df[col] < low) | (df[col] > high))
        return df, f"df[{c}] = df[{c}].mask((df[{c}] < {low}) | (df[{c}] > {high}))"
    if op == "clip_outliers":
        report = outliers(df, col)
        low, high = report["bornes"]
        low = max(low, float(df[col].min()))  # pas de borne basse inférieure au minimum réel
        if pd.api.types.is_integer_dtype(df[col]):  # bornes entières pour une colonne d'entiers
            low, high = int(-(-low // 1)), int(high // 1)
        df[col] = df[col].clip(lower=low, upper=high)
        return df, f"df[{c}] = df[{c}].clip(lower={low}, upper={high})"
    if op == "fill_missing":
        strat = a.get("strategy") or "mode"
        if strat == "drop_rows":
            return df.dropna(subset=[col]).reset_index(drop=True), f"df = df.dropna(subset=[{c}]).reset_index(drop=True)"
        if strat in ("median", "mean"):
            if not is_number(df[col]):
                raise ValueError(f"« {col} » n'est pas numérique")
            v = round(float(getattr(df[col], strat)()), 2)
            if str(df[col].dtype) == "Int64":
                v = int(round(v))
        elif strat == "constant":
            v = a.get("value") or "Inconnu"
        else:
            v = df[col].mode().iloc[0]
        df[col] = df[col].fillna(v)
        return df, f"df[{c}] = df[{c}].fillna({v!r})"
    raise ValueError(f"opération non autorisée : {op}")


def execute_plan(df: pd.DataFrame, actions: list):
    """Applique les actions validées dans un ordre sûr. Renvoie (df, journal, script)."""
    df = df.copy()
    log, code = [], ["import pandas as pd", "", "df = pd.read_csv('mon_fichier.csv')"]
    for a in sorted(actions, key=lambda x: ORDER.index(x["operation"])):
        try:
            df, line = apply_action(df, a)
            log.append({"action": describe(a), "statut": "✅ appliquée"})
            code.append(line)
        except Exception as e:  # une action invalide n'arrête pas les autres
            log.append({"action": describe(a), "statut": f"⚠️ ignorée : {e}"})
    code.append("df.to_csv('mon_fichier_nettoye.csv', index=False)")
    return df, log, "\n".join(code)


# ---------------------------------------------------------------- mode démo
def total_rows_mask(df: pd.DataFrame) -> pd.Series:
    """Lignes « TOTAL » / « Sous-total » ajoutées sous un tableau (souvent dans les exports Excel)."""
    is_total = df.astype("string").apply(
        lambda c: c.str.strip().str.match(r"(?i)^(sous[- ]?)?total")).fillna(False).any(axis=1)
    return is_total & (df.isna().mean(axis=1) >= 0.4)


def is_identifier(col: str, s: pd.Series = None) -> bool:
    """order_id, N° facture, référence… : on n'y touche pas."""
    c = col.lower().strip()
    if c == "id" or c.endswith("_id") or c.startswith("id_") or c.endswith(" id"):
        return True
    if any(k in c for k in ("n°", "numéro", "numero", "num ", "réf", "ref", "code", "siret", "iban")):
        return True
    if s is not None and is_text(s):  # texte presque toujours unique, ex. « F-1001 »
        txt = s.dropna().astype(str)
        if _date_share(s) > 0.5 or _numeric_share(s) > 0.5 or txt.str.contains("@").mean() > 0.5:
            return False
        return len(txt) >= 10 and txt.nunique() / len(txt) > 0.95 and txt.str.contains(" ").mean() < 0.5
    return False


def is_contact(s: pd.Series) -> bool:
    """Emails / téléphones : on ne remplit jamais avec une valeur inventée."""
    txt = s.dropna().astype(str)
    return bool(len(txt)) and txt.str.contains("@").mean() > 0.5


def _column_rules(df: pd.DataFrame, col: str) -> list:
    """Règles pour une colonne."""
    actions = []
    s = df[col]
    if s.isna().all():
        actions.append(dict(operation="drop_column", column=col, reason="colonne entièrement vide"))
        return actions
    numeric = is_number(s)
    if is_identifier(col, s):
        return actions
    if is_text(s):
        txt = s.dropna().astype(str)
        if (txt != txt.str.strip()).any():
            actions.append(dict(operation="strip_whitespace", column=col, reason="espaces en début ou fin de valeur"))
        if _date_share(s) > 0.8:
            patterns = s.dropna().astype(str).str.strip().str.replace(r"\d", "9", regex=True).str[:10].nunique()
            actions.append(dict(operation="convert_type", column=col, target_type="datetime",
                                reason="dates stockées en texte" + (f", {patterns} formats différents" if patterns > 1 else "")))
            return actions
        if _numeric_share(s) > 0.9 and not is_identifier(col):
            nums = pd.to_numeric(s.dropna().astype(str).str.replace(",", ".", regex=False), errors="coerce").dropna()
            target = "int" if len(nums) and (nums % 1 == 0).all() else "float"
            actions.append(dict(operation="convert_type", column=col, target_type=target, reason="colonne numérique lue comme du texte"))
            numeric = True
        elif txt.str.strip().groupby(txt.str.strip().str.lower()).nunique().gt(1).any():
            actions.append(dict(operation="normalize_case", column=col, case="most_frequent",
                                reason="même valeur écrite avec des casses différentes"))
        elif not txt.str.contains("@").any():
            # « bob durand » au milieu de « Alice Martin » : on harmonise.
            # Une colonne entièrement en MAJUSCULES (« SARL EXEMPLE », « EUR ») est un choix de style : on n'y touche pas.
            stripped = txt.str.strip()
            words = stripped.str.contains(r"[A-Za-zÀ-ÿ]{3,}") & ~stripped.str.contains(r"[._/\\]")
            badly = words & ((stripped == stripped.str.lower()) | (stripped == stripped.str.upper()))
            well = words & ~badly
            if badly.any() and well.mean() >= 0.3:
                actions.append(dict(operation="normalize_case", column=col, case="title",
                                    reason=f"{int(badly.sum())} valeurs tout en minuscules ou en majuscules"))
    n_out = 0
    if numeric and not is_identifier(col):
        out = outliers(df, col)
        n_out = out.get("nb_valeurs_aberrantes", 0)
        typos = typo_outliers(df, col)
        if typos:  # seulement les fautes de saisie évidentes (999, -1, valeurs absurdes)
            ex = ", ".join(f"{v:g}" for v in typos[:3])
            actions.append(dict(operation="nullify_outliers", column=col,
                                reason=f"{n_out} valeurs extrêmes (ex. {ex}), probables fautes de saisie"))
        else:
            n_out = 0  # valeurs rares mais plausibles : on n'y touche pas
    n_missing = int(s.isna().sum())
    if (n_missing or n_out) and not is_contact(s):
        total = n_missing + n_out
        if numeric:
            actions.append(dict(operation="fill_missing", column=col, strategy="median",
                                reason=f"{total} valeurs à compléter par la médiane"))
        elif is_text(s):
            actions.append(dict(operation="fill_missing", column=col, strategy="constant", value="Inconnu",
                                reason=f"{n_missing} valeurs manquantes"))
    return actions


SENTINELS = {99, 999, 9999, 99999, 999999, -1, -9, -99, -999, -9999}


def typo_outliers(df: pd.DataFrame, col: str) -> list:
    """Valeurs extrêmes qui ressemblent à des fautes de saisie : 999, -1, 9999…, ou très loin des autres (10 × IQR)."""
    s = df[col]
    if is_text(s):
        s = pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    s = s.dropna().astype(float)
    if s.empty:
        return []
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    far = (s < q1 - 10 * iqr) | (s > q3 + 10 * iqr) if iqr > 0 else pd.Series(False, index=s.index)
    extreme3 = (s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)
    sentinel = s.isin(SENTINELS) & extreme3
    return sorted(set(s[far | sentinel].tolist()))


def validate_plan(df: pd.DataFrame, actions: list):
    """Garde-fou : écarte les actions de l'IA impossibles ou sans effet sur ce fichier.

    Renvoie (actions gardées, actions écartées avec la raison du rejet).
    """
    kept, rejected = [], []
    planned = {(a["operation"], a.get("column")) for a in actions}
    for a in actions:
        op, col = a["operation"], a.get("column")
        why = None
        try:
            if col and col not in df.columns:
                why = f"la colonne « {col} » n'existe pas"
            elif op == "drop_duplicates":
                n = df.duplicated(subset=[col] if col else None).sum()
                why = None if n else "aucun doublon"
            elif op == "drop_total_rows":
                why = None if total_rows_mask(df).any() else "aucune ligne de total"
            elif op == "drop_column":
                if is_identifier(col, df[col]):
                    why = "c'est un identifiant"
                elif df[col].isna().mean() < 0.95:
                    why = f"la colonne contient des données ({df[col].notna().sum()} valeurs) : on ne supprime pas de données"
            else:
                s = df[col]
                txt = s.dropna().astype(str).str.strip() if is_text(s) else None
                if op == "strip_whitespace":
                    why = None if txt is not None and (s.dropna().astype(str) != txt).any() else "aucun espace parasite"
                elif op == "normalize_case":
                    if txt is None:
                        why = "colonne non textuelle"
                    elif is_identifier(col, s):
                        why = "c'est un identifiant"
                    else:
                        variants = txt.groupby(txt.str.lower()).nunique().gt(1).any()
                        words = txt.str.contains(r"[A-Za-zÀ-ÿ]{3,}") & ~txt.str.contains(r"[._/\\]")
                        badly = words & ((txt == txt.str.lower()) | (txt == txt.str.upper()))
                        mixed = badly.any() and (words & ~badly).mean() >= 0.3
                        why = None if variants or mixed else "aucune incohérence de casse"
                        if not why and mixed and not variants and a.get("case") != "title":
                            # « bob durand » n'a pas de variante bien écrite : most_frequent le laisserait tel quel
                            a = dict(a, case="title")
                elif op == "fill_missing":
                    has_out = (("nullify_outliers", col) in planned)
                    why = None if s.isna().any() or has_out else "aucune valeur manquante"
                    if not why and is_contact(s):
                        why = "on n'invente pas d'email ou de contact"
                elif op in ("nullify_outliers", "clip_outliers"):
                    if not outliers(df, col).get("nb_valeurs_aberrantes"):
                        why = "aucune valeur extrême (3 × IQR)"
                    elif not typo_outliers(df, col):
                        a = dict(a, default=False, reason=(a.get("reason", "") +
                                 " — à vérifier : valeurs rares mais plausibles, pas des fautes de saisie évidentes"))
                elif op == "convert_type" and is_identifier(col, s) and a.get("target_type") != "string":
                    why = "c'est un identifiant"
        except Exception as e:
            why = f"vérification impossible ({e})"
        (rejected if why else kept).append(dict(a, rejected_reason=why) if why else a)
    # 2e passe : « compléter » ne sert que s'il y a des manques, ou si une action « valeurs extrêmes » est gardée
    nullify = {a["column"]: a.get("default", True) for a in kept if a["operation"] == "nullify_outliers"}
    final = []
    for a in kept:
        col = a.get("column")
        if a["operation"] == "fill_missing" and not df[col].isna().any():
            if col not in nullify:
                rejected.append(dict(a, rejected_reason="aucune valeur manquante"))
                continue
            a = dict(a, default=nullify[col])
        final.append(a)
    return final, rejected


def rule_based_plan(df: pd.DataFrame) -> list:
    """Plan généré par des règles simples (utilisé sans clé Groq ou en secours)."""
    actions = []
    n_total = int(total_rows_mask(df).sum())
    if n_total:
        actions.append(dict(operation="drop_total_rows", reason=f"{n_total} ligne(s) de total sous le tableau"))
        df = df[~total_rows_mask(df)]
    if df.duplicated().any():
        actions.append(dict(operation="drop_duplicates", reason=f"{int(df.duplicated().sum())} lignes en double"))
    base = df.drop_duplicates()
    for col in base.columns:
        try:
            if not is_identifier(col, base[col]) and not any(k in col.lower() for k in ("facture", "invoice", "commande", "order")):
                continue
            dup = base[base[col].notna() & base[col].duplicated(keep=False)]
            if dup.empty:
                continue
            # même identifiant ET la plupart des autres champs identiques -> vrai doublon (pas une ligne d'article)
            same = dup.groupby(col).apply(lambda g: (g.astype(str).nunique() == 1).mean(), include_groups=False)
            n = int(base[col].duplicated().sum())
            if same.min() >= 0.75:
                actions.append(dict(operation="drop_duplicates", column=col,
                                    reason=f"{n} ligne(s) avec le même « {col} » et {same.min():.0%} des champs identiques"))
        except Exception:
            continue
    for col in df.columns:
        try:
            actions += _column_rules(df, col)
        except Exception:  # une colonne inattendue ne doit pas bloquer tout le plan
            continue
    return actions
