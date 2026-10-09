"""Interface Streamlit de l'AI Data Cleaning Agent.

Lancement : streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from agent.engine import describe
from agent.graph import build_graph, get_llm, new_config
from agent.tools import load_file, overview, prepare

load_dotenv()
st.set_page_config(page_title="AI Data Cleaning Agent", page_icon="🧹", layout="wide")

LABELS = {
    "profile": "🔍 Profilage du dataset",
    "agent": "🧠 Raisonnement de l'agent",
    "tools": "🛠️ Exécution des outils d'inspection",
    "plan": "📋 Génération du plan de nettoyage",
    "human_review": "⏸️ En attente de votre validation",
    "execute": "⚙️ Application des opérations validées",
}


def run_with_status(graph_input, label_start, label_end):
    """Exécute le graphe et affiche chaque étape en direct."""
    graph, config = st.session_state.graph, st.session_state.config
    with st.status(label_start, expanded=True) as status:
        for update in graph.stream(graph_input, config, stream_mode="updates"):
            for node, output in update.items():
                if node == "__interrupt__":
                    continue
                st.write(LABELS.get(node, node))
                for msg in (output or {}).get("messages", []):
                    for call in getattr(msg, "tool_calls", None) or []:
                        args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                        st.caption(f"↳ appel de `{call['name']}({args})`")
        status.update(label=label_end, state="complete", expanded=False)


def show_table(df: pd.DataFrame):
    """Affiche un tableau ; les dates sans heure sont affichées au format jour seulement."""
    config = {}
    for col in df.select_dtypes(include="datetime").columns:
        if (df[col].dropna().dt.normalize() == df[col].dropna()).all():
            config[col] = st.column_config.DateColumn(col, format="YYYY-MM-DD")
    st.dataframe(df.head(10), width="stretch", column_config=config)


def metrics(df: pd.DataFrame, title: str):
    o = overview(df)
    st.markdown(f"**{title}**")
    a, b, c = st.columns(3)
    a.metric("Lignes", o["lignes"])
    b.metric("Doublons", o["doublons"])
    c.metric("Valeurs manquantes", sum(o["valeurs_manquantes"].values()))


# ------------------------------------------------------------------ en-tête
st.title("🧹 AI Data Cleaning Agent")
APP_VERSION = "1.6"
st.caption(f"L'agent inspecte, propose un plan, et n'applique rien sans votre validation. · version {APP_VERSION}")

llm = get_llm()
if llm is None:
    st.info("Mode démo : aucune clé Groq détectée, le plan est généré par des règles. "
            "Ajoutez `GROQ_API_KEY` dans le fichier `.env` pour activer le LLM.")

# ------------------------------------------------------------------ 1. import
uploaded = st.file_uploader("Importez votre dataset (CSV, XLSX ou XLS)", type=["csv", "xlsx", "xls"])
use_demo = st.button("Utiliser le fichier de démo")

source = None
if uploaded:
    source = (uploaded.name, uploaded.getvalue())
elif use_demo or st.session_state.get("source", ("",))[0] == "dirty_cafe_sales.csv":
    path = Path(__file__).parent / "data" / "dirty_cafe_sales.csv"
    source = (path.name, path.read_bytes())

if source and st.session_state.get("source") != source:
    df = prepare(load_file(*source))
    graph, results = build_graph(df, llm)
    st.session_state.update(source=source, df=df, graph=graph, results=results, config=new_config())

if "df" not in st.session_state:
    st.stop()

df = st.session_state.df
st.subheader(f"Aperçu : {st.session_state.source[0]}")
show_table(df)
metrics(df, "Avant nettoyage")

graph, config, results = st.session_state.graph, st.session_state.config, st.session_state.results
state = graph.get_state(config)

# ------------------------------------------------------------------ 2. analyse
if not state.values and st.button("🚀 Analyser avec l'agent", type="primary"):
    run_with_status({"messages": []}, "L'agent analyse vos données...", "✅ Plan prêt à valider")
    st.rerun()

# ------------------------------------------------------------------ 3. validation
if state.next and state.next[0] == "human_review":
    plan_box = st.empty()
    with plan_box.container():
        st.subheader("📋 Plan de nettoyage proposé")
        if results.get("warning"):
            st.warning(results["warning"])
        if results.get("rejected"):
            with st.expander(f"🛡️ {len(results['rejected'])} proposition(s) de l'IA écartée(s) par le garde-fou"):
                for a in results["rejected"]:
                    label = "Proposition illisible" if a.get("_raw") else describe(a)
                    st.markdown(f"- ~~{label}~~ : {a['rejected_reason']}")
        st.caption("Décochez les actions que vous refusez.")
        plan = state.values["plan"]
        if all(a["operation"] == "convert_type" for a in plan):
            st.success("✨ Aucun problème de qualité détecté : ce fichier est déjà propre. "
                       "Seules des conversions de type (texte → nombre ou date) sont proposées.")
        groups = [("ia", "🧠 Proposées par l'IA"),
                  ("regles", "🧩 Suggestions complémentaires (règles), oubliées par l'IA"
                   if any(a.get("source") == "ia" for a in plan) else "🧩 Proposées par les règles")]
        approved = []
        for source, title in groups:
            items = [(i, a) for i, a in enumerate(plan) if a.get("source", "regles") == source]
            if not items:
                continue
            st.markdown(f"**{title}** ({len(items)})")
            approved += [a for i, a in items
                         if st.checkbox(f"{describe(a)} — _{a.get('reason', '')}_", value=a.get("default", True), key=f"a{i}")]
        clicked = st.button(f"✅ Appliquer {len(approved)} action(s)", type="primary")
    if clicked:
        plan_box.empty()  # le plan disparaît pendant l'exécution
        run_with_status(Command(resume=approved), "Nettoyage en cours...", "✅ Données nettoyées")
        st.rerun()

# ------------------------------------------------------------------ 4. résultat
if "cleaned" in results:
    cleaned = results["cleaned"]
    st.subheader("📊 Résultat")
    metrics(cleaned, "Après nettoyage")
    left = {c: int(n) for c, n in cleaned.isna().sum().items() if n}
    if left:
        st.caption("Valeurs laissées vides volontairement (aucune valeur inventée) : "
                   + ", ".join(f"{c} ({n})" for c, n in left.items()))
    st.dataframe(pd.DataFrame(results["log"]), width="stretch", hide_index=True)
    show_table(cleaned)
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Télécharger le CSV nettoyé", cleaned.to_csv(index=False).encode("utf-8"),
                       "donnees_nettoyees.csv", "text/csv")
    c2.download_button("⬇️ Télécharger le script Pandas", results["script"], "nettoyage.py", "text/x-python")
    with st.expander("Voir le script Pandas reproductible"):
        st.code(results["script"], language="python")
    if st.button("🔄 Recommencer"):
        st.session_state.clear()
        st.rerun()
