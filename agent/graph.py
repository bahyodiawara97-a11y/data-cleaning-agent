"""Graphe LangGraph de l'agent.

profile → agent ⇄ tools → plan → human_review (pause) → execute
"""
import json
import os
import uuid
from typing import Annotated, Optional, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from agent.engine import Action, CleaningPlan, describe, execute_plan, rule_based_plan, validate_plan
from agent.tools import build_tools, overview

MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = """Tu es un agent expert en qualité des données.
1. Inspecte le dataset avec tes outils (get_overview, inspect_column, find_duplicates, detect_outliers).
   Inspecte au moins chaque colonne de type texte qui semble problématique.
2. Quand tu as assez d'informations, réponds simplement « Analyse terminée ».
Tu ne modifies jamais les données toi-même."""

PLAN_PROMPT = """Propose maintenant un plan de nettoyage, en n'utilisant QUE ces opérations :
drop_total_rows, drop_duplicates, strip_whitespace, normalize_case (case), convert_type (target_type),
fill_missing (strategy, value), nullify_outliers, clip_outliers, drop_column.
Règles :
- convertis les colonnes numériques ou dates stockées en texte AVANT de les compléter ;
- valeurs extrêmes (ex. une quantité de 999) : préfère nullify_outliers (la valeur devient manquante),
  puis fill_missing en median ; n'utilise clip_outliers que si la valeur est plausible mais trop haute ;
- ne traite pas les valeurs extrêmes d'un prix ou d'un montant sauf si elles sont absurdes ;
- n'utilise median/mean que sur des colonnes numériques ;
- ne touche jamais aux identifiants (order_id…) ;
- ne remplis jamais un email, un téléphone ou un identifiant avec une valeur inventée ;
- drop_duplicates avec "column" = un identifiant (ex. numero_facture) supprime les lignes qui ont le même
  identifiant (même facture extraite deux fois) ; sans "column", seules les lignes 100 % identiques sont supprimées ;
- ne change pas la casse d'une colonne entièrement en MAJUSCULES (raisons sociales, codes devise) ni des noms de fichiers ;
- supprime les colonnes entièrement vides et les lignes de TOTAL sous le tableau (drop_total_rows) ;
- normalize_case : préfère case="most_frequent" (garde l'orthographe la plus courante, respecte « SA », « SARL ») ;
  n'utilise "title" que si toutes les valeurs sont mal écrites (ex. tout en minuscules).
Pour chaque action, donne une raison TRÈS courte en français (12 mots maximum) qui cite
le chiffre observé lors de l'inspection (ex. « 12 lignes en double », « 7 prix manquants »).
Propose 15 actions au maximum, une seule par couple (opération, colonne)."""


class State(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list
    approved: list
    mode: str


JSON_FORMAT = (
    'Réponds UNIQUEMENT avec un objet JSON de la forme {"actions": [{"operation": "...", "column": "...", '
    '"case": null, "target_type": null, "strategy": null, "value": null, "reason": "..."}]}. '
    "Mets null pour les champs inutiles. Aucun texte avant ou après le JSON.")


def parse_actions(text: str):
    """Lit le JSON du LLM et valide chaque action séparément. Renvoie (valides, rejetées)."""
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1] if start != -1 else text)
    items = data.get("actions", []) if isinstance(data, dict) else data
    good, bad = [], []
    for item in items if isinstance(items, list) else []:
        try:
            good.append(Action.model_validate(item).model_dump())
        except Exception:
            label = item.get("operation", "?") if isinstance(item, dict) else str(item)[:40]
            bad.append({"operation": "drop_duplicates", "column": None, "reason": "",
                        "rejected_reason": f"action mal formée ignorée ({label})", "_raw": True})
    return good, bad


def merge_plans(llm_actions: list, rule_actions: list) -> list:
    """Plan de l'IA + actions des règles qu'elle a oubliées (« suggestions complémentaires »)."""
    plan = [dict(a, source="ia") for a in llm_actions]
    family = lambda op: "outliers" if op in ("nullify_outliers", "clip_outliers") else op
    covered = {(family(a["operation"]), a.get("column")) for a in llm_actions}
    dropped = {a.get("column") for a in llm_actions if a["operation"] == "drop_column"}
    global_ops = {(a["operation"], a.get("column")) for a in llm_actions
                  if a["operation"] in ("drop_duplicates", "drop_total_rows")}
    for a in rule_actions:
        if a["operation"] in ("drop_duplicates", "drop_total_rows"):
            if (a["operation"], a.get("column")) in global_ops:
                continue
        elif (family(a["operation"]), a.get("column")) in covered or a.get("column") in dropped:
            continue
        plan.append(dict(a, source="regles"))
    return plan


def get_llm():
    """Renvoie le LLM Groq, ou None si aucune clé n'est configurée (mode démo)."""
    if not os.getenv("GROQ_API_KEY"):
        return None
    from langchain_groq import ChatGroq
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
    # max_tokens élevé : sinon la réponse JSON du plan est coupée en plein milieu
    return ChatGroq(model=model, temperature=0, max_tokens=8192, **extra)


def build_graph(df: pd.DataFrame, llm=None):
    """Construit le graphe pour un DataFrame. Renvoie (graph, results)."""
    tools = build_tools(df)
    results: dict = {}  # le DataFrame nettoyé et le rapport sont stockés ici

    def profile(state: State):
        summary = json.dumps(overview(df), ensure_ascii=False)
        return {"messages": [SystemMessage(SYSTEM_PROMPT),
                             HumanMessage(f"Voici le dataset à analyser : {summary}")],
                "mode": "llm" if llm else "demo"}

    def agent(state: State):
        if llm:
            return {"messages": [llm.bind_tools(tools).invoke(state["messages"])]}
        # Mode démo : l'agent appelle les outils selon un scénario fixe.
        if not any(isinstance(m, ToolMessage) for m in state["messages"]):
            calls = [{"name": "find_duplicates", "args": {}, "id": "call_dup"}]
            calls += [{"name": "inspect_column", "args": {"column": c}, "id": f"call_{i}"}
                      for i, c in enumerate(df.columns)]
            return {"messages": [AIMessage(content="", tool_calls=calls)]}
        return {"messages": [AIMessage(content="Analyse terminée")]}

    def route(state: State):
        last = state["messages"][-1]
        rounds = sum(1 for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls)
        if getattr(last, "tool_calls", None) and rounds <= MAX_TOOL_ROUNDS:
            return "tools"
        return "plan"

    def plan(state: State):
        if llm:
            # On résume les résultats des outils en texte : si on renvoyait l'historique des appels,
            # le modèle essaierait de rappeler inspect_column, qui n'est plus disponible à cette étape.
            findings = "\n".join(f"- {m.name} : {m.content}" for m in state["messages"]
                                 if isinstance(m, ToolMessage))
            prompt = [SystemMessage("Tu es un expert en qualité des données. Tu n'as plus accès aux outils."),
                      HumanMessage(f"Colonnes : {list(df.columns)}\n\nRésultats de l'inspection :\n"
                                   f"{findings[:12000]}\n\n{PLAN_PROMPT}")]
            errors = []
            # 1) JSON libre, analysé action par action : une action mal formée est écartée sans perdre les autres
            try:
                raw = llm.bind(response_format={"type": "json_object"}).invoke(prompt + [HumanMessage(JSON_FORMAT)])
                actions, bad = parse_actions(raw.content)
                if actions:
                    actions, rejected = validate_plan(df, actions)  # garde-fou avant affichage
                    results["rejected"] = bad + rejected
                    return {"plan": merge_plans(actions, rule_based_plan(df))}
                errors.append("JSON sans action valide")
            except Exception as e:
                errors.append(f"JSON : {str(e)[:300]}")
            # 2) secours : sortie structurée par appel d'outil
            try:
                out = llm.with_structured_output(CleaningPlan, method="function_calling").invoke(prompt)
                actions = [x.model_dump() for x in out.actions]
                if actions:
                    actions, rejected = validate_plan(df, actions)
                    results["rejected"] = rejected
                    return {"plan": merge_plans(actions, rule_based_plan(df))}
            except Exception as e:
                errors.append(f"outil : {str(e)[:300]}")
            results["warning"] = f"Le LLM n'a pas pu produire de plan ({' | '.join(errors) if errors else 'plan vide'}). Plan de secours utilisé."
        return {"plan": [dict(a, source="regles") for a in rule_based_plan(df)]}

    def human_review(state: State):
        # ⏸ Le graphe s'arrête ici jusqu'à la validation dans l'interface.
        approved = interrupt({"plan": state["plan"]})
        return {"approved": approved}

    def execute(state: State):
        cleaned, log, script = execute_plan(df, state["approved"])
        results.update(cleaned=cleaned, log=log, script=script)
        return {}

    g = StateGraph(State)
    g.add_node("profile", profile)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    g.add_node("plan", plan)
    g.add_node("human_review", human_review)
    g.add_node("execute", execute)
    g.add_edge(START, "profile")
    g.add_edge("profile", "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "plan": "plan"})
    g.add_edge("tools", "agent")
    g.add_edge("plan", "human_review")
    g.add_edge("human_review", "execute")
    g.add_edge("execute", END)
    return g.compile(checkpointer=MemorySaver()), results


def new_config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 50}


__all__ = ["build_graph", "get_llm", "new_config", "describe"]
