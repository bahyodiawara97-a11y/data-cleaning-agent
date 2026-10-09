# 🧹 AI Data Cleaning Agent

Un agent IA qui inspecte un fichier CSV ou Excel, détecte les problèmes de qualité et propose un plan de nettoyage. **Rien n'est appliqué sans votre validation** (human-in-the-loop).

## Lancer l'application (3 étapes)

Ouvrez le **Terminal** et tapez ces commandes une par une.

**1. Aller dans le dossier du projet**
```bash
cd ~/Downloads/data-cleaning-agent
```

**2. Installer les dépendances** (une seule fois)
```bash
conda activate anaconda-nlp
pip install -r requirements.txt
```

**3. Lancer l'application**
```bash
streamlit run app.py
```
Une page s'ouvre dans votre navigateur (http://localhost:8501). Cliquez sur **« Utiliser le fichier de démo »**, puis **« Analyser avec l'agent »**.

Pour arrêter : `Ctrl + C` dans le Terminal.

## Activer le vrai LLM (Groq, gratuit)

Sans clé, l'application tourne en **mode démo** : le plan est créé par des règles, sans IA.

1. Créez une clé sur https://console.groq.com/keys
2. Copiez le fichier `.env.example` et renommez la copie en `.env`
3. Collez votre clé après `GROQ_API_KEY=`
4. Relancez `streamlit run app.py`

## Comment ça marche

```
profile → agent ⇄ tools → plan → human_review (pause) → execute
```

| Fichier | Rôle |
|---|---|
| `app.py` | Interface Streamlit, affichage des étapes en direct |
| `agent/graph.py` | Graphe LangGraph : boucle de tool calling, pause `interrupt()` |
| `agent/tools.py` | Chargement des fichiers + outils d'inspection (lecture seule) |
| `agent/engine.py` | Moteur en **liste blanche** : le LLM ne génère aucun code |
| `data/dirty_cafe_sales.csv` | Fichier de test volontairement sale |

**Sécurité :** le LLM choisit uniquement parmi 7 opérations prédéfinies (`drop_duplicates`, `strip_whitespace`, `normalize_case`, `convert_type`, `fill_missing`, `clip_outliers`, `drop_column`). Une action invalide est ignorée sans bloquer les autres.

**Bonus :** l'application exporte un **script Pandas reproductible** du nettoyage appliqué.

## Stack

LangGraph · Groq · Python & Pandas · Streamlit
