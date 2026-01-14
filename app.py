import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import os

# Gemini (Google Generative AI)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False
    genai = None

# =========================
# Sécurité - mot de passe
# =========================
def check_password():
    def password_entered():
        if st.session_state.get("password") == "Digitad2025!":
            st.session_state["password_correct"] = True
            if "password" in st.session_state:
                del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Accès réservé à l’équipe Digitad",
            type="password",
            on_change=password_entered,
            key="password"
        )
        return False

    if not st.session_state["password_correct"]:
        st.text_input(
            "Accès réservé à l’équipe Digitad",
            type="password",
            on_change=password_entered,
            key="password"
        )
        st.error("Mot de passe incorrect")
        return False

    return True


if not check_password():
    st.stop()

# =========================
# Configuration API
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY manquante dans les variables d’environnement")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# =========================
# Fonctions internes
# =========================
def get_embedding(text, model="text-embedding-3-large"):
    text = (text or "").replace("\n", " ").strip()
    if not text:
        return np.zeros(1, dtype=float)
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return np.array(response.data[0].embedding, dtype=float)

def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0.0
    if vec1.size == 1 and vec1[0] == 0 and vec2.size == 1 and vec2[0] == 0:
        return 0.0
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)

def extract_text_from_url(url, max_chars=12000):
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = " ".join(text.split())
    return text[:max_chars]

def seo_analysis(percent):
    if percent < 40:
        return (
            "Faible alignement sémantique",
            "La page ne répond pas clairement à l’intention du mot-clé.",
            [
                "Revoir l’angle éditorial de la page",
                "Ajouter une section dédiée au sujet du mot-clé",
                "Renforcer les termes sémantiques proches",
                "Vérifier l’intention de recherche ciblée"
            ]
        )

    if percent < 60:
        return (
            "Alignement sémantique partiel",
            "Le sujet est traité mais reste secondaire.",
            [
                "Renforcer le mot-clé dans les titres",
                "Clarifier l’introduction",
                "Développer une section plus explicite",
                "Ajouter des exemples concrets"
            ]
        )

    if percent < 75:
        return (
            "Bon alignement sémantique",
            "La page est pertinente pour le mot-clé.",
            [
                "Optimiser légèrement les titres",
                "Renforcer la profondeur sémantique",
                "Vérifier la cohérence globale du contenu"
            ]
        )

    return (
        "Très fort alignement sémantique",
        "La page est très bien alignée avec le mot-clé.",
        [
            "Maintenir la structure actuelle",
            "Travailler des requêtes secondaires",
            "Surveiller les performances SEO"
        ]
    )

def _get_gemini_model():
    """
    Compat multi-versions:
    - Certains SDK attendent "models/gemini-2.5-pro"
    - D’autres acceptent "gemini-2.5-pro"
    """
    if not GEMINI_AVAILABLE:
        raise RuntimeError("Lib Gemini non installée: google-generativeai")
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY manquante dans les variables d’environnement")

    # Essai 1
    try:
        return genai.GenerativeModel("models/gemini-2.5-pro")
    except Exception:
        pass

    # Essai 2
    try:
        return genai.GenerativeModel("gemini-2.5-pro")
    except Exception as e:
        raise RuntimeError(f"Impossible d’initialiser Gemini 2.5 Pro: {e}")

def rewrite_with_gemini(text, keyword):
    model = _get_gemini_model()

    safe_text = (text or "").strip()
    safe_keyword = (keyword or "").strip()

    if not safe_text:
        return None

    prompt = (
        "Tu es un assistant spécialisé en optimisation sémantique SEO.\n\n"
        "Objectif:\n"
        "Réécrire le texte ci-dessous afin d’augmenter sa proximité sémantique "
        "avec le mot-clé cible, sans modifier l’intention, le ton ni le niveau de détail.\n\n"
        "Contraintes:\n"
        "- Ne pas ajouter d’informations nouvelles\n"
        "- Ne pas rallonger significativement le texte\n"
        "- Ne pas faire de keyword stuffing\n"
        "- Rester naturel et fluide\n\n"
        f"Mot-clé cible: {safe_keyword}\n\n"
        "Texte à améliorer:\n"
        f"{safe_text}\n\n"
        "Fournis uniquement la version réécrite du texte."
    )

    response = model.generate_content(prompt)

    out = None
    if response is not None:
        out = getattr(response, "text", None)

    if not out or not str(out).strip():
        return None

    return str(out).strip()

# =========================
# Interface Streamlit
# =========================
st.set_page_config(
    page_title="Embedding : Analyse de similarité sémantique",
    layout="centered"
)

st.title("Embedding : Analyse de similarité sémantique")
st.caption("Outil interne Digitad")
st.markdown("Comparer un mot-clé avec une page web ou un contenu texte.")

st.divider()

source = st.radio(
    "Source du contenu à analyser",
    ["Page web (URL)", "Contenu texte"]
)

keyword = st.text_input(
    "Mot-clé",
    placeholder="Ex : assurance voyage canada"
)

if source == "Page web (URL)":
    url = st.text_input(
        "URL de la page",
        placeholder="https://www.exemple.com/page"
    )
    text_content = None
else:
    url = None
    text_content = st.text_area(
        "Contenu texte",
        height=180,
        placeholder="Collez ici le texte à analyser"
    )

st.divider()

enable_rewrite = st.checkbox(
    "Proposer une reformulation via Gemini 2.5 Pro si la similarité est faible",
    value=True,
    help="Nécessite GEMINI_API_KEY et le package google-generativeai"
)

rewrite_threshold = st.slider(
    "Seuil de similarité (%)",
    min_value=40,
    max_value=80,
    value=65,
    step=5
)

embedding_model = st.selectbox(
    "Modèle d’embedding OpenAI",
    ["text-embedding-3-large", "text-embedding-3-small"],
    index=0
)

st.divider()

if st.button("Analyser la similarité"):
    if not keyword:
        st.error("Merci de renseigner un mot-clé")
        st.stop()

    with st.spinner("Analyse en cours..."):
        if source == "Page web (URL)":
            if not url:
                st.error("Merci de renseigner une URL")
                st.stop()
            try:
                page_text = extract_text_from_url(url)
            except Exception as e:
                st.error(f"Impossible de récupérer la page: {e}")
                st.stop()
        else:
            if not text_content:
                st.error("Merci de coller un contenu texte")
                st.stop()
            page_text = text_content

        emb_keyword = get_embedding(keyword, model=embedding_model)
        emb_page = get_embedding(page_text, model=embedding_model)

        score = cosine_similarity(emb_keyword, emb_page)
        percent = score * 100.0

        niveau, diagnostic, recommandations = seo_analysis(percent)

        rewritten_text = None
        new_percent = None
        rewrite_error = None

        should_rewrite = bool(enable_rewrite and percent < float(rewrite_threshold))

        if should_rewrite:
            st.info(f"Similarité sous le seuil ({percent:.2f}% < {rewrite_threshold}%), reformulation déclenchée")
            try:
                rewritten_text = rewrite_with_gemini(page_text, keyword)
                if rewritten_text:
                    emb_rewritten = get_embedding(rewritten_text, model=embedding_model)
                    new_score = cosine_similarity(emb_keyword, emb_rewritten)
                    new_percent = new_score * 100.0
                else:
                    rewrite_error = "Gemini n’a retourné aucun contenu"
            except Exception as e:
                rewrite_error = str(e)

    st.subheader("Résultat")
    st.metric("Similarité sémantique", f"{percent:.2f}%")

    st.subheader("Analyse SEO")
    st.write(f"**Niveau :** {niveau}")
    st.write(f"**Diagnostic :** {diagnostic}")

    st.subheader("Recommandations")
    for reco in recommandations:
        st.write(f"- {reco}")

    if should_rewrite:
        if rewrite_error:
            st.warning(f"Reformulation non disponible: {rewrite_error}")
            if not GEMINI_AVAILABLE:
                st.caption("Installe: pip install google-generativeai")
            if not GEMINI_API_KEY:
                st.caption("Ajoute GEMINI_API_KEY dans les variables d’environnement")
        elif rewritten_text:
            st.divider()
            st.subheader("Proposition de reformulation")

            st.metric(
                "Nouvelle similarité sémantique",
                f"{new_percent:.2f}%",
                delta=f"{(new_percent - percent):.2f}%"
            )

            st.text_area(
                "Texte reformulé",
                rewritten_text,
                height=240
            )
