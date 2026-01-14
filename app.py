import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import os

# =========================
# Sécurité - mot de passe
# =========================
def check_password():
    def password_entered():
        if st.session_state["password"] == "Digitad2025!":
            st.session_state["password_correct"] = True
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


# Bloque toute l’app si le mot de passe est faux
if not check_password():
    st.stop()

# =========================
# Configuration OpenAI
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# Fonctions internes
# =========================
def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

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

if st.button("Analyser la similarité"):
    if not keyword:
        st.error("Merci de renseigner un mot-clé")
    else:
        with st.spinner("Analyse en cours..."):
            if source == "Page web (URL)":
                if not url:
                    st.error("Merci de renseigner une URL")
                    st.stop()
                page_text = extract_text_from_url(url)
            else:
                if not text_content:
                    st.error("Merci de coller un contenu texte")
                    st.stop()
                page_text = text_content

            emb_keyword = get_embedding(keyword)
            emb_page = get_embedding(page_text)

            score = cosine_similarity(emb_keyword, emb_page)
            percent = score * 100

            niveau, diagnostic, recommandations = seo_analysis(percent)

        st.subheader("Résultat")
        st.metric("Similarité sémantique", f"{percent:.2f}%")

        st.subheader("Analyse SEO")
        st.write(f"**Niveau :** {niveau}")
        st.write(f"**Diagnostic :** {diagnostic}")

        st.subheader("Recommandations")
        for reco in recommandations:
            st.write(f"- {reco}")
