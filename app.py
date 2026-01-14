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

de

