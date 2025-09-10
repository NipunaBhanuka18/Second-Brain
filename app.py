# app.py

import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline

import database as db
import ui_components as ui

# This must be the first Streamlit command in your script
st.set_page_config(layout="wide")

# Load all resources (models, database, etc.)
@st.cache_resource
def load_resources():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    summarizer = pipeline("summarization", model="google/flan-t5-base")
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    db.create_db_and_tables()
    faiss_index = db.load_faiss_data(embedding_model)
    return embedding_model, summarizer, qa_pipeline, faiss_index

model, summarizer, qa_pipeline, faiss_index = load_resources()

# Initialize session state for different views and data
if 'view' not in st.session_state: st.session_state.view = 'default'
if 'selected_note' not in st.session_state: st.session_state.selected_note = None
if 'editing' not in st.session_state: st.session_state.editing = None
if 'tag_filter' not in st.session_state: st.session_state.tag_filter = None
if 'summary' not in st.session_state: st.session_state.summary = None

# --- Main Application UI ---
st.title("🧠 Second Brain")

# Render the sidebar and get the selected theme
selected_theme = ui.render_sidebar(model, faiss_index, qa_pipeline)

# The main content area is for primary interaction
ui.render_main_content(model, faiss_index, summarizer, qa_pipeline)