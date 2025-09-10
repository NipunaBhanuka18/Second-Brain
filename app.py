# app.py

import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline

import database as db
import ui_components as ui

# This must be the first Streamlit command in your script
st.set_page_config(layout="wide")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_generative_models():
    summarizer = pipeline("summarization", model="google/flan-t5-base")
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    return summarizer, qa_pipeline

# Load all resources
model = load_embedding_model()
summarizer, qa_pipeline = load_generative_models()
db.create_db_and_tables()
faiss_index, note_id_map = db.load_faiss_data(model)

# Initialize session state
if 'selected_note' not in st.session_state: st.session_state.selected_note = None
if 'editing' not in st.session_state: st.session_state.editing = None
if 'graph_data' not in st.session_state: st.session_state.graph_data = None
if 'tag_filter' not in st.session_state: st.session_state.tag_filter = None
if 'view' not in st.session_state: st.session_state.view = 'default'
if 'summary' not in st.session_state: st.session_state.summary = None

# --- Main Application UI ---
st.title("🧠 Second Brain")

# Render the sidebar and get the selected theme
# The sidebar is now defined in ui_components and is mobile-friendly
ui.render_sidebar(model, faiss_index, note_id_map, qa_pipeline)

# The main content area is now the default part of the app
ui.render_main_content(model, faiss_index, note_id_map, summarizer)