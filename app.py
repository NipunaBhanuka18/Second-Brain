# --- 1. Imports and Setup ---
import streamlit as st
import sqlmodel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
import datetime
from streamlit_agraph import agraph, Node, Edge, Config
from typing import Optional, List

# --- 2. Database and AI Model Configuration ---
DATABASE_URL = "sqlite:///database.db"
engine = sqlmodel.create_engine(DATABASE_URL, echo=False)
SIMILARITY_THRESHOLD = 1.1  # This is a distance threshold for FAISS

# --- 3. File Paths for FAISS Persistence ---
FAISS_INDEX_PATH = "faiss_index.idx"
ID_MAP_PATH = "id_map.pkl"


# --- 4. Database Model Definition ---
class Note(sqlmodel.SQLModel, table=True):
    id: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    title: str
    content: str
    created_at: datetime.datetime = sqlmodel.Field(default_factory=datetime.datetime.utcnow, nullable=False)
    __table_args__ = {'extend_existing': True}


# --- 5. Core Logic and AI Functions (Rewritten for FAISS) ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


def create_db_and_tables():
    sqlmodel.SQLModel.metadata.create_all(engine)


def load_faiss_data():
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(ID_MAP_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(ID_MAP_PATH, "rb") as f:
            id_map = pickle.load(f)
    else:
        # Get the dimension of the embeddings from the model
        embedding_dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_dim))
        id_map = {}
    return index, id_map


def save_faiss_data(index, id_map):
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(ID_MAP_PATH, "wb") as f:
        pickle.dump(id_map, f)


def get_all_notes():
    with sqlmodel.Session(engine) as session:
        return session.exec(sqlmodel.select(Note).order_by(Note.created_at.desc())).all()


def get_related_notes(note: Note, index, id_map):
    if not note or index.ntotal == 0:
        return []
    query_embedding = np.array([model.encode(note.content)], dtype='float32')
    # FAISS returns distances and the IDs we stored in the index
    distances, ids = index.search(query_embedding, k=min(6, index.ntotal))

    similar_note_ids = []
    for i, dist in enumerate(distances[0]):
        retrieved_id = ids[0][i]
        if retrieved_id != note.id and dist <= SIMILARITY_THRESHOLD:
            similar_note_ids.append(retrieved_id)

    if not similar_note_ids:
        return []

    with sqlmodel.Session(engine) as session:
        statement = sqlmodel.select(Note).where(Note.id.in_(similar_note_ids))
        return session.exec(statement).all()


def search_notes(query: str, index, id_map):
    if not query or index.ntotal == 0:
        return []
    query_embedding = np.array([model.encode(query)], dtype='float32')
    distances, ids = index.search(query_embedding, k=min(5, index.ntotal))

    similar_note_ids = [int(id) for id, dist in zip(ids[0], distances[0]) if dist <= SIMILARITY_THRESHOLD]

    if not similar_note_ids:
        return []

    with sqlmodel.Session(engine) as session:
        statement = sqlmodel.select(Note).where(Note.id.in_(similar_note_ids))
        return session.exec(statement).all()


# --- 6. Application Startup ---
model = load_model()
create_db_and_tables()
faiss_index, note_id_map = load_faiss_data()

if 'selected_note' not in st.session_state:
    st.session_state.selected_note = None
if 'editing' not in st.session_state:
    st.session_state.editing = None
if 'graph_data' not in st.session_state:
    st.session_state.graph_data = None

# --- 7. Main Application UI ---
st.set_page_config(layout="wide")
st.title("🧠 Second Brain")

col1, col2 = st.columns([1, 2])

# --- UI Layout ---
# ... (The st.set_page_config, st.title, and col1, col2 lines are the same) ...

# The first column acts as a sidebar for controls and lists.
with col1:
    # A form for creating a new note.
    st.header("Create New Note")
    with st.form("new_note_form", clear_on_submit=True):
        new_title = st.text_input("Title")
        new_content = st.text_area("Content")
        submitted = st.form_submit_button("Save Note")
        if submitted and new_title and new_content:
            with sqlmodel.Session(engine) as session:
                note_to_add = Note(title=new_title, content=new_content)
                session.add(note_to_add)
                session.commit()
                session.refresh(note_to_add)

                # Add to FAISS
                embedding = np.array([model.encode(note_to_add.content)], dtype='float32')
                faiss_index.add_with_ids(embedding, np.array([note_to_add.id]))
                save_faiss_data(faiss_index, note_id_map)
                st.success("Note saved!")

    # --- NEW HYBRID SEARCH SECTION ---
    st.header("Search")
    all_notes = get_all_notes()  # Get all notes once for both lists
    search_query = st.text_input("Type to get suggestions or press Enter for AI search...")

    # 1. INSTANT SUGGESTIONS (KEYWORD AUTOCOMPLETE)
    if search_query:
        # Filter notes whose titles start with the search query (case-insensitive)
        suggestions = [note for note in all_notes if note.title.lower().startswith(search_query.lower())]
        if suggestions:
            st.subheader("Suggestions")
            for note in suggestions:
                if st.button(note.title, key=f"suggestion_{note.id}"):
                    st.session_state.selected_note = note
                    st.session_state.editing = None

        # 2. DEEP SEARCH (AI-POWERED)
        st.subheader(f"AI Search Results for '{search_query}'")
        search_results = search_notes(search_query, faiss_index, note_id_map)
        if search_results:
            for note in search_results:
                if st.button(note.title, key=f"search_{note.id}"):
                    st.session_state.selected_note = note
                    st.session_state.editing = None
        else:
            st.info("No semantic matches found.")

    # The list of all notes (only shown when not searching)
    if not search_query:
        st.header("All Notes")
        for note in all_notes:
            if st.button(note.title, key=f"note_{note.id}"):
                st.session_state.selected_note = note
                st.session_state.editing = None
                st.session_state.graph_data = None

# Main Content (Column 2)
with col2:
    if st.session_state.selected_note:
        note = st.session_state.selected_note
        if st.session_state.editing == note.id:
            st.header(f"Editing: {note.title}")
            with st.form("edit_note_form"):
                edited_title, edited_content = st.text_input("Title", value=note.title), st.text_area("Content",
                                                                                                      value=note.content,
                                                                                                      height=250)
                if st.form_submit_button("Save Changes"):
                    with sqlmodel.Session(engine) as session:
                        note_to_update = session.get(Note, note.id)
                        note_to_update.title, note_to_update.content = edited_title, edited_content
                        session.add(note_to_update)
                        session.commit()

                        # Update FAISS (remove old, add new)
                        faiss_index.remove_ids(np.array([note.id]))
                        new_embedding = np.array([model.encode(edited_content)], dtype='float32')
                        faiss_index.add_with_ids(new_embedding, np.array([note.id]))
                        save_faiss_data(faiss_index, note_id_map)

                    st.session_state.selected_note = note_to_update
                    st.session_state.editing = None
                    st.success("Note updated!")
                    st.experimental_rerun()
        else:
            c1, c2, c3 = st.columns([4, 1, 1])
            with c1:
                st.header(note.title)
            with c2:
                if st.button("Edit"):
                    st.session_state.editing = note.id
                    st.experimental_rerun()
            with c3:
                if st.button("Delete", type="primary"):
                    with sqlmodel.Session(engine) as session:
                        session.delete(session.get(Note, note.id))
                        session.commit()
                        faiss_index.remove_ids(np.array([note.id]))
                        save_faiss_data(faiss_index, note_id_map)
                    st.session_state.selected_note = None
                    st.success("Note deleted!")
                    st.experimental_rerun()

            st.write(note.content)
            st.subheader("Related Ideas")
            related = get_related_notes(note, faiss_index, note_id_map)
            if related:
                for r_note in related:
                    with st.expander(r_note.title):
                        st.write(r_note.content)
            else:
                st.info("No related ideas found.")
    else:
        st.header("Select or create a note.")

    st.header("Knowledge Graph")
    if st.button("Generate Full Graph"):
        nodes, edges = [], []
        all_notes_for_graph = get_all_notes()
        note_ids_set = {note.id for note in all_notes_for_graph}
        for note in all_notes_for_graph:
            nodes.append(Node(id=str(note.id), label=note.title, size=15))
            related = get_related_notes(note, faiss_index, note_id_map)
            for r_note in related:
                if r_note.id in note_ids_set:
                    edges.append(Edge(source=str(note.id), target=str(r_note.id)))
        st.session_state.graph_data = (nodes, edges)

    if st.session_state.graph_data:
        nodes, edges = st.session_state.graph_data
        config = Config(width=800, height=600, directed=True, physics=True, hierarchical=False)
        st.info("Displaying the knowledge graph of all notes.")
        agraph(nodes=nodes, edges=edges, config=config)