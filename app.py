# --- 1. Imports and Setup ---
# This block imports all the necessary libraries for the application to run.
import streamlit as st
from sqlmodel import Field, Session, SQLModel, create_engine, select
import datetime
from sentence_transformers import SentenceTransformer
import chromadb
from streamlit_agraph import agraph, Node, Edge, Config
from typing import Optional, List
import os

if 'STREAMLIT_IN_CLOUD' in os.environ or 'STREAMLIT_SERVER_RUNNING_IN_CLOUD' in os.environ:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# --- 2. Database and AI Model Configuration ---
# Here, we define the database connection, load the AI models, and set constants.

# Define the location of our local SQLite database file.
DATABASE_URL = "sqlite:///database.db"
# The engine is the entry point to our database. `echo=False` keeps the logs clean.
engine = create_engine(DATABASE_URL, echo=False)

# This is a constant that determines how "similar" notes need to be to be considered related.
# A lower number means stricter similarity.
SIMILARITY_THRESHOLD = 1.1

# --- 3. Database Model Definition ---
# This class defines the structure of the 'note' table in our database.
# SQLModel makes it work as both a Python object and a database table.
class Note(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    content: str
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow, nullable=False)
    # This argument prevents an error when Streamlit re-runs the script.
    __table_args__ = {'extend_existing': True}

# --- 4. Core Logic and AI Functions ---
# This section contains all the functions that interact with the database and the AI model.

# This function uses Streamlit's cache to load the AI model and database connection
# only once, which makes the app much faster.
@st.cache_resource
def load_models_and_db():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="notes")
    return model, collection

# Function to create the database and its tables if they don't already exist.
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

# Function to get all notes from the database, ordered by most recent.
def get_all_notes():
    with Session(engine) as session:
        return session.exec(select(Note).order_by(Note.created_at.desc())).all()

# This is the main AI function to find notes related to a given note.
def get_related_notes(note: Note):
    if not note:
        return []
    # Create an embedding for the note's content.
    query_embedding = model.encode(note.content).tolist()
    # Query the vector database for similar notes.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=6,
        include=['distances']
    )
    # Filter the results based on our similarity threshold.
    similar_note_ids = []
    ids, distances = results['ids'][0], results['distances'][0]
    for id_str, dist in zip(ids, distances):
        current_id = int(id_str)
        if current_id != note.id and dist <= SIMILARITY_THRESHOLD:
            similar_note_ids.append(current_id)
    if not similar_note_ids:
        return []
    # Retrieve the full note details from the main database.
    with Session(engine) as session:
        statement = select(Note).where(Note.id.in_(similar_note_ids))
        return session.exec(statement).all()

# This function powers the search bar, finding notes related to a text query.
def search_notes(query: str):
    if not query:
        return []
    # Create an embedding for the search query.
    query_embedding = model.encode(query).tolist()
    # Query the vector database.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=['distances']
    )
    # Filter the results.
    similar_note_ids = []
    ids, distances = results['ids'][0], results['distances'][0]
    for id_str, dist in zip(ids, distances):
        if dist <= SIMILARITY_THRESHOLD:
            similar_note_ids.append(int(id_str))
    if not similar_note_ids:
        return []
    # Retrieve the full note details.
    with Session(engine) as session:
        statement = select(Note).where(Note.id.in_(similar_note_ids))
        return session.exec(statement).all()

# --- 5. Application Startup ---
# This code runs once when the app starts to set everything up.

# Load the AI model and vector DB connection.
model, collection = load_models_and_db()
# Create the database tables.
create_db_and_tables()

# Initialize Streamlit's "session state" to keep track of variables.
if 'selected_note' not in st.session_state:
    st.session_state.selected_note = None
if 'editing' not in st.session_state:
    st.session_state.editing = None
if 'graph_data' not in st.session_state:
    st.session_state.graph_data = None

# --- 6. Main Application UI ---
# This is the main part of the script that renders the user interface.

st.set_page_config(layout="wide")
st.title("🧠 Second Brain")

# The UI is organized into two main columns.
col1, col2 = st.columns([1, 2])

# The first column acts as a sidebar for controls and lists.
with col1:
    # A form for creating a new note.
    st.header("Create New Note")
    with st.form("new_note_form", clear_on_submit=True):
        new_title = st.text_input("Title")
        new_content = st.text_area("Content")
        submitted = st.form_submit_button("Save Note")
        if submitted and new_title and new_content:
            with Session(engine) as session:
                note_to_add = Note(title=new_title, content=new_content)
                session.add(note_to_add)
                session.commit()
                session.refresh(note_to_add)
                embedding = model.encode(note_to_add.content).tolist()
                collection.add(ids=[str(note_to_add.id)], embeddings=[embedding])
                st.success("Note saved!")

    # The AI-powered search bar.
    st.header("Search")
    search_query = st.text_input("Find notes by meaning...")
    if search_query:
        search_results = search_notes(search_query)
        st.subheader("Search Results")
        if search_results:
            for note in search_results:
                if st.button(note.title, key=f"search_{note.id}"):
                    st.session_state.selected_note = note
                    st.session_state.editing = None # Exit edit mode
        else:
            st.info("No relevant notes found.")

    # The list of all notes.
    st.header("All Notes")
    all_notes = get_all_notes()
    for note in all_notes:
        if st.button(note.title, key=f"note_{note.id}"):
            st.session_state.selected_note = note
            st.session_state.editing = None # Exit edit mode
            st.session_state.graph_data = None # Clear graph

# The second column is the main content area.
with col2:
    if st.session_state.selected_note:
        note = st.session_state.selected_note
        # Display the UI for editing a note if the user clicked "Edit".
        if st.session_state.editing == note.id:
            st.header(f"Editing: {note.title}")
            with st.form("edit_note_form"):
                edited_title = st.text_input("Title", value=note.title)
                edited_content = st.text_area("Content", value=note.content, height=250)
                submitted = st.form_submit_button("Save Changes")
                if submitted:
                    with Session(engine) as session:
                        note_to_update = session.get(Note, note.id)
                        note_to_update.title, note_to_update.content = edited_title, edited_content
                        session.add(note_to_update)
                        session.commit()
                        session.refresh(note_to_update)
                        new_embedding = model.encode(edited_content).tolist()
                        collection.update(ids=[str(note.id)], embeddings=[new_embedding])
                    st.session_state.selected_note = note_to_update
                    st.session_state.editing = None
                    st.success("Note updated!")
                    st.experimental_rerun()
        # Display the UI for viewing a note.
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
                    with Session(engine) as session:
                        session.delete(session.get(Note, note.id))
                        session.commit()
                        collection.delete(ids=[str(note.id)])
                    st.session_state.selected_note = None
                    st.success("Note deleted!")
                    st.experimental_rerun()
            st.write(note.content)
            # Display related notes.
            st.subheader("Related Ideas")
            related = get_related_notes(note)
            if related:
                for r_note in related:
                    with st.expander(r_note.title):
                        st.write(r_note.content)
            else:
                st.info("No related ideas found.")
    else:
        st.header("Select or create a note.")

    # The Knowledge Graph visualization section.
    st.header("Knowledge Graph")
    if st.button("Generate Full Graph"):
        nodes, edges = [], []
        all_notes_for_graph = get_all_notes()
        note_ids = {note.id for note in all_notes_for_graph}
        for note in all_notes_for_graph:
            nodes.append(Node(id=str(note.id), label=note.title, size=15))
            related = get_related_notes(note)
            for r_note in related:
                if r_note.id in note_ids:
                    edges.append(Edge(source=str(note.id), target=str(r_note.id)))
        st.session_state.graph_data = (nodes, edges)

    if st.session_state.graph_data:
        nodes, edges = st.session_state.graph_data
        config = Config(width=800, height=600, directed=True, physics=True, hierarchical=False)
        st.info("Displaying the knowledge graph of all notes.")
        agraph(nodes=nodes, edges=edges, config=config)