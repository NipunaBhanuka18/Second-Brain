# --- 1. Imports and Setup ---
import streamlit as st
import sqlmodel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
import datetime
import re
from streamlit_agraph import agraph, Node, Edge, Config
from typing import Optional, List

# --- 2. Database and AI Model Configuration ---
DATABASE_URL = "sqlite:///database.db"
engine = sqlmodel.create_engine(DATABASE_URL, echo=False)
SIMILARITY_THRESHOLD = 1.1

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


class Tag(sqlmodel.SQLModel, table=True):
    id: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    name: str = sqlmodel.Field(unique=True, index=True)
    __table_args__ = {'extend_existing': True}


class NoteTagLink(sqlmodel.SQLModel, table=True):
    note_id: Optional[int] = sqlmodel.Field(default=None, foreign_key="note.id", primary_key=True)
    tag_id: Optional[int] = sqlmodel.Field(default=None, foreign_key="tag.id", primary_key=True)
    __table_args__ = {'extend_existing': True}


class NoteVersion(sqlmodel.SQLModel, table=True):
    id: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    note_id: int = sqlmodel.Field(foreign_key="note.id")
    title: str
    content: str
    edited_at: datetime.datetime = sqlmodel.Field(default_factory=datetime.datetime.utcnow)
    __table_args__ = {'extend_existing': True}


class Attachment(sqlmodel.SQLModel, table=True):
    id: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    note_id: int = sqlmodel.Field(foreign_key="note.id")
    filename: str
    file_data: bytes = sqlmodel.Field(sa_column=sqlmodel.Column(sqlmodel.LargeBinary))
    __table_args__ = {'extend_existing': True}


class NoteLink(sqlmodel.SQLModel, table=True):
    source_id: Optional[int] = sqlmodel.Field(default=None, foreign_key="note.id", primary_key=True)
    target_id: Optional[int] = sqlmodel.Field(default=None, foreign_key="note.id", primary_key=True)
    __table_args__ = {'extend_existing': True}


# --- 5. Core Logic and AI Functions ---
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
        embedding_dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_dim))
        id_map = {}
    return index, id_map


def save_faiss_data(index, id_map):
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(ID_MAP_PATH, "wb") as f: pickle.dump(id_map, f)


def update_note_links(note: Note):
    with sqlmodel.Session(engine) as session:
        linked_titles = re.findall(r"\[\[(.*?)\]\]", note.content)
        session.exec(sqlmodel.delete(NoteLink).where(NoteLink.source_id == note.id))
        if linked_titles:
            target_notes = session.exec(sqlmodel.select(Note).where(Note.title.in_(linked_titles))).all()
            for target_note in target_notes:
                if target_note.id != note.id:
                    session.add(NoteLink(source_id=note.id, target_id=target_note.id))
        session.commit()


def get_outgoing_links(note_id: int):
    with sqlmodel.Session(engine) as session:
        return session.exec(sqlmodel.select(Note).join(NoteLink, Note.id == NoteLink.target_id).where(
            NoteLink.source_id == note_id)).all()


def get_backlinks(note_id: int):
    with sqlmodel.Session(engine) as session:
        return session.exec(sqlmodel.select(Note).join(NoteLink, Note.id == NoteLink.source_id).where(
            NoteLink.target_id == note_id)).all()


def get_all_notes():
    with sqlmodel.Session(engine) as session:
        return session.exec(sqlmodel.select(Note).order_by(Note.created_at.desc())).all()


def get_versions_for_note(note_id: int):
    with sqlmodel.Session(engine) as session:
        return session.exec(sqlmodel.select(NoteVersion).where(NoteVersion.note_id == note_id).order_by(
            NoteVersion.edited_at.desc())).all()


def get_attachments_for_note(note_id: int):
    with sqlmodel.Session(engine) as session:
        return session.exec(sqlmodel.select(Attachment).where(Attachment.note_id == note_id)).all()


def delete_attachment(attachment_id: int):
    with sqlmodel.Session(engine) as session:
        session.delete(session.get(Attachment, attachment_id))
        session.commit()


def get_related_notes(note: Note, index, id_map):
    if not note or index.ntotal == 0: return []
    query_embedding = np.array([model.encode(note.content)], dtype='float32')
    distances, ids = index.search(query_embedding, k=min(6, index.ntotal))
    similar_note_ids = [ids[0][i] for i, dist in enumerate(distances[0]) if
                        ids[0][i] != note.id and dist <= SIMILARITY_THRESHOLD]
    if not similar_note_ids: return []
    with sqlmodel.Session(engine) as session:
        return session.exec(sqlmodel.select(Note).where(Note.id.in_(similar_note_ids))).all()


def search_notes(query: str, index, id_map):
    if not query or index.ntotal == 0: return []
    query_embedding = np.array([model.encode(query)], dtype='float32')
    distances, ids = index.search(query_embedding, k=min(5, index.ntotal))
    similar_note_ids = [int(id) for id, dist in zip(ids[0], distances[0]) if dist <= SIMILARITY_THRESHOLD]
    if not similar_note_ids: return []
    with sqlmodel.Session(engine) as session:
        return session.exec(sqlmodel.select(Note).where(Note.id.in_(similar_note_ids))).all()


def get_all_tags():
    with sqlmodel.Session(engine) as session:
        return session.exec(sqlmodel.select(Tag)).all()


def get_tags_for_note(note_id: int):
    with sqlmodel.Session(engine) as session:
        return session.exec(sqlmodel.select(Tag).join(NoteTagLink).where(NoteTagLink.note_id == note_id)).all()


def get_notes_by_tag(tag_id: int):
    with sqlmodel.Session(engine) as session:
        return session.exec(sqlmodel.select(Note).join(NoteTagLink).where(NoteTagLink.tag_id == tag_id)).all()


def update_tags_for_note(note_id: int, selected_tag_names: List[str]):
    with sqlmodel.Session(engine) as session:
        all_tags_in_db = session.exec(sqlmodel.select(Tag)).all()
        tag_name_to_id = {tag.name: tag.id for tag in all_tags_in_db}
        for tag_name in selected_tag_names:
            if tag_name not in tag_name_to_id:
                new_tag = Tag(name=tag_name)
                session.add(new_tag)
                session.commit()
                session.refresh(new_tag)
                tag_name_to_id[new_tag.name] = new_tag.id
        session.exec(sqlmodel.delete(NoteTagLink).where(NoteTagLink.note_id == note_id))
        for tag_name in selected_tag_names:
            session.add(NoteTagLink(note_id=note_id, tag_id=tag_name_to_id[tag_name]))
        session.commit()


# --- 6. Application Startup ---
model = load_model()
create_db_and_tables()
faiss_index, note_id_map = load_faiss_data()

if 'selected_note' not in st.session_state: st.session_state.selected_note = None
if 'editing' not in st.session_state: st.session_state.editing = None
if 'graph_data' not in st.session_state: st.session_state.graph_data = None
if 'tag_filter' not in st.session_state: st.session_state.tag_filter = None

# --- 7. Main Application UI ---
st.set_page_config(layout="wide")
st.title("🧠 Second Brain")

col1, col2 = st.columns([1, 2])

# Sidebar (Column 1)
with col1:
    st.header("Create New Note")
    with st.form("new_note_form", clear_on_submit=True):
        new_title, new_content = st.text_input("Title"), st.text_area("Content")
        if st.form_submit_button("Save Note") and new_title and new_content:
            with sqlmodel.Session(engine) as session:
                note_to_add = Note(title=new_title, content=new_content)
                session.add(note_to_add)
                session.commit()
                session.refresh(note_to_add)
                embedding = np.array([model.encode(note_to_add.content)], dtype='float32')
                faiss_index.add_with_ids(embedding, np.array([note_to_add.id]))
                save_faiss_data(faiss_index, note_id_map)
                update_note_links(note_to_add)
                st.success("Note saved!")

    st.header("Search")
    search_query = st.text_input("Find notes by meaning...")
    if search_query:
        st.subheader(f"AI Search Results for '{search_query}'")
        search_results = search_notes(search_query, faiss_index, note_id_map)
        if search_results:
            for note in search_results:
                if st.button(note.title, key=f"search_{note.id}"):
                    st.session_state.selected_note = note
                    st.session_state.editing = None
        else:
            st.info("No semantic matches found.")
    else:
        st.header("Filter by Tag")
        all_tags = get_all_tags()
        if st.button("All Notes"): st.session_state.tag_filter = None
        for tag in all_tags:
            if st.button(tag.name, key=f"tag_{tag.id}"): st.session_state.tag_filter = tag

        st.header("Notes")
        if st.session_state.tag_filter:
            st.subheader(f"Tagged with: {st.session_state.tag_filter.name}")
            notes_to_display = get_notes_by_tag(st.session_state.tag_filter.id)
        else:
            notes_to_display = get_all_notes()

        for note in notes_to_display:
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
                edited_title = st.text_input("Title", value=note.title)
                all_tags, current_note_tags = get_all_tags(), get_tags_for_note(note.id)
                selected_tags = st.multiselect("Tags", options=[t.name for t in all_tags],
                                               default=[t.name for t in current_note_tags])
                edited_content = st.text_area("Content", value=note.content, height=250)
                uploaded_files = st.file_uploader("Add images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
                if st.form_submit_button("Save Changes"):
                    with sqlmodel.Session(engine) as session:
                        note_to_update = session.get(Note, note.id)
                        session.add(
                            NoteVersion(note_id=note.id, title=note_to_update.title, content=note_to_update.content))
                        note_to_update.title, note_to_update.content = edited_title, edited_content
                        session.add(note_to_update)
                        session.commit()
                        session.refresh(note_to_update)

                        faiss_index.remove_ids(np.array([note.id]))
                        new_embedding = np.array([model.encode(edited_content)], dtype='float32')
                        faiss_index.add_with_ids(new_embedding, np.array([note.id]))
                        save_faiss_data(faiss_index, note_id_map)

                        for uploaded_file in uploaded_files:
                            session.add(Attachment(note_id=note.id, filename=uploaded_file.name,
                                                   file_data=uploaded_file.getvalue()))
                        session.commit()

                    update_note_links(note_to_update)
                    update_tags_for_note(note.id, selected_tags)
                    st.session_state.selected_note = session.get(Note, note.id)
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
                        session.exec(sqlmodel.delete(NoteLink).where(
                            (NoteLink.source_id == note.id) | (NoteLink.target_id == note.id)))
                        session.exec(sqlmodel.delete(Attachment).where(Attachment.note_id == note.id))
                        session.exec(sqlmodel.delete(NoteVersion).where(NoteVersion.note_id == note.id))
                        session.exec(sqlmodel.delete(NoteTagLink).where(NoteTagLink.note_id == note.id))
                        session.delete(session.get(Note, note.id))
                        session.commit()
                        faiss_index.remove_ids(np.array([note.id]))
                        save_faiss_data(faiss_index, note_id_map)
                    st.session_state.selected_note = None
                    st.success("Note deleted!")
                    st.experimental_rerun()

            st.write(note.content)
            current_tags = get_tags_for_note(note.id)
            if current_tags: st.markdown(f"**Tags:** {', '.join([f'`{t.name}`' for t in current_tags])}")

            st.subheader("Linked Notes")
            link_col1, link_col2 = st.columns(2)
            with link_col1:
                st.markdown("**Outgoing Links**")
                outgoing_links = get_outgoing_links(note.id)
                if outgoing_links:
                    for link in outgoing_links:
                        if st.button(f"→ {link.title}", key=f"out_{link.id}"):
                            st.session_state.selected_note = link
                            st.experimental_rerun()
                else:
                    st.info("No outgoing links.")
            with link_col2:
                st.markdown("**Backlinks (Linked From)**")
                backlinks = get_backlinks(note.id)
                if backlinks:
                    for link in backlinks:
                        if st.button(f"← {link.title}", key=f"back_{link.id}"):
                            st.session_state.selected_note = link
                            st.experimental_rerun()
                else:
                    st.info("No backlinks.")

            st.subheader("Attachments")
            attachments = get_attachments_for_note(note.id)
            if attachments:
                for att in attachments:
                    att_col1, att_col2 = st.columns([4, 1])
                    with att_col1:
                        st.image(att.file_data, caption=att.filename, use_column_width=True)
                    with att_col2:
                        if st.button("Delete Image", key=f"del_att_{att.id}"):
                            delete_attachment(att.id)
                            st.experimental_rerun()
            else:
                st.info("No attachments.")

            with st.expander("History"):
                versions = get_versions_for_note(note.id)
                if versions:
                    for version in versions:
                        st.markdown(f"**Version from:** `{version.edited_at.strftime('%Y-%m-%d %H:%M')}`")
                        if st.button("Revert to this version", key=f"revert_{version.id}"):
                            with sqlmodel.Session(engine) as session:
                                current_note = session.get(Note, note.id)
                                session.add(NoteVersion(note_id=note.id, title=current_note.title,
                                                        content=current_note.content))
                                current_note.title, current_note.content = version.title, version.content
                                session.add(current_note)
                                session.commit()
                                faiss_index.remove_ids(np.array([note.id]))
                                new_embedding = np.array([model.encode(version.content)], dtype='float32')
                                faiss_index.add_with_ids(new_embedding, np.array([note.id]))
                                save_faiss_data(faiss_index, note_id_map)
                                update_note_links(current_note)
                            st.session_state.selected_note = session.get(Note, note.id)
                            st.success("Note reverted!")
                            st.experimental_rerun()
                else:
                    st.info("No previous versions found.")

            st.subheader("Related Ideas")
            related = get_related_notes(note, faiss_index, note_id_map)
            if related:
                for r_note in related:
                    with st.expander(r_note.title): st.write(r_note.content)
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