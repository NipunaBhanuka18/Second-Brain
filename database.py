# database.py

import sqlmodel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
import datetime
import re
from typing import Optional, List

# --- Constants and Engine ---
DATABASE_URL = "sqlite:///database.db"
engine = sqlmodel.create_engine(DATABASE_URL, echo=False)
FAISS_INDEX_PATH = "faiss_index.idx"
ID_MAP_PATH = "id_map.pkl"

# --- Database Models ---
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

# --- Core Logic Functions ---
def create_db_and_tables():
    sqlmodel.SQLModel.metadata.create_all(engine)

def load_faiss_data(model: SentenceTransformer):
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
    with open(ID_MAP_PATH, "wb") as f:
        pickle.dump(id_map, f)

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
        return session.exec(sqlmodel.select(Note).join(NoteLink, Note.id == NoteLink.target_id).where(NoteLink.source_id == note_id)).all()

def get_backlinks(note_id: int):
    with sqlmodel.Session(engine) as session:
        return session.exec(sqlmodel.select(Note).join(NoteLink, Note.id == NoteLink.source_id).where(NoteLink.target_id == note_id)).all()

def get_all_notes():
    with sqlmodel.Session(engine) as session:
        return session.exec(sqlmodel.select(Note).order_by(Note.created_at.desc())).all()

def get_versions_for_note(note_id: int):
    with sqlmodel.Session(engine) as session:
        return session.exec(sqlmodel.select(NoteVersion).where(NoteVersion.note_id == note_id).order_by(NoteVersion.edited_at.desc())).all()

def get_attachments_for_note(note_id: int):
    with sqlmodel.Session(engine) as session:
        return session.exec(sqlmodel.select(Attachment).where(Attachment.note_id == note_id)).all()

def delete_attachment(attachment_id: int):
    with sqlmodel.Session(engine) as session:
        session.delete(session.get(Attachment, attachment_id))
        session.commit()

def get_related_notes(note: Note, model: SentenceTransformer, index, id_map, threshold: float):
    if not note or index.ntotal == 0: return []
    query_embedding = np.array([model.encode(note.content)], dtype='float32')
    distances, ids = index.search(query_embedding, k=min(6, index.ntotal))
    similar_note_ids = [ids[0][i] for i, dist in enumerate(distances[0]) if ids[0][i] != note.id and dist <= threshold]
    if not similar_note_ids: return []
    with sqlmodel.Session(engine) as session:
        return session.exec(sqlmodel.select(Note).where(Note.id.in_(similar_note_ids))).all()

def search_notes(query: str, model: SentenceTransformer, index, id_map, threshold: float):
    if not query or index.ntotal == 0: return []
    query_embedding = np.array([model.encode(query)], dtype='float32')
    distances, ids = index.search(query_embedding, k=min(5, index.ntotal))
    similar_note_ids = [int(id) for id, dist in zip(ids[0], distances[0]) if dist <= threshold]
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
                session.commit(); session.refresh(new_tag)
                tag_name_to_id[new_tag.name] = new_tag.id
        session.exec(sqlmodel.delete(NoteTagLink).where(NoteTagLink.note_id == note_id))
        for tag_name in selected_tag_names:
            session.add(NoteTagLink(note_id=note_id, tag_id=tag_name_to_id[tag_name]))
        session.commit()