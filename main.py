from fastapi import FastAPI, HTTPException
from sqlmodel import Field, Session, SQLModel, create_engine, select
from typing import Optional, List
import datetime

# --- AI Model & Vector DB Imports ---
from sentence_transformers import SentenceTransformer
import chromadb

# --- Database Setup (Same as before) ---
DATABASE_URL = "sqlite:///database.db"
engine = create_engine(DATABASE_URL, echo=False)  # Set echo=False for cleaner output


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


# --- Data Model Definition (Same as before) ---
class Note(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    content: str
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow, nullable=False)


# --- AI Model & Vector DB Initialization ---
# 1. Load the sentence transformer model. This is done once when the app starts.
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Initialize the ChromaDB client. It will store data in memory or on disk.
chroma_client = chromadb.Client()
# 3. Get or create a "collection" which is like a table for our vectors.
collection = chroma_client.get_or_create_collection(name="notes")

# --- FastAPI App Initialization ---
app = FastAPI(title="Second Brain API")


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to your Second Brain"}


@app.post("/notes/", response_model=Note)
def create_note(note: Note):
    with Session(engine) as session:
        # Save the note text to the regular database
        session.add(note)
        session.commit()
        session.refresh(note)

        # --- AI Part: Generate and store embedding ---
        # 1. Generate the embedding for the note's content
        embedding = model.encode(note.content).tolist()

        # 2. Store the embedding in ChromaDB, using the note's ID as the unique identifier
        collection.add(
            ids=[str(note.id)],
            embeddings=[embedding]
        )

        return note


@app.get("/notes/", response_model=List[Note])
def read_notes():
    with Session(engine) as session:
        notes = session.exec(select(Note)).all()
        return notes


# --- The "Magic" Endpoint (with Debugging) ---
@app.get("/notes/{note_id}/related", response_model=List[Note])
def get_related_notes(note_id: int):
    SIMILARITY_THRESHOLD = 1.1

    with Session(engine) as session:
        target_note = session.get(Note, note_id)
        if not target_note:
            raise HTTPException(status_code=404, detail="Note not found")

        query_embedding = model.encode(target_note.content).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=6,
            include=['distances']
        )

        # --- TEMPORARY DEBUGGING CODE ---
        print("\n--- Similarity Search Results ---")
        ids = results['ids'][0]
        distances = results['distances'][0]
        for id_str, dist in zip(ids, distances):
            print(f"Note ID: {id_str}, Distance: {dist}")
        print("---------------------------------\n")
        # --- END OF DEBUGGING CODE ---

        similar_note_ids = []
        for id_str, dist in zip(ids, distances):
            current_id = int(id_str)
            if current_id != note_id and dist <= SIMILARITY_THRESHOLD:
                similar_note_ids.append(current_id)

        if not similar_note_ids:
            return []

        statement = select(Note).where(Note.id.in_(similar_note_ids))
        related_notes = session.exec(statement).all()

        return related_notes