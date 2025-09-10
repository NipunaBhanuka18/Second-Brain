# ui_components.py

import streamlit as st
from streamlit_quill import st_quill
import database as db
import numpy as np
import sqlmodel
from streamlit_agraph import agraph, Node, Edge, Config


# --- AI ENHANCEMENT FUNCTIONS ---
def summarize_text(text: str, summarizer):
    if len(text.split()) < 50:
        return "Note is too short to summarize."
    result = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return result[0]['summary_text']


def cluster_notes(notes: list, model, num_clusters: int):
    from sklearn.cluster import KMeans
    if len(notes) < num_clusters:
        return None
    embeddings = model.encode([note.content for note in notes])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(embeddings)
    clusters = {i: [] for i in range(num_clusters)}
    for note, label in zip(notes, kmeans.labels_):
        clusters[label].append(note)
    return clusters


def answer_question(query: str, model, index, id_map, threshold, qa_pipeline):
    relevant_notes = db.search_notes(query, model, index, id_map, threshold)
    if not relevant_notes:
        return "I couldn't find any relevant notes to answer your question."
    context = "\n".join([f"Note Title: {note.title}\nContent: {note.content}" for note in relevant_notes])
    prompt = f"Based on the following notes, please answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    result = qa_pipeline(prompt, max_length=200)
    return result[0]['generated_text']


# --- SIDEBAR UI ---
def render_sidebar(model, faiss_index, note_id_map, qa_pipeline):
    with st.sidebar:
        st.header("Second Brain")
        theme = st.radio("Select Theme", ["Dark", "Light"], index=0)

        st.header("Create New Note")
        with st.form("new_note_form", clear_on_submit=True):
            new_title, new_content = st.text_input("Title"), st_quill(placeholder="Start writing...", html=True,
                                                                      key="new_note_quill")
            if st.form_submit_button("Save Note") and new_title and new_content:
                with sqlmodel.Session(db.engine) as session:
                    note_to_add = db.Note(title=new_title, content=new_content)
                    session.add(note_to_add);
                    session.commit();
                    session.refresh(note_to_add)
                    embedding = np.array([model.encode(note_to_add.content)], dtype='float32')
                    faiss_index.add_with_ids(embedding, np.array([note_to_add.id]))
                    db.save_faiss_data(faiss_index, note_id_map)
                    db.update_note_links(note_to_add)
                    st.success("Note saved!")

        st.header("🤖 AI Assistant")
        question = st.text_input("Ask a question about your notes...")
        if st.button("Get Answer"):
            st.session_state.view = 'qa'
            with st.spinner("Synthesizing answer..."):
                st.session_state.qa_answer = answer_question(question, model, faiss_index, note_id_map,
                                                             db.SIMILARITY_THRESHOLD, qa_pipeline)

        st.header("🔍 Topic Clustering")
        all_notes = db.get_all_notes()
        num_clusters = st.number_input("Number of topics to find:", min_value=2, max_value=10, value=3)
        if st.button("Group Notes"):
            if len(all_notes) >= num_clusters:
                st.session_state.view = 'cluster'
                with st.spinner("Clustering notes..."):
                    st.session_state.clusters = cluster_notes(all_notes, model, num_clusters)
            else:
                st.warning("Not enough notes to create clusters.")

        st.header("Search")
        search_query = st.text_input("Find notes by meaning...")
        if search_query:
            st.subheader(f"AI Search Results for '{search_query}'")
            search_results = db.search_notes(search_query, model, faiss_index, note_id_map, db.SIMILARITY_THRESHOLD)
            if search_results:
                for note in search_results:
                    if st.button(note.title, key=f"search_{note.id}"):
                        st.session_state.selected_note = note;
                        st.session_state.editing = None
            else:
                st.info("No semantic matches found.")
        else:
            st.header("Filter by Tag")
            all_tags = db.get_all_tags()
            if st.button("All Notes"): st.session_state.tag_filter = None
            for tag in all_tags:
                if st.button(tag.name, key=f"tag_{tag.id}"): st.session_state.tag_filter = tag

            st.header("Notes")
            if st.session_state.tag_filter:
                st.subheader(f"Tagged with: {st.session_state.tag_filter.name}")
                notes_to_display = db.get_notes_by_tag(st.session_state.tag_filter.id)
            else:
                notes_to_display = all_notes
            for note in notes_to_display:
                if st.button(note.title, key=f"note_{note.id}"):
                    st.session_state.selected_note = note
                    st.session_state.editing = None;
                    st.session_state.graph_data = None
                    st.session_state.view = 'default'

    return theme


# --- MAIN CONTENT UI ---
def render_main_content(model, faiss_index, note_id_map, summarizer):
    if st.session_state.view == 'qa':
        st.header("AI-Generated Answer")
        st.write(st.session_state.get("qa_answer", "Something went wrong."))
        if st.button("Back to Notes"):
            st.session_state.view = 'default';
            st.experimental_rerun()
    elif st.session_state.view == 'cluster' and st.session_state.get('clusters'):
        st.header("Note Clusters by Topic")
        for i, notes_in_cluster in st.session_state.clusters.items():
            with st.expander(f"**Topic {i + 1}** ({len(notes_in_cluster)} notes)"):
                for note in notes_in_cluster: st.markdown(f"- **{note.title}**")
        if st.button("Back to Notes"):
            st.session_state.view = 'default';
            st.experimental_rerun()
    elif st.session_state.selected_note:
        st.session_state.view = 'default'
        note = st.session_state.selected_note
        if st.session_state.get('editing') == note.id:
            st.header(f"Editing: {note.title}")
            with st.form("edit_note_form"):
                edited_title = st.text_input("Title", value=note.title)
                all_tags, current_note_tags = db.get_all_tags(), db.get_tags_for_note(note.id)
                selected_tags = st.multiselect("Tags", options=[t.name for t in all_tags],
                                               default=[t.name for t in current_note_tags])

                st.write("Content")
                edited_content = st_quill(value=note.content, html=True, key="edit_quill")

                st.subheader("Contextual Suggestions")
                if edited_content:
                    contextual_suggestions = db.search_notes(edited_content, model, faiss_index, note_id_map,
                                                             db.SIMILARITY_THRESHOLD)
                    if contextual_suggestions:
                        for suggestion in contextual_suggestions:
                            if suggestion.id != note.id: st.info(f"**Related idea:** {suggestion.title}")
                    else:
                        st.write("Keep writing to see related ideas...")

                uploaded_files = st.file_uploader("Add images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

                if st.form_submit_button("Save Changes"):
                    if not edited_content:
                        st.error("Content cannot be empty.")
                    else:
                        with sqlmodel.Session(db.engine) as session:
                            note_to_update = session.get(db.Note, note.id)
                            session.add(db.NoteVersion(note_id=note.id, title=note_to_update.title,
                                                       content=note_to_update.content))
                            note_to_update.title, note_to_update.content = edited_title, edited_content
                            session.add(note_to_update);
                            session.commit()
                            faiss_index.remove_ids(np.array([note.id]))
                            new_embedding = np.array([model.encode(edited_content)], dtype='float32')
                            faiss_index.add_with_ids(new_embedding, np.array([note.id]))
                            db.save_faiss_data(faiss_index, note_id_map)
                            for uploaded_file in uploaded_files:
                                session.add(db.Attachment(note_id=note.id, filename=uploaded_file.name,
                                                          file_data=uploaded_file.getvalue()))
                            session.commit()
                        db.update_note_links(note_to_update)
                        db.update_tags_for_note(note.id, selected_tags)
                        st.session_state.selected_note = session.get(db.Note, note.id)
                        st.session_state.editing = None
                        st.success("Note updated!");
                        st.experimental_rerun()
        else:  # Viewing Mode
            c1, c2, c3, c4 = st.columns([3, 1, 1, 2])
            with c1:
                st.header(note.title)
            with c2:
                if st.button("Edit"): st.session_state.editing = note.id; st.experimental_rerun()
            with c3:
                if st.button("Delete", type="primary"):
                    with sqlmodel.Session(db.engine) as session:
                        session.exec(sqlmodel.delete(db.NoteLink).where(
                            (db.NoteLink.source_id == note.id) | (db.NoteLink.target_id == note.id)))
                        session.exec(sqlmodel.delete(db.Attachment).where(db.Attachment.note_id == note.id))
                        session.exec(sqlmodel.delete(db.NoteVersion).where(db.NoteVersion.note_id == note.id))
                        session.exec(sqlmodel.delete(db.NoteTagLink).where(db.NoteTagLink.note_id == note.id))
                        session.delete(session.get(db.Note, note.id));
                        session.commit()
                        faiss_index.remove_ids(np.array([note.id]))
                        db.save_faiss_data(faiss_index, note_id_map)
                    st.session_state.selected_note = None
                    st.success("Note deleted!");
                    st.experimental_rerun()
            with c4:
                if st.button("✨ Auto-Summarize Note"):
                    with st.spinner("Generating summary..."):
                        st.session_state.summary = summarize_text(note.content, summarizer)

            if 'summary' in st.session_state and st.session_state.summary and st.session_state.selected_note.id == note.id:
                st.info(f"**AI Summary:** {st.session_state.summary}")

            st.markdown(note.content, unsafe_allow_html=True)
            current_tags = db.get_tags_for_note(note.id)
            if current_tags: st.markdown(f"**Tags:** {', '.join([f'`{t.name}`' for t in current_tags])}")
            st.subheader("Linked Notes");
            link_col1, link_col2 = st.columns(2)
            with link_col1:
                st.markdown("**Outgoing Links**")
                outgoing_links = db.get_outgoing_links(note.id)
                if outgoing_links:
                    for link in outgoing_links:
                        if st.button(f"→ {link.title}", key=f"out_{link.id}"):
                            st.session_state.selected_note = link;
                            st.experimental_rerun()
                else:
                    st.info("No outgoing links.")
            with link_col2:
                st.markdown("**Backlinks (Linked From)**")
                backlinks = db.get_backlinks(note.id)
                if backlinks:
                    for link in backlinks:
                        if st.button(f"← {link.title}", key=f"back_{link.id}"):
                            st.session_state.selected_note = link;
                            st.experimental_rerun()
                else:
                    st.info("No backlinks.")
            st.subheader("Attachments")
            attachments = db.get_attachments_for_note(note.id)
            if attachments:
                for att in attachments:
                    att_col1, att_col2 = st.columns([4, 1])
                    with att_col1:
                        st.image(att.file_data, caption=att.filename, use_column_width=True)
                    with att_col2:
                        if st.button("Delete Image", key=f"del_att_{att.id}"):
                            db.delete_attachment(att.id);
                            st.experimental_rerun()
            else:
                st.info("No attachments.")
            with st.expander("History"):
                versions = db.get_versions_for_note(note.id)
                if versions:
                    for version in versions:
                        st.markdown(f"**Version from:** `{version.edited_at.strftime('%Y-%m-%d %H:%M')}`")
                        if st.button("Revert to this version", key=f"revert_{version.id}"):
                            with sqlmodel.Session(db.engine) as session:
                                current_note = session.get(db.Note, note.id)
                                session.add(db.NoteVersion(note_id=note.id, title=current_note.title,
                                                           content=current_note.content))
                                current_note.title, current_note.content = version.title, version.content
                                session.add(current_note);
                                session.commit()
                                faiss_index.remove_ids(np.array([note.id]))
                                new_embedding = np.array([model.encode(version.content)], dtype='float32')
                                faiss_index.add_with_ids(new_embedding, np.array([note.id]))
                                db.save_faiss_data(faiss_index, note_id_map)
                                db.update_note_links(current_note)
                            st.session_state.selected_note = session.get(db.Note, note.id)
                            st.success("Note reverted!");
                            st.experimental_rerun()
                else:
                    st.info("No previous versions found.")
            st.subheader("AI-Suggested Related Ideas")
            related = db.get_related_notes(note, model, faiss_index, note_id_map, db.SIMILARITY_THRESHOLD)
            if related:
                for r_note in related:
                    with st.expander(r_note.title): st.write(r_note.content)
            else:
                st.info("No related ideas found.")
    else:
        st.header("Select or create a note.")
        if st.button("Generate Full Knowledge Graph"):
            st.session_state.view = 'graph';
            st.experimental_rerun()
    if st.session_state.view == 'graph':
        st.header("Your Second Brain Map")
        nodes, edges = [], []
        all_notes_for_graph = db.get_all_notes()
        note_ids_set = {note.id for note in all_notes_for_graph}
        for note in all_notes_for_graph:
            nodes.append(Node(id=str(note.id), label=note.title, size=15))
            related = db.get_related_notes(note, model, faiss_index, note_id_map, db.SIMILARITY_THRESHOLD)
            for r_note in related:
                if r_note.id in note_ids_set:
                    edges.append(Edge(source=str(note.id), target=str(r_note.id)))
        config = Config(width=800, height=600, directed=True, physics=True, hierarchical=False)
        clicked_node_id = agraph(nodes=nodes, edges=edges, config=config)
        if clicked_node_id:
            with sqlmodel.Session(db.engine) as session:
                clicked_note = session.get(db.Note, int(clicked_node_id))
                if clicked_note:
                    st.session_state.selected_note = clicked_note
                    st.session_state.view = 'default'
                    st.experimental_rerun()
        if st.button("Back to Notes"):
            st.session_state.view = 'default';
            st.experimental_rerun()