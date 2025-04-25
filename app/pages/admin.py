from db.utils import (
    add_pdf,
    remove_pdf,
    list_local_pdfs,
    PDF_DATA_DIR,
)

import os
import streamlit as st

st.set_page_config(page_title="Admin - Manage Documents", layout="wide")
st.title("📄 Document Management Console")
st.caption("Upload new PDF documents or remove existing ones from the knowledge base.")


os.makedirs(PDF_DATA_DIR, exist_ok=True)

st.header("Upload New Document")
uploaded_file = st.file_uploader(
    "Choose a PDF file", type="pdf", accept_multiple_files=False
)

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_path = os.path.join(PDF_DATA_DIR, file_name)

    if os.path.exists(file_path):
        st.warning(
            f"File '{uploaded_file.name}' already exists in the directory. Uploading will overwrite the local file and potentially add duplicate vectors if not deleted first."
        )
        overwrite = st.button("Overwrite and Re-index", key="overwrite_btn")
        if not overwrite:
            st.stop()

    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(
            f"Successfully saved '{uploaded_file.name}' locally to '{PDF_DATA_DIR}'."
        )

        # Process and add to Pinecone
        with st.spinner(
            f"Processing '{uploaded_file.name}' and adding to vector store... This may take a moment."
        ):
            add_pdf(file_name)
    except Exception as e:
        st.error(f"An error occurred during file saving or processing: {e}")


st.header("Manage Existing Documents")

if "confirming_delete" not in st.session_state:
    st.session_state.confirming_delete = None
if "confirming_reindex" not in st.session_state:
    st.session_state.confirming_reindex = None

pdf_files = list_local_pdfs(PDF_DATA_DIR)

if not pdf_files:
    st.info(f"No PDF documents found in the '{PDF_DATA_DIR}' directory.")
else:
    st.write(f"Found {len(pdf_files)} documents:")
    col_header_1, col_header_2, col_header_3 = st.columns([3, 1, 1], gap="small")

    with col_header_1:
        st.write("**Filename**")
    with col_header_2:
        st.write("**Delete**")
    with col_header_3:
        st.write("**Re-index**")

    st.markdown("---")

    for filename in sorted(pdf_files):
        col1, col2, col3 = st.columns([3, 1, 1], gap="small")

        with col1:
            st.write(filename)
            if st.session_state.confirming_delete == filename:
                st.warning(f"⚠️ Are you sure you want to delete '{filename}'?")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(
                        "Confirm Deletion",
                        key=f"confirm_del_{filename}",
                        type="primary",
                    ):
                        delete_success = False
                        with st.spinner(f"Deleting '{filename}'..."):
                            try:
                                remove_pdf(filename)
                                delete_success = True
                            except Exception as e:
                                st.error(f"Failed to delete '{filename}': {e}")

                        if delete_success:
                            st.success(f"Document '{filename}' successfully removed.")
                        st.session_state.confirming_delete = None
                        st.rerun()
                with c2:
                    if st.button("Cancel", key=f"cancel_del_{filename}"):
                        st.session_state.confirming_delete = None
                        st.rerun()

            if st.session_state.confirming_reindex == filename:
                st.warning(
                    f"⚠️ Are you sure you want to re-index '{filename}'? (Deletes existing vectors first)"
                )
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(
                        "Confirm Re-index",
                        key=f"confirm_reindex_{filename}",
                        type="primary",
                    ):
                        reindex_success = False
                        with st.spinner(f"Re-indexing '{filename}'..."):
                            try:
                                add_pdf(filename)
                                reindex_success = True
                            except Exception as e:
                                st.error(f"Failed to re-index '{filename}': {e}")

                        if reindex_success:
                            st.success(
                                f"Document '{filename}' successfully re-indexed."
                            )

                        st.session_state.confirming_reindex = None
                        st.rerun()
                with c2:
                    if st.button("Cancel", key=f"cancel_reindex_{filename}"):
                        st.session_state.confirming_reindex = None
                        st.rerun()

        with col2:
            if st.session_state.confirming_delete != filename:
                if st.button("Delete", key=f"del_{filename}"):
                    st.session_state.confirming_delete = filename
                    st.session_state.confirming_reindex = None
                    st.rerun()

        with col3:
            if st.session_state.confirming_reindex != filename:
                if st.button("Re-index", key=f"reindex_{filename}"):
                    st.session_state.confirming_reindex = filename
                    st.session_state.confirming_delete = None
                    st.rerun()

        st.markdown("---")
