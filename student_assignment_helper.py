pip install gradio pypdf langchain sentence-transformers faiss-cpu transformers huggingface_hub python-dotenv

import os
import io
import json
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import gradio as gr
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Optional: Chat model via HuggingFace endpoints (if you have a HF token and model)
try:
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    HF_CHAT_AVAILABLE = True
except Exception:
    HF_CHAT_AVAILABLE = False

# -----------------------------
# Global state
# -----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_TOKEN", "")

embeddings = None  # HuggingFaceEmbeddings instance
vectorstores: Dict[str, FAISS] = {}  # file_id -> FAISS store
file_metadata: Dict[str, Dict] = {}  # file_id -> metadata

# LLM wrapper (ChatHuggingFace) - optional
llm = None

# -----------------------------
# Utility functions
# -----------------------------

def init_embeddings():
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return embeddings


def init_llm(repo_id: str = "tiiuae/falcon-7b-instruct") -> Optional[ChatHuggingFace]:
    global llm
    if not HF_CHAT_AVAILABLE:
        print("HuggingFace chat integration not available. Skipping LLM init.")
        return None
    if not HF_TOKEN:
        print("HF_TOKEN not set. Skipping LLM init.")
        return None
    try:
        endpoint = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=HF_TOKEN, task="text-generation", temperature=0.2, max_new_tokens=512)
        llm = ChatHuggingFace(llm=endpoint)
        # quick test
        _ = llm.invoke("Hello")
        print("LLM initialized")
        return llm
    except Exception as e:
        print("Failed to initialize HuggingFace LLM:", e)
        return None


def extract_text_from_pdf(file_obj: io.BytesIO) -> Tuple[str, List[Tuple[int, str]]]:
    """Return full text and list of (page_number, page_text)"""
    reader = PdfReader(file_obj)
    pages = []
    full_text_parts = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i + 1, text))
        full_text_parts.append(text)
    full_text = "\n\n".join(full_text_parts)
    return full_text, pages


def chunk_text_with_metadata(filename: str, pages: List[Tuple[int, str]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents: List[Document] = []
    for page_num, text in pages:
        if not text or not text.strip():
            continue
        # to help preserve page-level metadata, include page id
        chunks = splitter.split_text(text)
        for idx, chunk in enumerate(chunks):
            metadata = {
                "source": filename,
                "page": page_num,
                "chunk_id": f"{filename}_p{page_num}_c{idx}"
            }
            documents.append(Document(page_content=chunk, metadata=metadata))
    return documents


def build_vectorstore_for_file(file_id: str, documents: List[Document]) -> FAISS:
    init_embeddings()
    if not documents:
        raise ValueError("No documents to index")
    vs = FAISS.from_documents(documents, embeddings)
    vectorstores[file_id] = vs
    return vs


def summarize_documents(documents: List[Document], max_chunks: int = 6) -> str:
    """Create a short summary by asking the LLM (if available) or doing a simple extractive summary."""
    # create context from top chunks
    context = "\n\n".join([f"(From {d.metadata.get('source')}, page {d.metadata.get('page')}):\n{d.page_content[:800]}" for d in documents[:max_chunks]])
    prompt = f"You are a helpful assistant. Summarize the following extracted chunks into concise bullet points (3-8 bullets). Include short citations like [filename:page].\n\n{context}\n\nSummary:" 

    if llm:
        try:
            response = llm.invoke(prompt)
            if response and response.content:
                return response.content
        except Exception as e:
            print("LLM summarize failed:", e)

    # fallback: naive extractive summary: pick sentences with high length
    bullets = []
    for d in documents[:max_chunks]:
        text = d.page_content.strip()
        s = text.split('\n')[:3]
        candidate = ' '.join([seg.strip() for seg in s if seg.strip()])
        if candidate:
            bullets.append(f"- {candidate} [{d.metadata.get('source')}:{d.metadata.get('page')}]")
    if not bullets:
        return "(No extractable summary found)"
    return "\n".join(bullets)


def answer_question_with_citations(question: str, selected_files: List[str], k: int = 5) -> str:
    """Retrieve top chunks across selected files and ask LLM to answer with inline citations."""
    # collect retrievers
    retrieved_docs: List[Document] = []
    for fid in selected_files:
        if fid not in vectorstores:
            continue
        retriever = vectorstores[fid].as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(question)
        retrieved_docs.extend(docs)

    if not retrieved_docs:
        return "No relevant content found in the selected assignments. Try uploading/processing files first."

    # Build context with source tags
    context_parts = []
    for d in retrieved_docs[:10]:
        src = d.metadata.get('source', 'unknown')
        page = d.metadata.get('page', '?')
        chunk_preview = d.page_content.replace('\n', ' ')[:600]
        context_parts.append(f"[Source: {src}, page {page}]\n{chunk_preview}\n---\n")
    context = "\n\n".join(context_parts)

    prompt = f"You are a helpful knowledgeable assistant. Use the provided extracted chunks from student assignment PDFs to answer the question. When you state facts, add short citations like [filename:page]. If the answer is not present in the context, say 'Not found in provided sources'.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    if llm:
        try:
            response = llm.invoke(prompt)
            if response and response.content:
                return response.content
        except Exception as e:
            print("LLM answer failed:", e)

    # Fallback: simple combined excerpt answer (no generative LLM)
    answer = "".join([f"From {d.metadata.get('source')} (page {d.metadata.get('page')}):\n{d.page_content[:600]}\n\n" for d in retrieved_docs[:6]])
    return "(No LLM available)\n\n" + answer

# -----------------------------
# Gradio app functions
# -----------------------------

def process_uploaded_files(uploaded_files) -> Tuple[str, List[str]]:
    """Process one or multiple uploaded PDFs. Returns a status message and list of processed file_ids."""
    processed = []
    errors = []

    if not uploaded_files:
        return "No files uploaded.", []

    for file_obj in uploaded_files:
        try:
            raw = file_obj.read()
            file_basename = Path(file_obj.name).name if hasattr(file_obj, 'name') else f"uploaded_{len(file_metadata)+1}.pdf"
            file_id = f"{file_basename}_{int(datetime.now().timestamp())}"

            full_text, pages = extract_text_from_pdf(io.BytesIO(raw))
            docs = chunk_text_with_metadata(file_basename, pages)

            if not docs:
                errors.append(f"{file_basename}: no text extracted")
                continue

            vs = build_vectorstore_for_file(file_id, docs)

            file_metadata[file_id] = {
                "filename": file_basename,
                "uploaded_at": datetime.now().isoformat(),
                "pages": len(pages),
                "chunks": len(docs)
            }
            processed.append(file_id)
        except Exception as e:
            tb = traceback.format_exc()
            errors.append(f"Error processing file: {str(e)}\n{tb}")

    status = f"Processed {len(processed)} files."
    if errors:
        status += "\nErrors:\n" + "\n".join(errors[:5])
    return status, processed


def summarize_file(file_id: str) -> str:
    if file_id not in vectorstores:
        return "File not processed or unknown file id."
    vs = vectorstores[file_id]
    # pull top-k docs to summarize
    docs = vs.similarity_search("", k=6)
    return summarize_documents(docs, max_chunks=6)


def batch_summarize(file_ids: List[str]) -> Dict[str, str]:
    results = {}
    for fid in file_ids:
        try:
            results[fid] = summarize_file(fid)
        except Exception as e:
            results[fid] = f"Error: {e}"
    return results


def export_processed_data(file_ids: List[str]) -> str:
    """Export metadata and sample chunks as a downloadable JSON (base64 link generated by Gradio download)."""
    export = {"exported_at": datetime.now().isoformat(), "files": {}}
    for fid in file_ids:
        meta = file_metadata.get(fid, {})
        vs = vectorstores.get(fid)
        sample_chunks = []
        if vs:
            docs = vs.similarity_search("", k=5)
            sample_chunks = [{"text": d.page_content[:1000], "metadata": d.metadata} for d in docs]
        export["files"][fid] = {"metadata": meta, "sample_chunks": sample_chunks}

    json_bytes = json.dumps(export, indent=2).encode("utf-8")
    return json_bytes

# -----------------------------
# Build Gradio UI
# -----------------------------

def build_ui():
    with gr.Blocks(title="Student Assignment Helper - RAG", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎓 Student Assignment Helper\nUpload assignment PDFs, auto-summarize, and ask questions with citations.")

        with gr.Tab("Upload & Batch Process"):
            uploaded = gr.File(file_count="multiple", label="Upload assignment PDFs (PDF)")
            process_btn = gr.Button("📥 Process Uploads")
            process_status = gr.Textbox(label="Status", interactive=False)
            processed_list = gr.Dataframe(headers=["file_id", "filename", "pages", "chunks", "uploaded_at"], interactive=False)

        with gr.Tab("Summarize"):
            summarize_select = gr.Dropdown(choices=[], label="Select processed file to summarize")
            summarize_btn = gr.Button("📝 Summarize Selected File")
            summary_output = gr.Markdown()
            batch_summarize_btn = gr.Button("🧺 Batch Summarize All")
            batch_summary_output = gr.JSON()

        with gr.Tab("Q&A with Citations"):
            qa_files = gr.CheckboxGroup(choices=[], label="Select files to query")
            question = gr.Textbox(label="Question", lines=2)
            ask_btn = gr.Button("🤖 Ask")
            answer_output = gr.Textbox(label="Answer (with citations)", lines=12, interactive=False)

        with gr.Tab("Export"):
            export_select = gr.CheckboxGroup(choices=[], label="Select files to export")
            export_btn = gr.Button("📤 Export Selected (JSON)")
            export_download = gr.File(label="Download export")

        # Callbacks
        def on_process(files):
            status, processed = process_uploaded_files(files)
            # update processed_list rows
            rows = []
            for fid in processed:
                meta = file_metadata.get(fid, {})
                rows.append([fid, meta.get('filename'), meta.get('pages'), meta.get('chunks'), meta.get('uploaded_at')])
            return status, gr.Dataframe.update(value=rows), gr.Dropdown.update(choices=list(file_metadata.keys())), gr.CheckboxGroup.update(choices=list(file_metadata.keys())), gr.CheckboxGroup.update(choices=list(file_metadata.keys()))

        process_btn.click(on_process, inputs=[uploaded], outputs=[process_status, processed_list, summarize_select, qa_files, export_select])

        summarize_btn.click(lambda fid: summarize_file(fid), inputs=[summarize_select], outputs=[summary_output])

        batch_summarize_btn.click(lambda: batch_summarize(list(file_metadata.keys())), inputs=None, outputs=[batch_summary_output])

        ask_btn.click(lambda q, files: answer_question_with_citations(q, files), inputs=[question, qa_files], outputs=[answer_output])

        def on_export(selected):
            if not selected:
                return None
            data = export_processed_data(selected)
            # write to temp file and return path for Gradio File component
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            tmp.write(data)
            tmp.flush()
            tmp.close()
            return tmp.name

        export_btn.click(on_export, inputs=[export_select], outputs=[export_download])

    return demo

# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    print("Initializing embeddings...")
    init_embeddings()
    print("Attempting to init LLM (optional)")
    try:
        init_llm()
    except Exception:
        pass

    app = build_ui()
    app.launch(server_name="0.0.0.0", debug=True, share=False)
