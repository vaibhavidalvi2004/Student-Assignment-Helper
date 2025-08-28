# 🎓 Student Assignment Helper (RAG-based)

A Retrieval-Augmented Generation (RAG) app that processes assignment PDFs, creates summaries, and answers questions with inline citations via an interactive Gradio UI.

## ✨ Features
- RAG pipeline using **HuggingFace MiniLM**, **LangChain**, and **FAISS**
- Process and index PDFs (50+ pages) with page-level metadata
- Automated summaries & real-time Q&A with inline citations
- Batch processing & JSON export of metadata/chunks
- Interactive **Gradio** dashboard

## 🚀 Quickstart
```bash
# 1) Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) enable a chat LLM via Hugging Face
#    Create a .env file and put: HF_TOKEN=your_hf_token_here

# 4) Run the app
python student_assignment_helper.py
```
Open the local Gradio link shown in the terminal, upload PDFs, and explore **Summarize**, **Q&A with Citations**, and **Export** tabs.

## 🧩 Project Structure
```
.
├── student_assignment_helper.py   # Main app
├── requirements.txt               # Python dependencies
├── README.md                      # Project docs
├── .gitignore                     # Ignore temp/env files
└── LICENSE                        # MIT license (edit name/year)
```

## ⚙️ Environment (optional)
Create a file named `.env` with:
```
HF_TOKEN=your_hf_token_here
```
If not set, the app still works (it will fall back to extractive summaries and non-LLM answers).

## 🛠️ Troubleshooting
- If `faiss-cpu` fails to install on Windows, try:
  - Upgrading pip: `python -m pip install --upgrade pip`
  - Using a virtual environment as shown above
  - Alternatively, use WSL or Conda environments

## 📜 License
MIT — see `LICENSE`.
