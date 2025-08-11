# 🤖🚗 AI Driving Assistant
A Retrieval-Augmented Generation (RAG) chatbot for answering country-specific driving regulation questions using provided documents as the knowledge source.


## ⚙️ Installation

```bash
git clone ...
cd RAG
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

## 📚 Preparation: Knowledge Base

### 📂 Structure
Knowledge Base contain driving regulations, which organized by country and language in `documetns/`.
`documents/` folder should have this structure: 
```
documents/
├── Czechia/
│   ├── cs/
│   │   ├── regulation1.pdf
│   │   └── regulation2.pdf
│   └── en/
└── UK/
    └── en/
```
📄 Supported file types: `.pdf`

### 📝 Example documents
You can create example knowledge base using scripts `scripts/example_documents.sh` and `scripts/example_documents.ps1`: 

```bash
# 🐧🍎 POSIX shell (Linux, macOS, BSD, etc.)
./scripts/example_documents.sh

# 🪟 Windows
Set-ExecutionPolicy -Scope Process Bypass -Force
.\scripts\example_documents.ps1
```

## ▶️ Usage

### 🏁 Basic command
```bash
python3 main.py --country [Belarus|Czechia|UK] 
```

### ⚙️ Optional arguments

| Argument                 | Description                                                          |
| ------------------------ | -------------------------------------------------------------------- |
| `--country`              | **Required.** Country name (`Belarus`, `Czechia`, , `UK`)            |
| `--language`             | Optional. Language folder name (e.g., `be`, `en`, `cs`)              |
| `--embedding_model`      | Override embedding model in locale config                            |
| `--rewriter_model`       | Override rewriter LLM                                                |
| `--guardrails_model`     | Override guardrails LLM                                              |
| `--reranker_model`       | Override reranker cross-encoder                                      |
| `--conversational_llm`   | Override conversational LLM                                          |
| `--documents_path`       | Path to Documents (default: `documents`)                             |
| `--db_path`              | Path to FAISS DB (default: `vectorstore`)                            |
| `--vectorstore_recreate` | Recreate vectorstore from documents, if it exists (default: `False`) |

### Available commands during chatting

+ `/help` ❓ -  Show help message
+ `/quit` 🚪 - Exit the assistant
+ `/reload` 🔄 - Reload the retriever with updated documents"""



```