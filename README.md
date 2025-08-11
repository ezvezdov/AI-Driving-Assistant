# 🤖🚗 AI Driving Assistant
A Retrieval-Augmented Generation (RAG) chatbot for answering country-specific driving regulation questions using provided documents as the knowledge source.

🎓 This project was developed as part of the **Language Technologies in Practice** ([NPFL128](https://ufal.mff.cuni.cz/courses/npfl128)) course at **CUNI MFF**.

💡 Inspired by and building upon the concepts from the blog post [Emerging Patterns in Building GenAI Products](https://martinfowler.com/articles/gen-ai-patterns/) by Bharani Subramaniam & Martin Fowler.

📑 I also created a presentation summarizing the blog post, available here: [Google Slides link](https://docs.google.com/presentation/d/1FYDCcIA5clFAhHEnmLEPT0t6cucdbtr5G1096XSKaAk/edit?usp=sharing).

## ⚙️ Installation

```bash
# 🐧🍎 POSIX shell (Linux, macOS, BSD, etc.)
git clone https://github.com/ezvezdov/AI-Driving-Assistant.git
cd AI-Driving-Assistant
python3 -m venv .venv
source .venv/bin/activate
pip install .

# 🪟 Windows
git clone https://github.com/ezvezdov/AI-Driving-Assistant.git
cd AI-Driving-Assistant
python -m venv .venv
.\.venv\Scripts\activate
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
# 🐧🍎 POSIX shell (Linux, macOS, BSD, etc.)

export OPENAI_API_KEY='your-api-key'

python3 main.py --country [Belarus|Czechia|UK]
```

```ps1
# 🪟 Windows

setx OPENAI_API_KEY 'your-api-key'

python main.py --country [Belarus|Czechia|UK]
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



## 🧠 How It Works

### Hight-level flow

```
┌─────────────────────┐      ┌───────────────────┐
│  PDFs by locale     │      │  locales/<lang>/  │
│  documents/<C>/<L>  │      │    config.py      │
└─────────┬───────────┘      └─────────┬─────────┘
          │                             │
          ▼                             ▼
   ProcessorPDF                   Runtime config
(load → split chunks)             (models/prompts)
          │
          ▼
  HybridRetriever ────────────────────────────────────────┐
  (build/load FAISS + BM25)                               │
          │                                               │
          ▼                                               │
    Rewriter LLM  →  {q1, q2, …, qn}                      │
          │                         per qi:               │
          │                     retrieve (FAISS+BM25)     │
          └──────────────►  aggregate candidate docs ◄────┘
                                   │
                                   ▼
                      CrossEncoder Reranker (top-k)
                                   │
                                   ▼
                    Concatenate context (top-k chunks)
                                   │
                                   ▼
                     Conversational LLM (answer)
                                   │
                                   ▼
                         Output Guardrails check

```