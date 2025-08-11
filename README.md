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
| `--top_k`                | Number of top documents to return after reranking (default: `5`)     |

### Available commands during chatting

+ `/help` ❓ -  Show help message
+ `/quit` 🚪 - Exit the assistant
+ `/reload` 🔄 - Reload the retriever with updated documents"""

### Example usage

1. UK (United Kingdom), English language
```
python3 main.py --country UK --language en

❓ Ask your question: Which side of the road should I ride on?

💬 Answer:  Ride on the left — in the same direction as the traffic. Where noted (e.g. one‑way streets) move with the flow, keep to the left, and on narrow or busy roads (or round bends) ride single file (never more than two abreast).
```

2. Czechia, Czech language
```
python3 main.py --country Czechia --language cs


❓ Zadejte svou otázku: Čím se liší „zastavit“ od „zastavit vozidlo“?

💬 Odpověď:  Podle uvedeného kontextu:
- „zastavit“ (definice o)) znamená úmyslně uvést vozidlo do klidu na dobu nezbytnou k bezodkladnému nastoupení/vystoupení osob nebo k naložení/složení nákladu (krátké, záměrné zastavení).
- „zastavit vozidlo“ (definice p) a použití v souvislosti s nehodou) značí přerušit jízdu z důvodu nezávislého na vůli řidiče – tedy nucené, neúmyslné zastavení (např. v důsledku poruchy nebo nehody); v případě nehody navíc „neprodleně zastavit vozidlo“ znamená učinit tak bezodkladně, aby nedošlo k dalšímu ohrožení.
```

3. Belarus, Belarusian language
```
❓ Задайце сваё пытанне: што такое абгон? 

💬 Адказ:  Па дадзеным кантэксце «абгон» — гэта праезд аднаго транспартнага сродку міма іншага, г.зн. абагнанне іншага транспартнага сродку.
```

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