# 💹 Facts-Only MF Assistant — RAG-Based Mutual Fund FAQ Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers **factual questions** about HDFC Mutual Fund schemes using **verified sources** from official AMC, SEBI, and AMFI websites. Every answer includes a citation link. No investment advice is given.

**Platform:** [Groww](https://groww.in)  
**AMC:** HDFC Mutual Fund  
**Last updated from sources:** December 2024

---

## 📋 Scope

### AMC & Schemes Covered

| Scheme | Category | Benchmark |
|---|---|---|
| HDFC Top 100 Fund | Large Cap | NIFTY 100 TRI |
| HDFC Flexi Cap Fund | Flexi Cap | NIFTY 500 TRI |
| HDFC ELSS Tax Saver | ELSS (3-yr lock-in, 80C) | NIFTY 500 TRI |
| HDFC Mid-Cap Opportunities Fund | Mid Cap | NIFTY Midcap 150 TRI |

### What the Assistant Answers
- Expense ratio (Regular and Direct plans)
- Exit load (scheme-wise)
- Minimum SIP amount
- Lock-in period (ELSS)
- Riskometer / risk level
- Benchmark index
- How to download account statements and capital gains statements
- Section 80C tax saving with ELSS

### What It Refuses (Politely)
- Investment recommendations ("Should I buy/sell?")
- Portfolio advice
- Return predictions or performance claims

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Advice Filter (regex patterns)
    │ (factual query)
    ▼
Embedding (sentence-transformers/all-MiniLM-L6-v2)
    │
    ▼
FAISS Vector Search → Top-5 relevant chunks from corpus
    │
    ▼
OpenAI GPT-3.5-turbo (with strict system prompt)
    │
    ▼
Answer (≤3 sentences) + Citation Link
```

**Tech Stack:**
- **UI:** Streamlit
- **Embeddings:** `sentence-transformers` (all-MiniLM-L6-v2) — free, no API key needed
- **Vector Store:** FAISS (in-memory, built at startup)
- **LLM:** OpenAI GPT-3.5-turbo (requires API key)
- **Corpus:** 22 text files sourced from HDFC AMC, SEBI, and AMFI

---

## 🚀 Setup Steps

### Prerequisites
- Python 3.10 or higher
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 1. Clone the repository
```bash
git clone https://github.com/shaiksayeed7/RAG-BASED-MUTUAL-FUND-FAQ-CHATBOT-.git
cd RAG-BASED-MUTUAL-FUND-FAQ-CHATBOT-
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your OpenAI API key
```bash
cp .env.example .env
# Edit .env and replace sk-...your-key-here... with your actual key
```

Or set the environment variable directly:
```bash
export OPENAI_API_KEY="sk-your-actual-key"   # macOS/Linux
# set OPENAI_API_KEY=sk-your-actual-key      # Windows CMD
```

### 5. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
├── app.py                  # Main Streamlit UI
├── rag_engine.py           # RAG pipeline (embeddings + FAISS + OpenAI)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore
├── README.md               # This file
└── data/
    ├── sources.csv         # 22 official source URLs
    ├── sample_qa.md        # 10+ sample Q&A with answers and citations
    └── corpus/             # 22 text files from official sources
        ├── 001_hdfc_top100_scheme.txt
        ├── 002_hdfc_flexicap_scheme.txt
        ├── 003_hdfc_elss_scheme.txt
        ├── 004_hdfc_midcap_scheme.txt
        ├── 005_hdfc_factsheet_nov2024.txt
        ├── 006_hdfc_top100_kim.txt
        ├── 007_hdfc_elss_kim.txt
        ├── 008_amfi_elss_faq.txt
        ├── 009_amfi_sip_faq.txt
        ├── 010_amfi_expense_ratio.txt
        ├── 011_amfi_exit_load.txt
        ├── 012_amfi_riskometer.txt
        ├── 013_amfi_benchmark.txt
        ├── 014_amfi_account_statement.txt
        ├── 015_sebi_riskometer_circular.txt
        ├── 016_sebi_categorization.txt
        ├── 017_sebi_ter_circular.txt
        ├── 018_hdfc_statement_guide.txt
        ├── 019_hdfc_tax_documents.txt
        ├── 020_hdfc_fee_charges.txt
        ├── 021_amfi_how_to_invest.txt
        └── 022_amfi_types_of_mf.txt
```

---

## 🔐 Privacy & Security

- **No PII collected or stored.** The app does not accept, process, or store PAN, Aadhaar, account numbers, OTPs, emails, or phone numbers.
- **Public sources only.** All corpus data is from official, publicly available pages.
- **No performance claims.** The assistant will not compute, predict, or compare returns.

---

## ⚠️ Disclaimer

> This assistant provides **factual information only**, sourced exclusively from official HDFC Mutual Fund, SEBI, and AMFI websites. It does **not** provide investment advice, return predictions, or portfolio recommendations. For investment decisions, please consult a **SEBI-registered investment advisor**. Mutual fund investments are subject to market risks. Read all scheme-related documents carefully before investing.

---

## 🧠 Known Limits

1. **Data currency:** Corpus data was collected in December 2024. Expense ratios, NAVs, and AUM figures change frequently — always verify from official sources.
2. **Scheme scope:** Only 4 HDFC schemes are covered. Questions about other AMCs or schemes outside scope may not be answered accurately.
3. **LLM dependence:** Requires an active OpenAI API key. Without it, the app will show an error.
4. **No real-time data:** The chatbot does not fetch live NAV, live expense ratios, or current portfolio data.
5. **Tax regime changes:** Tax rules can change with annual budgets. Always verify current tax treatment from official sources.

---

## 📚 Sources

See [`data/sources.csv`](data/sources.csv) for the full list of 22 official URLs used.

Key sources include:
- HDFC Mutual Fund website: https://www.hdfcfund.com
- AMFI Investor Education: https://www.amfiindia.com/investor-corner
- SEBI Circulars: https://www.sebi.gov.in/legal/circulars
- CAMS Statement Portal: https://www.camsonline.com

---

## 📝 Sample Q&A

See [`data/sample_qa.md`](data/sample_qa.md) for 10+ sample questions with the assistant's answers and source citations.
