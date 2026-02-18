# In-Class Activity: Policy Document RAG System Using ChromaDB

## Objective

Build a working RAG system that answers questions from Kenyan policy documents using ChromaDB.

---

## 1. Data Requirement

**Choose one category:**

- Kenya Constitution
- Finance Bills
- Economic Surveys
- County Development Plans
- Parliamentary Acts

**Minimum requirements:**

- 3 PDFs
- At least 50 total pages

---

## 2. Implementation Tasks

### Step 1 — Extract Text

**Use:** `pypdf`

Create a function to extract text from PDFs.

**Explain:**

- Why PDFs must be converted to text
- Challenges in text extraction

---

### Step 2 — Chunking

**Use:** `RecursiveCharacterTextSplitter`

**Parameters:**

- `chunk_size`: 800–1000
- `chunk_overlap`: 150–200

**Explain:**

- Why chunking is necessary
- Why overlap improves retrieval

---

### Step 3 — Embeddings

**Use:** `BAAI/bge-small-en-v1.5`

Generate normalized embeddings for all chunks.

**Explain:**

- What embeddings represent
- Why semantic search is better than keyword search

---

### Step 4 — Store in ChromaDB

Create a persistent collection.

**Store:**

- `ids`
- `documents`
- `embeddings`
- `metadata` (source filename)

**Explain:**

- Why metadata is important
- What happens if embeddings are not stored

---

### Step 5 — Query & Retrieval

- Embed user query
- Retrieve top 3 results
- Display retrieved text and similarity score

**Explain:**

- What similarity score means
- Why lower cosine distance = better match

---

## 3. Evaluation (100 Marks)

| Component           | Marks |
|---------------------|-------|
| Extraction          | 15    |
| Chunking            | 20    |
| Embeddings          | 20    |
| ChromaDB Storage    | 20    |
| Query & Retrieval   | 15    |
| Explanations        | 10    |
| **Total**           | **100** |

---

> **Submission:** Working code + written explanations.
