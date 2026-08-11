# Architecture Decision Records

## ADR-001 — Framework

**Decision:** LangChain

**Reason:** Greater control over the retrieval pipeline compared with LlamaIndex.

---

## ADR-002 — Privacy & Cost

**Decision:** Ollama (Llama 3.2 + Nomic)

**Reason:**

- Zero operational cost
- Full local execution
- Strong privacy guarantees

---

## ADR-003 — Retrieval Strategy

**Decision:** Hybrid Search + Cross-Encoder

**Reason:** Solve the precision loss problem of naïve vector search by combining semantic and keyword retrieval.

---

## ADR-004 — Evaluation

**Decision:** Ragas Framework

**Reason:** Quantitative evaluation of:

- Faithfulness
- Context Precision

Local sequential evaluation (`max_workers=1`) was chosen to avoid cloud rate limits.

---

## ADR-005 — API & Deployment

**Decision:** FastAPI + Docker + GitHub Actions

**Reason:**

- Stateless scaling
- Async performance
- Consistent deployments
- Automated CI validation

---