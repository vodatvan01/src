flowchart LR
  %% ===== FRONTEND =====
  subgraph FE[Frontend (React 18)]
    U[Người dùng]
    R18[React 18 SPA<br/>Chat UI + Mic + WebSocket]
  end

  %% ===== BACKEND =====
  subgraph BE[Backend (FastAPI)]
    GW[FastAPI Gateway<br/>/chat (REST/WebSocket)]
    LG[LangGraph RAG Orchestrator]
    subgraph NODES[LangGraph Nodes]
      N1[Input Validation<br/>(Bedrock Guardrails)]
      N2[Intent Analysis & Routing]
      N3[Document Retrieval (Hybrid)<br/>VectorRetriever + Reranker]
      N4[Answer Generation<br/>(LLM Prompting)]
      N5[Output Validation<br/>(Bedrock Guardrails)]
    end
    OBS[Langfuse SDK<br/>Tracing/Errors/Metrics]
  end

  %% ===== DATA & MODELS =====
  subgraph DATA[Data & Model Layer]
    DOCS[Raw Docs / PDFs / HTML]
    EMB[Embedding: Qwen3 0.6B]
    RER[Re-ranker: BAAI/bge-reranker-v2-m3]
    QDR[Qdrant Vector DB]
    LLM[Llama 4 (AWS Bedrock)<br/>via Boto3]
  end

  %% ===== FLOWS =====
  U <--> R18
  R18 -- HTTP/WS --> GW
  GW --> LG

  %% LangGraph node order
  LG --> N1 --> N2 --> N3 --> N4 --> N5
  N1 -.policy check.-> OBS
  N5 -.policy check.-> OBS

  %% Retrieval path
  N3 -->|embed| EMB
  N3 -->|vector search| QDR
  N3 -->|rerank top-K| RER
  N4 -->|generate| LLM

  %% Observability
  LG -. emit traces .-> OBS
  N2 -. intents & metadata .-> OBS
  N3 -. retrieval metrics .-> OBS
  N4 -. token usage .-> OBS

  %% Ingest
  DOCS -->|chunk & embed| EMB -->|store vectors| QDR

  %% Response
  N5 --> LG --> GW --> R18 --> U
