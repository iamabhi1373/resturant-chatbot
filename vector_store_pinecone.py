# vector_store_pinecone.py
import os
import uuid
from typing import List, Dict, Optional

from pinecone import Pinecone, ServerlessSpec

EMBED_DIM = 768  # Gemini text-embedding-004 produces 768-dim vectors

class PineconeVectorStore:
    """
    Pinecone-backed store that mirrors your SimpleVectorStore API:
      - add_documents(docs, metadata)
      - search(query, top_k=5)
    It expects an external 'embed_fn(text)->List[float]' for embeddings.
    """
    def __init__(
        self,
        embed_fn,
        index_name: str,
        api_key: Optional[str] = None,
        cloud: Optional[str] = None,
        region: Optional[str] = None,
        metric: str = "cosine",
        log_sink: Optional[list] = None
    ):
        self.embed_fn = embed_fn
        self.index_name = index_name
        self.metric = metric
        self._logs = log_sink if isinstance(log_sink, list) else None

        api_key = api_key or os.getenv("PINECONE_API_KEY", "")
        cloud = cloud or os.getenv("PINECONE_CLOUD", "aws")
        region = region or os.getenv("PINECONE_REGION", "us-east-1")

        if not api_key:
            raise ValueError("PINECONE_API_KEY missing")

        self.pc = Pinecone(api_key=api_key)

        # Create index if missing (serverless)
        existing = {ix["name"] for ix in self.pc.list_indexes()}
        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=EMBED_DIM,
                metric=self.metric,
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            self._log(f"Created Pinecone index '{self.index_name}' ({cloud}/{region}, {self.metric}, {EMBED_DIM}d)")

        self.index = self.pc.Index(self.index_name)

    def _log(self, msg: str):
        if self._logs is not None:
            self._logs.append(msg)

    def add_documents(self, documents: List[str], metadata: List[Dict]):
        if not documents:
            return
        vectors = []
        # embed in small batches to avoid timeouts if you extend this later
        for doc, meta in zip(documents, metadata):
            try:
                vec = self.embed_fn(doc)
                if not vec or len(vec) != EMBED_DIM:
                    raise ValueError("Bad embedding length")
                md = dict(meta)
                md["text"] = doc
                vectors.append({"id": str(uuid.uuid4()), "values": vec, "metadata": md})
            except Exception as e:
                self._log(f"Pinecone add_documents error: {e}")

        if vectors:
            self.index.upsert(vectors=vectors)
            self._log(f"Upserted {len(vectors)} docs to '{self.index_name}'")

    def search(self, query: str, top_k: int = 5, filter_eq: Optional[Dict] = None) -> List[Dict]:
        try:
            q = self.embed_fn(query)
            res = self.index.query(
                vector=q,
                top_k=top_k,
                include_metadata=True,
                filter=filter_eq or {}
            )
            matches = res.matches if hasattr(res, "matches") else res.get("matches", [])
            out = []
            for m in matches:
                md = m.metadata or {}
                out.append({
                    "document": md.get("text", ""),
                    "metadata": {k: v for k, v in md.items() if k != "text"},
                    "similarity": float(m.score)  # cosine similarity
                })
            return out
        except Exception as e:
            self._log(f"Pinecone search error: {e}")
            return []
