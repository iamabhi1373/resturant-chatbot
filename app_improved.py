# app_improved.py
"""
Improved Restaurant RAG Chatbot with better PDF processing and question answering.
"""

import os
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# LLM + embeddings SDK (Gemini wrapper)
import google.generativeai as genai

# PDF reader
import fitz  # PyMuPDF

# Try to import Pinecone wrapper, make it optional
try:
    from vector_store_pinecone import PineconeVectorStore
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    PineconeVectorStore = None

# -------------------------
# Config / env
# -------------------------
load_dotenv()
DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# -------------------------
# Domain models
# -------------------------
@dataclass
class CustomerFeedback:
    customer_name: str
    feedback: str
    rating: int
    date: str
    order_id: Optional[str] = None

@dataclass
class MenuItem:
    id: str
    name: str
    description: str
    category: str
    price: float
    ingredients: List[str]
    dietary_tags: List[str]
    spice_level: str

# -------------------------
# Simple in-memory vector store (fallback)
# -------------------------
class SimpleVectorStore:
    def __init__(self, embed_service):
        self.embed_service = embed_service
        self.documents: List[str] = []
        self.embeddings: List[List[float]] = []
        self.metadata: List[Dict] = []

    def add_documents(self, documents: List[str], metadata: List[Dict]):
        for doc, meta in zip(documents, metadata):
            emb = self.embed_service.generate_embedding(doc)
            self.documents.append(doc)
            self.embeddings.append(emb)
            self.metadata.append(meta)
            _log(f"[SimpleStore] Added doc: {doc[:80]}...")

    def search(self, query: str, top_k: int = 5):
        if not self.embeddings:
            return []
        q_emb = self.embed_service.generate_embedding(query)
        sims = [self._cosine_similarity(q_emb, e) for e in self.embeddings]
        idxs = np.argsort(sims)[-top_k:][::-1]
        return [
            {
                "document": self.documents[i],
                "metadata": self.metadata[i],
                "similarity": float(sims[i])
            }
            for i in idxs
        ]

    def _cosine_similarity(self, a, b):
        a = np.array(a); b = np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return 0.0 if denom == 0 else float(np.dot(a, b) / denom)

# -------------------------
# Gemini RAG service wrapper (embeddings + LLM)
# -------------------------
class GeminiRAGService:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY missing")
        genai.configure(api_key=api_key)
        # Use a generative model for text responses
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_embedding(self, text: str) -> List[float]:
        """
        Uses google.generativeai.embed_content to produce embeddings.
        Expects model "models/text-embedding-004" (768d).
        """
        try:
            resp = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            
            # Handle different response formats
            if hasattr(resp, 'embedding'):
                return resp.embedding
            elif isinstance(resp, dict):
                emb = resp.get("embedding") or resp.get("data") or resp.get("embeddings")
                if isinstance(emb, dict) and "values" in emb:
                    return emb["values"]
                elif isinstance(emb, list):
                    return emb
            elif isinstance(resp, list):
                return resp
            
            # If all else fails, try to get the embedding attribute
            return list(resp.embedding) if hasattr(resp, 'embedding') else [0.0] * 768
            
        except Exception as e:
            _log(f"[Gemini] embedding error: {e}")
            # safe fallback: zero vector of length 768
            return [0.0] * 768

    def generate_response(self, prompt: str, context: str = "") -> str:
        """
        Generate an answer given prompt and context (context appended).
        """
        try:
            full_prompt = f"Context:\n{context}\n\nUser Question:\n{prompt}\n\nAnswer concisely and helpfully."
            response = self.model.generate_content(full_prompt)
            # response.text is the typical attribute
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            _log(f"[Gemini] generate_response error: {e}")
            return f"I apologize, but I encountered an error processing your request: {e}"

# -------------------------
# PDF Processing and Menu Parsing
# -------------------------
class MenuParser:
    def __init__(self, gemini_service: GeminiRAGService):
        self.gemini_service = gemini_service
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with better formatting preservation."""
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Get text with better formatting
                text = page.get_text("text")
                # Clean up common PDF artifacts
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
                text_parts.append(text.strip())
            
            doc.close()
            full_text = "\n\n".join(text_parts)
            _log(f"Extracted {len(full_text)} characters from PDF")
            return full_text
            
        except Exception as e:
            _log(f"Failed to read PDF: {e}")
            return ""
    
    def smart_chunk_text(self, text: str) -> List[str]:
        """Intelligently chunk text into menu items."""
        # Split by common menu separators
        chunks = []
        
        # Try to split by price patterns (₹, $, etc.)
        price_pattern = r'[₹$]\s*\d+'
        price_splits = re.split(price_pattern, text)
        
        if len(price_splits) > 1:
            # Found price-based splits
            for i, chunk in enumerate(price_splits[1:], 1):  # Skip first empty chunk
                if len(chunk.strip()) > 10:
                    chunks.append(chunk.strip())
        else:
            # Fallback: split by double newlines or common separators
            separators = ['\n\n', '•', '▪', '▫', '○', '●']
            for sep in separators:
                if sep in text:
                    chunks = [c.strip() for c in text.split(sep) if len(c.strip()) > 10]
                    break
            
            if not chunks:
                # Last resort: split by single newlines
                lines = text.split('\n')
                current_chunk = []
                for line in lines:
                    line = line.strip()
                    if line:
                        current_chunk.append(line)
                    elif current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
        
        # Filter and clean chunks
        cleaned_chunks = []
        for chunk in chunks:
            if len(chunk) > 20 and len(chunk) < 1000:  # Reasonable size
                # Remove common PDF artifacts
                chunk = re.sub(r'^\s*[0-9]+\s*', '', chunk)  # Remove page numbers
                chunk = re.sub(r'\s+', ' ', chunk)  # Normalize whitespace
                cleaned_chunks.append(chunk.strip())
        
        _log(f"Created {len(cleaned_chunks)} chunks from text")
        return cleaned_chunks
    
    def parse_menu_items(self, text: str) -> List[MenuItem]:
        """Parse text into structured menu items."""
        chunks = self.smart_chunk_text(text)
        menu_items = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Try to extract structured info using LLM
                parse_prompt = f"""
Parse this menu item text into a JSON object with these fields:
- name: dish name (string)
- description: description (string, can be empty)
- category: appetizer/main course/dessert/beverage/etc (string)
- price: price as number (float, extract numeric value only)
- ingredients: list of main ingredients (array of strings)
- dietary_tags: vegetarian/vegan/gluten-free/etc (array of strings)
- spice_level: mild/medium/spicy/unknown (string)

Menu text: "{chunk}"

Return ONLY valid JSON, no other text:"""

                try:
                    response = self.gemini_service.generate_response(parse_prompt)
                    response = response.strip()
                    
                    # Clean response
                    if "```json" in response:
                        response = response.split("```json")[1].split("```")[0].strip()
                    elif "```" in response:
                        parts = response.split("```")
                        if len(parts) >= 3:
                            response = parts[1].strip()
                    
                    item_data = json.loads(response)
                    
                    # Create MenuItem
                    menu_item = MenuItem(
                        id=str(i + 1),
                        name=item_data.get('name', chunk[:50]),
                        description=item_data.get('description', ''),
                        category=item_data.get('category', 'Uncategorized'),
                        price=float(item_data.get('price', 0.0) or 0.0),
                        ingredients=item_data.get('ingredients', []) or [],
                        dietary_tags=item_data.get('dietary_tags', []) or [],
                        spice_level=item_data.get('spice_level', 'unknown') or 'unknown'
                    )
                    menu_items.append(menu_item)
                    
                except Exception as e:
                    _log(f"LLM parsing failed for chunk {i}: {e}")
                    # Fallback: create basic item
                    menu_item = MenuItem(
                        id=str(i + 1),
                        name=chunk[:50],
                        description=chunk,
                        category="Uncategorized",
                        price=0.0,
                        ingredients=[],
                        dietary_tags=[],
                        spice_level="unknown"
                    )
                    menu_items.append(menu_item)
                    
            except Exception as e:
                _log(f"Error processing chunk {i}: {e}")
                continue
        
        _log(f"Successfully parsed {len(menu_items)} menu items")
        return menu_items

# -------------------------
# Restaurant RAG System
# -------------------------
class RestaurantRAGSystem:
    def __init__(self, api_key: str, use_pinecone: bool = False, pinecone_conf: Dict = None):
        self.gemini_service = GeminiRAGService(api_key)
        self.menu_parser = MenuParser(self.gemini_service)
        self.use_pinecone = bool(use_pinecone and PINECONE_AVAILABLE)
        self._pinecone_conf = pinecone_conf or {}

        if self.use_pinecone:
            _log("Initializing Pinecone vector store...")
            try:
                # Build PineconeVectorStore - relies on external file vector_store_pinecone.py
                self.menu_vector_store = PineconeVectorStore(
                    embed_fn=self.gemini_service.generate_embedding,
                    index_name=self._pinecone_conf.get("menu_index", "chatbot2"),
                    api_key=self._pinecone_conf.get("api_key", os.getenv("PINECONE_API_KEY", "")),
                    cloud=self._pinecone_conf.get("cloud", os.getenv("PINECONE_CLOUD", "aws")),
                    region=self._pinecone_conf.get("region", os.getenv("PINECONE_REGION", "us-east-1")),
                    log_sink=st.session_state._logs
                )
                self.feedback_vector_store = PineconeVectorStore(
                    embed_fn=self.gemini_service.generate_embedding,
                    index_name=self._pinecone_conf.get("feedback_index", "chatbot2-feedback"),
                    api_key=self._pinecone_conf.get("api_key", os.getenv("PINECONE_API_KEY", "")),
                    cloud=self._pinecone_conf.get("cloud", os.getenv("PINECONE_CLOUD", "aws")),
                    region=self._pinecone_conf.get("region", os.getenv("PINECONE_REGION", "us-east-1")),
                    log_sink=st.session_state._logs
                )
            except Exception as e:
                _log(f"Failed to initialize Pinecone, falling back to SimpleVectorStore: {e}")
                self.use_pinecone = False
                self.menu_vector_store = SimpleVectorStore(self.gemini_service)
                self.feedback_vector_store = SimpleVectorStore(self.gemini_service)
        else:
            _log("Using in-memory vector store (SimpleVectorStore).")
            self.menu_vector_store = SimpleVectorStore(self.gemini_service)
            self.feedback_vector_store = SimpleVectorStore(self.gemini_service)

        self.order_history = []

    def initialize_sample_data(self, pdf_path: str = "taj-amer-food-menu.pdf"):
        if not os.path.exists(pdf_path):
            _log(f"PDF not found at {pdf_path}")
            return

        # Extract and parse menu items
        text = self.menu_parser.extract_text_from_pdf(pdf_path)
        if not text.strip():
            _log("PDF appears to be empty or unreadable")
            return

        menu_items = self.menu_parser.parse_menu_items(text)
        if not menu_items:
            _log("No menu items could be parsed from PDF")
            return

        # Prepare documents and metadata for vector store
        docs = []
        meta = []
        
        for item in menu_items:
            # Create rich text representation for better search
            text = f"{item.name} - {item.description} - Category: {item.category} - Price: ₹{item.price} - Ingredients: {', '.join(item.ingredients)} - Dietary: {', '.join(item.dietary_tags)} - Spice Level: {item.spice_level}"
            docs.append(text)
            meta.append({
                "id": item.id,
                "name": item.name,
                "description": item.description,
                "price": item.price,
                "category": item.category,
                "ingredients": item.ingredients,
                "dietary_tags": item.dietary_tags,
                "spice_level": item.spice_level
            })

        _log(f"Ingesting {len(docs)} menu items into vector store...")
        self.menu_vector_store.add_documents(docs, meta)
        _log("Menu ingestion complete.")

        # Seed feedback sample into feedback store (only when using SimpleVectorStore)
        if isinstance(self.feedback_vector_store, SimpleVectorStore) and len(self.feedback_vector_store.documents) == 0:
            sample_fb = [
                CustomerFeedback("John Doe", "Loved the lotus stem! A bit too salty though.", 4, "2024-01-15"),
                CustomerFeedback("Jane Smith", "Prawns were perfectly cooked. Excellent service!", 5, "2024-01-16"),
                CustomerFeedback("Mike Johnson", "The biryani was amazing but took too long to arrive.", 3, "2024-01-17"),
                CustomerFeedback("Sarah Wilson", "The butter chicken was creamy and delicious. Will definitely order again!", 5, "2024-01-18"),
                CustomerFeedback("David Brown", "The naan bread was fresh and the curry had perfect spice level.", 4, "2024-01-19"),
            ]
            fb_docs, fb_meta = [], []
            for f in sample_fb:
                fb_docs.append(f"Customer: {f.customer_name} - Rating: {f.rating}/5 - Feedback: {f.feedback}")
                fb_meta.append({
                    "customer_name": f.customer_name, 
                    "rating": f.rating, 
                    "feedback": f.feedback, 
                    "date": f.date
                })
            self.feedback_vector_store.add_documents(fb_docs, fb_meta)
            _log("Seeded feedback examples.")

    def process_customer_query(self, query: str):
        """Process customer queries with improved context and prompts."""
        # Search for relevant menu items and feedback
        menu_results = self.menu_vector_store.search(query, top_k=5)
        feedback_results = self.feedback_vector_store.search(query, top_k=3)
        
        # Build comprehensive context
        context_parts = []
        
        if menu_results:
            menu_context = "Available Menu Items:\n" + "\n".join([
                f"• {m['metadata'].get('name','Unknown')} (₹{m['metadata'].get('price','N/A')})\n"
                f"  Category: {m['metadata'].get('category','')}\n"
                f"  Description: {m['metadata'].get('description','')}\n"
                f"  Ingredients: {', '.join(m['metadata'].get('ingredients',[]))}\n"
                f"  Dietary: {', '.join(m['metadata'].get('dietary_tags',[]))}\n"
                f"  Spice Level: {m['metadata'].get('spice_level','unknown')}"
                for m in menu_results
            ])
            context_parts.append(menu_context)
        
        if feedback_results:
            feedback_context = "Customer Feedback:\n" + "\n".join([
                f"• {f['metadata'].get('customer_name','Anonymous')} (Rating: {f['metadata'].get('rating','N/A')}/5): {f['metadata'].get('feedback','')}"
                for f in feedback_results
            ])
            context_parts.append(feedback_context)
        
        context = "\n\n".join(context_parts)
        
        # Use improved prompt for better responses
        system_prompt = """You are a helpful restaurant assistant. Based on the menu items and customer feedback provided, answer the customer's question accurately and helpfully. 

If the customer is asking about:
- Menu items: Provide details about the dishes, prices, ingredients, and dietary information
- Recommendations: Suggest items based on their preferences and popular choices
- Customer feedback: Share relevant customer experiences and ratings
- General questions: Provide helpful information about the restaurant

Be friendly, informative, and specific. If you don't have enough information, say so politely."""
        
        full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nCustomer Question: {query}\n\nAssistant:"
        
        answer = self.gemini_service.generate_response(full_prompt)
        
        return {
            "response": answer,
            "menu_context": [m.get("metadata", {}) for m in menu_results],
            "feedback_context": [f.get("metadata", {}) for f in feedback_results]
        }

    def get_menu_recommendations(self, customer_query: str):
        """Get personalized menu recommendations."""
        similar = self.menu_vector_store.search(customer_query, top_k=8)
        if not similar:
            return {"response": "I couldn't find any menu items matching your request. Please try a different search term.", "recommendations": []}
        
        # Build detailed context for recommendations
        context = "Menu Items for Recommendations:\n" + "\n".join([
            f"• {s['metadata'].get('name','Unknown')} (₹{s['metadata'].get('price','N/A')})\n"
            f"  Category: {s['metadata'].get('category','')}\n"
            f"  Description: {s['metadata'].get('description','')}\n"
            f"  Ingredients: {', '.join(s['metadata'].get('ingredients',[]))}\n"
            f"  Dietary: {', '.join(s['metadata'].get('dietary_tags',[]))}\n"
            f"  Spice Level: {s['metadata'].get('spice_level','unknown')}"
            for s in similar
        ])
        
        recommendation_prompt = f"""As a restaurant expert, analyze the customer's request and recommend the best menu items from the available options.

Customer Request: "{customer_query}"

Consider:
- Dietary preferences (vegetarian, vegan, etc.)
- Spice level preferences
- Price range
- Category preferences (appetizer, main course, etc.)
- Popular items and customer favorites

Provide 3-5 specific recommendations with brief explanations for each choice."""

        response = self.gemini_service.generate_response(recommendation_prompt, context)
        
        return {
            "response": response, 
            "recommendations": [s['metadata'] for s in similar]
        }

# -------------------------
# Utilities
# -------------------------
def _log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] {msg}"
    if "_logs" in st.session_state:
        st.session_state._logs.append(entry)
    else:
        print(entry)

# -------------------------
# Streamlit UI (minimal chatbot)
# -------------------------
st.set_page_config(page_title="Restaurant RAG — Chatbot", page_icon="🍜", layout="wide")

# Initialize session state
if "_logs" not in st.session_state:
    st.session_state._logs = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # each item: {"role","text","sources"}

if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Sidebar
with st.sidebar:
    st.header("🔧 Setup")
    api_key = st.text_input("GEMINI_API_KEY", value=DEFAULT_GEMINI_API_KEY, type="password")
    
    # Only show Pinecone options if available
    if PINECONE_AVAILABLE:
        use_pinecone = st.checkbox("Use Pinecone", value=False)
        pc_api = st.text_input("PINECONE_API_KEY", value=os.getenv("PINECONE_API_KEY", ""), type="password", disabled=not use_pinecone)
        pc_cloud = st.text_input("PINECONE_CLOUD", value=os.getenv("PINECONE_CLOUD", "aws"), disabled=not use_pinecone)
        pc_region = st.text_input("PINECONE_REGION", value=os.getenv("PINECONE_REGION", "us-east-1"), disabled=not use_pinecone)
        pc_menu_index = st.text_input("Menu Index", value="chatbot2", disabled=not use_pinecone)
        pc_fb_index = st.text_input("Feedback Index", value="chatbot2-feedback", disabled=not use_pinecone)
    else:
        st.info("Pinecone wrapper not available. Using in-memory vector store.")
        use_pinecone = False
        pc_api = pc_cloud = pc_region = pc_menu_index = pc_fb_index = ""

    st.markdown("---")
    st.header("📄 Menu PDF")
    uploaded_pdf = st.file_uploader("Upload your menu PDF", type=["pdf"])
    
    if uploaded_pdf:
        save_path = os.path.join(os.getcwd(), "taj-amer-food-menu.pdf")
        with open(save_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        st.success(f"✅ PDF saved!")
        _log(f"Uploaded menu PDF saved to: {save_path}")

    st.markdown("---")
    if st.button("🚀 Initialize System", type="primary"):
        if not api_key:
            st.error("❌ Please provide GEMINI_API_KEY")
        elif not uploaded_pdf and not os.path.exists("taj-amer-food-menu.pdf"):
            st.error("❌ Please upload menu PDF first")
        else:
            conf = None
            if use_pinecone and PINECONE_AVAILABLE:
                conf = {
                    "api_key": pc_api,
                    "cloud": pc_cloud,
                    "region": pc_region,
                    "menu_index": pc_menu_index,
                    "feedback_index": pc_fb_index
                }
            try:
                with st.spinner("Initializing RAG system..."):
                    st.session_state.rag = RestaurantRAGSystem(
                        api_key, 
                        use_pinecone=use_pinecone, 
                        pinecone_conf=conf
                    )
                    st.session_state.rag.initialize_sample_data()
                
                st.success("✅ RAG system initialized successfully!")
                st.session_state.initialized = True
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Failed to initialize RAG system: {e}")
                _log(f"Init error: {e}")

    if st.button("🧹 Clear Logs"):
        st.session_state._logs = []
        st.success("Logs cleared")

# Main UI
if not st.session_state.get("initialized"):
    st.title("🍽️ Restaurant RAG Chatbot")
    st.info("👈 Initialize the system from the sidebar first (upload PDF → Initialize System).")
    
    with st.expander("📋 Recent logs", expanded=False):
        if st.session_state._logs:
            for ln in st.session_state._logs[-20:]:
                st.text(ln)
        else:
            st.text("No logs yet...")
    st.stop()

# Get RAG system from session state
rag: RestaurantRAGSystem = st.session_state.rag

st.title("🍽️ Restaurant Chatbot")
st.write("Ask about the menu, get recommendations, or inquire about customer feedback!")

col1, col2 = st.columns([3, 1])

with col1:
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for turn in st.session_state.chat_history:
            if turn["role"] == "user":
                st.markdown(f"**🧑‍💼 You:** {turn['text']}")
            else:
                st.markdown(f"**🤖 Assistant:** {turn['text']}")
                if turn.get("sources"):
                    with st.expander("📚 Sources used"):
                        for s in turn["sources"]:
                            try:
                                name = s.get('name', s.get('id', 'Unknown'))
                                category = s.get('category', 'N/A')
                                price = s.get('price', 'N/A')
                                st.write(f"• **{name}** | Category: {category} | Price: ₹{price}")
                            except Exception:
                                st.write(f"• {s}")

    # Input section
    st.markdown("---")
    user_input = st.text_input(
        "💬 Your message:", 
        key="user_input", 
        placeholder="e.g., 'Recommend a spicy vegetarian starter' or 'What do customers say about the biryani?'"
    )
    
    col_b1, col_b2, col_b3 = st.columns([1, 1, 1])
    
    if col_b1.button("📤 Send", type="primary"):
        q = user_input.strip()
        if q:
            st.session_state.chat_history.append({"role": "user", "text": q})
            with st.spinner("🤔 Processing your query..."):
                ans = rag.process_customer_query(q)
            st.session_state.chat_history.append({
                "role": "assistant", 
                "text": ans.get("response", ""), 
                "sources": ans.get("menu_context", [])
            })
            st.rerun()
    
    if col_b2.button("🎯 Get Recommendations"):
        q = user_input.strip() or "Recommend me something popular"
        st.session_state.chat_history.append({"role": "user", "text": q})
        with st.spinner("🔍 Finding recommendations..."):
            res = rag.get_menu_recommendations(q)
        st.session_state.chat_history.append({
            "role": "assistant", 
            "text": res.get("response", ""), 
            "sources": res.get("recommendations", [])
        })
        st.rerun()
    
    if col_b3.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared!")
        st.rerun()

with col2:
    st.subheader("ℹ️ Session Info")
    st.write(f"**Vector Store:** {'🌲 Pinecone' if rag.use_pinecone else '💾 In-Memory'}")
    
    st.markdown("---")
    st.subheader("🚀 Quick Actions")
    
    if st.button("🌶️ Spicy veg options", key="quick1"):
        query = "Show me spicy vegetarian dishes"
        st.session_state.chat_history.append({"role": "user", "text": query})
        with st.spinner("Processing..."):
            ans = rag.process_customer_query(query)
        st.session_state.chat_history.append({
            "role": "assistant", 
            "text": ans.get("response", ""), 
            "sources": ans.get("menu_context", [])
        })
        st.rerun()
    
    if st.button("🥣 Light soups", key="quick2"):
        query = "Show me light and healthy soup options"
        st.session_state.chat_history.append({"role": "user", "text": query})
        with st.spinner("Processing..."):
            ans = rag.process_customer_query(query)
        st.session_state.chat_history.append({
            "role": "assistant", 
            "text": ans.get("response", ""), 
            "sources": ans.get("menu_context", [])
        })
        st.rerun()
    
    if st.button("⭐ Customer favorites", key="quick3"):
        query = "What do customers love most about this restaurant?"
        st.session_state.chat_history.append({"role": "user", "text": query})
        with st.spinner("Processing..."):
            ans = rag.process_customer_query(query)
        st.session_state.chat_history.append({
            "role": "assistant", 
            "text": ans.get("response", ""), 
            "sources": ans.get("feedback_context", [])
        })
        st.rerun()

    st.markdown("---")
    with st.expander("📝 System Logs", expanded=False):
        if st.session_state._logs:
            for ln in st.session_state._logs[-15:]:
                st.text(ln)
        else:
            st.text("No logs yet...")
