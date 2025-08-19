from vector_store_pinecone import PineconeVectorStore
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# =============================
# Configuration
# =============================
load_dotenv()
DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# =============================
# Domain Models
# =============================
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

@dataclass
class CustomerFeedback:
    customer_name: str
    feedback: str
    rating: int
    date: str
    order_id: Optional[str] = None

# =============================
# Simple Vector Store (fallback)
# =============================
class SimpleVectorStore:
    def __init__(self, gemini_service):
        self.gemini_service = gemini_service
        self.documents = []
        self.embeddings = []
        self.metadata = []

    def add_documents(self, documents: List[str], metadata: List[Dict]):
        for doc, meta in zip(documents, metadata):
            embedding = self.gemini_service.generate_embedding(doc)
            self.documents.append(doc)
            self.embeddings.append(embedding)
            self.metadata.append(meta)
            st.session_state._logs.append(f"[SimpleStore] Added doc: {doc[:50]}...")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.embeddings:
            return []
        query_embedding = self.gemini_service.generate_embedding(query)
        similarities = [self._cosine_similarity(query_embedding, emb) for emb in self.embeddings]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [
            {
                'document': self.documents[i],
                'metadata': self.metadata[i],
                'similarity': float(similarities[i])
            }
            for i in top_indices
        ]

    def _cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return 0.0 if denom == 0 else float(np.dot(a, b) / denom)

# =============================
# Gemini RAG Service
# =============================
class GeminiRAGService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_embedding(self, text: str) -> List[float]:
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",  # 768-dim
                content=text,
                task_type="retrieval_document"
            )
            embedding = result.get('embedding') if isinstance(result, dict) else None
            if isinstance(embedding, dict) and 'values' in embedding:
                return embedding['values']
            if isinstance(embedding, list):
                return embedding
            raise ValueError("Unexpected embedding response shape")
        except Exception as e:
            st.session_state._logs.append(f"Error generating embedding: {e}")
            return [0.0] * 768

    def generate_response(self, prompt: str, context: str = "") -> str:
        try:
            full_prompt = f"""
            Context Information:
            {context}

            User Question: {prompt}

            Based on the context, provide a helpful and accurate response.
            """
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {e}"

    def analyze_sentiment(self, text: str) -> Dict:
        try:
            prompt = f"""
            Analyze this customer feedback: "{text}" and return JSON with:
            sentiment, confidence, key_issues, suggested_actions, priority
            """
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            return json.loads(response_text)
        except Exception as e:
            return {
                "error": f"Analysis failed: {e}",
                "sentiment": "neutral",
                "priority": "medium"
            }

# =============================
# Restaurant RAG System
# =============================
import fitz  # PyMuPDF
import pinecone

# Replace the existing RestaurantRAGSystem definition with this one

class RestaurantRAGSystem:
    def __init__(self, api_key: str, use_pinecone: bool = False, pinecone_conf: Dict = None):
        """
        api_key: Gemini API key (for embeddings + LLM)
        use_pinecone: whether to use PineconeVectorStore
        pinecone_conf: dict with keys api_key, cloud, region, menu_index, feedback_index
        """
        self.gemini_service = GeminiRAGService(api_key)
        self.use_pinecone = bool(use_pinecone)
        self._pinecone_conf = pinecone_conf or {}

        # Choose store
        if self.use_pinecone:
            st.session_state._logs.append("Using Pinecone vector backend.")
            # PineconeVectorStore expects embed_fn and index_name + api_key/cloud/region
            self.menu_vector_store = PineconeVectorStore(
                embed_fn=self.gemini_service.generate_embedding,
                index_name=self._pinecone_conf.get("menu_index", "menu-index"),
                api_key=self._pinecone_conf.get("api_key", os.getenv("PINECONE_API_KEY", "")),
                cloud=self._pinecone_conf.get("cloud", os.getenv("PINECONE_CLOUD", "aws")),
                region=self._pinecone_conf.get("region", os.getenv("PINECONE_REGION", "us-east-1")),
                log_sink=st.session_state._logs
            )
            self.feedback_vector_store = PineconeVectorStore(
                embed_fn=self.gemini_service.generate_embedding,
                index_name=self._pinecone_conf.get("feedback_index", "feedback-index"),
                api_key=self._pinecone_conf.get("api_key", os.getenv("PINECONE_API_KEY", "")),
                cloud=self._pinecone_conf.get("cloud", os.getenv("PINECONE_CLOUD", "aws")),
                region=self._pinecone_conf.get("region", os.getenv("PINECONE_REGION", "us-east-1")),
                log_sink=st.session_state._logs
            )
        else:
            st.session_state._logs.append("Using in-memory SimpleVectorStore backend.")
            self.menu_vector_store = SimpleVectorStore(self.gemini_service)
            self.feedback_vector_store = SimpleVectorStore(self.gemini_service)

        # Simple order history placeholder
        self.order_history = []

    # ---------------------------
    # Initialization / ingestion
    # ---------------------------
    def initialize_sample_data(self):
        """
        Parse taj-amer-food-menu.pdf into structured MenuItem objects using Gemini,
        then ingest into the selected vector store (Pinecone or in-memory).
        """
        pdf_path = "taj-amer-food-menu.pdf"
        if not os.path.exists(pdf_path):
            st.session_state._logs.append(f"PDF not found: {pdf_path}")
            return

        # 1) Extract text from PDF (PyMuPDF)
        try:
            import fitz
        except Exception as e:
            st.session_state._logs.append("PyMuPDF (fitz) missing: pip install PyMuPDF")
            return

        try:
            doc = fitz.open(pdf_path)
            full_text = "\n\n".join([page.get_text("text") for page in doc])
            st.session_state._logs.append("Extracted text from PDF.")
        except Exception as e:
            st.session_state._logs.append(f"Failed to read PDF: {e}")
            return

        # 2) Ask Gemini to parse the raw menu into structured JSON (list of items)
        parse_prompt = f"""
You are a helpful parser bot. Convert the following restaurant menu text into a JSON array of menu items.
Each item must have these fields:
- id (string)
- name (string)
- description (string)
- category (string) — e.g., Appetizer, Mains, Noodles, Seafood, Desserts, etc.
- price (float) — numeric price in INR (if price not present, set 0.0)
- ingredients (list of strings) — best guess from description
- dietary_tags (list of strings) — possible values: "vegetarian", "vegan", "gluten-free", "non-vegetarian"
- spice_level (string) — one of "mild", "medium", "hot", "unknown"

Respond ONLY with valid JSON (an array of objects). Here is the menu text:

{full_text}
"""
        parsed_text = self.gemini_service.generate_response(parse_prompt, context="")
        # Try to locate JSON in LLM response
        parsed_text = parsed_text.strip()
        if parsed_text.startswith("```"):
            # strip code fences
            parsed_text = parsed_text.split("```")[-2].strip()

        try:
            menu_items_data = json.loads(parsed_text)
            if not isinstance(menu_items_data, list):
                raise ValueError("Parsed JSON is not a list")
            st.session_state._logs.append(f"Parsed {len(menu_items_data)} items from LLM output.")
        except Exception as e:
            st.session_state._logs.append(f"Failed to parse JSON from LLM: {e}")
            # Fallback: extremely simple parsing — split by double newlines and create items
            chunks = [c.strip() for c in full_text.split("\n\n") if len(c.strip()) > 20][:200]
            menu_items_data = []
            for i, chunk in enumerate(chunks):
                name = chunk.split("\n")[0][:60]
                menu_items_data.append({
                    "id": str(i+1),
                    "name": name,
                    "description": chunk.replace("\n", " ")[:400],
                    "category": "Uncategorized",
                    "price": 0.0,
                    "ingredients": [],
                    "dietary_tags": [],
                    "spice_level": "unknown"
                })
            st.session_state._logs.append(f"Fallback parsed {len(menu_items_data)} chunks.")

        # 3) Prepare docs + metadata and ingest
        menu_docs, menu_meta = [], []
        for idx, item in enumerate(menu_items_data):
            try:
                m_id = str(item.get("id", idx+1))
                name = item.get("name", "Unnamed Dish").strip()
                desc = item.get("description", "").strip()
                category = item.get("category", "Uncategorized")
                price = float(item.get("price", 0.0) or 0.0)
                ingredients = item.get("ingredients", []) or []
                dietary_tags = item.get("dietary_tags", []) or []
                spice = item.get("spice_level", "unknown") or "unknown"

                doc_text = (
                    f"{name} - {desc} - Category: {category} - "
                    f"Price: ₹{price} - Ingredients: {', '.join(ingredients)} - "
                    f"Dietary: {', '.join(dietary_tags)} - Spice Level: {spice}"
                )
                menu_docs.append(doc_text)
                menu_meta.append({
                    "id": m_id,
                    "name": name,
                    "price": price,
                    "category": category,
                    "dietary_tags": dietary_tags,
                    "spice_level": spice
                })
            except Exception as ex:
                st.session_state._logs.append(f"Skipping item parse error: {ex}")

        if not menu_docs:
            st.session_state._logs.append("No menu documents prepared for ingestion.")
            return

        # If using Pinecone, rely on PineconeVectorStore's index creation during its init.
        # Ingest docs
        st.session_state._logs.append(f"Ingesting {len(menu_docs)} menu items into vector store...")
        self.menu_vector_store.add_documents(menu_docs, menu_meta)
        st.session_state._logs.append("Menu ingestion complete.")

        # Optional: seed some example feedback and orders (safe to call once)
        if isinstance(self.feedback_vector_store, SimpleVectorStore) and len(self.feedback_vector_store.documents) == 0:
            sample_fb = [
                CustomerFeedback("John Doe", "Loved the lotus stem! A bit too salty.", 4, "2024-01-15"),
                CustomerFeedback("Jane Smith", "Prawns were perfectly cooked.", 5, "2024-01-16"),
            ]
            fb_docs, fb_meta = [], []
            for f in sample_fb:
                fb_docs.append(f"Customer: {f.customer_name} - Rating: {f.rating}/5 - Feedback: {f.feedback}")
                fb_meta.append({"customer_name": f.customer_name, "rating": f.rating, "feedback": f.feedback, "date": f.date})
            self.feedback_vector_store.add_documents(fb_docs, fb_meta)
            st.session_state._logs.append("Seeded sample feedback.")

    # ---------------------------
    # Features (unchanged)
    # ---------------------------
    def get_menu_recommendations(self, customer_query: str) -> Dict:
        similar = self.menu_vector_store.search(customer_query, top_k=5)
        if not similar:
            return {"response": "No matching items found.", "recommendations": []}
        context = "Available Menu Items:\n"
        recs = []
        for item in similar:
            m = item['metadata']
            # ensure keys exist safely
            name = m.get('name') if isinstance(m, dict) else None
            price = m.get('price') if isinstance(m, dict) else None
            category = m.get('category') if isinstance(m, dict) else None
            dietary = m.get('dietary_tags', []) if isinstance(m, dict) else []
            context += f"- {name} (₹{price}) - Category: {category}, Dietary: {', '.join(dietary)}\n"
            recs.append(m)
        prompt = f"Based on the customer's preferences: '{customer_query}', recommend the best menu items and explain why."
        response = self.gemini_service.generate_response(prompt, context)
        return {"response": response, "recommendations": recs, "similarity_scores": [x.get('similarity', 0) for x in similar]}

    def process_customer_query(self, query: str) -> Dict:
        menu_results = self.menu_vector_store.search(query, top_k=3)
        feedback_results = self.feedback_vector_store.search(query, top_k=3)
        context_parts = []
        if menu_results:
            context_parts.append("Relevant Menu Information:")
            for it in menu_results:
                context_parts.append(f"- {it.get('document','')}")
        if feedback_results:
            context_parts.append("Customer Feedback:")
            for it in feedback_results:
                context_parts.append(f"- {it.get('document','')}")
        context = "\n".join(context_parts)
        response = self.gemini_service.generate_response(query, context)
        return {
            "response": response,
            "menu_context": [it.get('metadata', {}) for it in menu_results],
            "feedback_context": [it.get('metadata', {}) for it in feedback_results]
        }

    def analyze_customer_feedback(self, feedback: str, customer_name: str, rating: int) -> Dict:
        analysis = self.gemini_service.analyze_sentiment(feedback)
        fb_obj = CustomerFeedback(customer_name, feedback, rating, datetime.now().strftime("%Y-%m-%d"))
        doc_text = f"Customer: {customer_name} - Rating: {rating}/5 - Feedback: {feedback}"
        self.feedback_vector_store.add_documents([doc_text], [{
            'customer_name': customer_name,
            'rating': rating,
            'feedback': feedback,
            'date': fb_obj.date
        }])
        similar = self.feedback_vector_store.search(feedback, top_k=3)
        return {
            "analysis": analysis,
            "similar_feedback": [x.get('metadata', {}) for x in similar if x.get('metadata', {}).get('customer_name') != customer_name]
        }

    def generate_sales_report(self) -> Dict:
        if not self.order_history:
            return {"error": "No order history available"}
        df = pd.DataFrame(self.order_history)
        total_revenue = float(df['total'].sum())
        total_orders = int(len(df))
        average_order_value = float(df['total'].mean())
        all_items = []
        for order in self.order_history:
            all_items.extend(order.get('items', []))
        item_counts = pd.Series(all_items).value_counts()
        report = {
            "total_revenue": total_revenue,
            "total_orders": total_orders,
            "average_order_value": average_order_value,
            "popular_items": {k: int(v) for k, v in item_counts.head(5).to_dict().items()}
        }
        context = f"Sales Data: {json.dumps(report, indent=2)}"
        prompt = "Analyze this restaurant sales data and provide actionable business insights and recommendations."
        insights = self.gemini_service.generate_response(prompt, context)
        return {"metrics": report, "insights": insights}

    def calculate_loyalty_reward(self, customer_id: str) -> Dict:
        customer_orders = [o for o in self.order_history if o.get('customer_id') == customer_id]
        if not customer_orders:
            return {"error": "Customer not found"}
        total_spent = float(sum(o.get('total', 0.0) for o in customer_orders))
        order_count = int(len(customer_orders))
        base_points = int(total_spent)
        bonus_points = order_count * 10
        total_points = base_points + bonus_points
        if total_points >= 100:
            tier, discount = "Gold", 0.15
        elif total_points >= 50:
            tier, discount = "Silver", 0.10
        else:
            tier, discount = "Bronze", 0.05
        return {
            "customer_id": customer_id,
            "total_points": int(total_points),
            "tier": tier,
            "discount_percentage": discount * 100,
            "total_spent": total_spent,
            "order_count": order_count,
            "next_tier_points": max(0, (100 if tier == "Bronze" else 200) - total_points)
        }

#  =========
# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Restaurant RAG System", page_icon="🍽️", layout="wide")

if "_logs" not in st.session_state:
    st.session_state._logs = []

# Sidebar — API Key and Vector DB settings
st.sidebar.header("🔐 API Settings")
api_key = st.sidebar.text_input("GEMINI_API_KEY", value=DEFAULT_GEMINI_API_KEY, type="password")

st.sidebar.header("🗄️ Vector Backend")
use_pinecone = st.sidebar.toggle("Use Pinecone", value=False)
pc_api = st.sidebar.text_input("PINECONE_API_KEY", value=os.getenv("PINECONE_API_KEY", ""), type="password", disabled=not use_pinecone)
pc_cloud = st.sidebar.text_input("PINECONE_CLOUD", value=os.getenv("PINECONE_CLOUD", "aws"), disabled=not use_pinecone)
pc_region = st.sidebar.text_input("PINECONE_REGION", value=os.getenv("PINECONE_REGION", "us-east-1"), disabled=not use_pinecone)
pc_menu_index = st.sidebar.text_input("Menu Index", value="chatbot2", disabled=not use_pinecone)
pc_fb_index = st.sidebar.text_input("Feedback Index", value="chatbot2", disabled=not use_pinecone)


# PDF uploader for menu ingestion
st.sidebar.header("📄 Menu PDF Upload")
uploaded_pdf = st.sidebar.file_uploader("Upload Taj Amer Menu PDF", type=["pdf"])

pdf_save_path = None
if uploaded_pdf:
    pdf_save_path = os.path.join(os.getcwd(), "taj-amer-food-menu.pdf")
    with open(pdf_save_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.sidebar.success(f"PDF saved as {pdf_save_path}")
    st.session_state._logs.append(f"Uploaded menu PDF saved to: {pdf_save_path}")

# Initialize system button
if st.sidebar.button("Initialize System", type="primary"):
    if not api_key:
        st.sidebar.error("Please provide GEMINI_API_KEY")
    elif not uploaded_pdf and not os.path.exists("taj-amer-food-menu.pdf"):
        st.sidebar.error("Please upload the menu PDF first")
    else:
        conf = None
        if use_pinecone:
            conf = {
                "api_key": pc_api,
                "cloud": pc_cloud,
                "region": pc_region,
                "menu_index": pc_menu_index,
                "feedback_index": pc_fb_index
            }
        st.session_state.rag = RestaurantRAGSystem(api_key, use_pinecone=use_pinecone, pinecone_conf=conf)

        with st.spinner("Loading menu data from PDF..."):
            st.session_state.rag.initialize_sample_data()

        st.session_state.initialized = True
        st.sidebar.success("Initialized!")

# Stop if not initialized
if not st.session_state.get("initialized"):
    st.info("Initialize system first")
    st.stop()

# Main app logic
rag: RestaurantRAGSystem = st.session_state.rag
col1, col2 = st.columns(2)

with col1:
    st.subheader("Menu Recommendations")
    pref = st.text_input("Customer preference", "I want something spicy and vegetarian")
    if st.button("Get Recommendations"):
        rec = rag.get_menu_recommendations(pref)
        st.write(rec["response"])
        if rec["recommendations"]:
            st.table(pd.DataFrame(rec["recommendations"]))

with col2:
    st.subheader("Customer Query")
    q = st.text_input("Ask about menu or feedback", "What do other customers think about your pizza?")
    if st.button("Process Query"):
        ans = rag.process_customer_query(q)
        st.write(ans["response"])
        with st.expander("Context used"):
            st.write({"menu": ans["menu_context"], "feedback": ans["feedback_context"]})
