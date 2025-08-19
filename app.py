# Restaurant RAG Application Demo — Streamlit Version
# - Runs ingestion only once per session
# - Each step triggered explicitly via buttons (no auto reruns of the whole pipeline)
# - Reuses your existing classes with minimal changes

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
# Vector Store (Simplified)
# =============================
class SimpleVectorStore:
    """Simplified vector store using cosine similarity"""

    def __init__(self, gemini_service):
        self.gemini_service = gemini_service
        self.documents = []
        self.embeddings = []
        self.metadata = []

    def add_documents(self, documents: List[str], metadata: List[Dict]):
        for doc, meta in zip(documents, metadata):
            try:
                embedding = self.gemini_service.generate_embedding(doc)
                self.documents.append(doc)
                self.embeddings.append(embedding)
                self.metadata.append(meta)
                # Streamlit-friendly logging
                st.session_state._logs.append(f"Added document: {doc[:50]}...")
            except Exception as e:
                st.session_state._logs.append(f"Error adding document: {e}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.embeddings:
            return []
        try:
            query_embedding = self.gemini_service.generate_embedding(query)
            similarities = []
            for embedding in self.embeddings:
                similarity = self._cosine_similarity(query_embedding, embedding)
                similarities.append(similarity)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            results = []
            for idx in top_indices:
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'similarity': float(similarities[idx])
                })
            return results
        except Exception as e:
            st.session_state._logs.append(f"Error searching: {e}")
            return []

    def _cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

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
            err = str(e)
            if "403" in err and "has not been used" in err:
                st.session_state._logs.append(
                    "Error generating embedding: Enable Generative Language API in Google Cloud Console."
                )
            else:
                st.session_state._logs.append(f"Error generating embedding: {e}")
            return [0.0] * 768

    def generate_response(self, prompt: str, context: str = "") -> str:
        try:
            full_prompt = f"""
            Context Information:
            {context}

            User Question: {prompt}

            Based on the context information above, provide a helpful and accurate response.
            If the context doesn't contain relevant information, say so and provide general guidance.
            """
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            err = str(e)
            if "403" in err and "has not been used" in err:
                return ("Error generating response: Generative Language API not enabled for this project. "
                        "Enable it here: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com")
            return f"Error generating response: {e}"

    def analyze_sentiment(self, text: str) -> Dict:
        try:
            prompt = f"""
            Analyze the sentiment of this customer feedback: "{text}"

            Provide your analysis in this exact JSON format:
            {{
                "sentiment": "positive/negative/neutral",
                "confidence": 0.95,
                "key_issues": ["list", "of", "issues"],
                "suggested_actions": ["list", "of", "actions"],
                "priority": "high/medium/low"
            }}
            """
            response = self.model.generate_content(prompt)
            try:
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3]
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3]
                return json.loads(response_text)
            except json.JSONDecodeError:
                return {
                    "sentiment": "neutral",
                    "confidence": 0.5,
                    "key_issues": ["Unable to parse response"],
                    "suggested_actions": ["Manual review required"],
                    "priority": "medium"
                }
        except Exception as e:
            return {
                "error": f"Analysis failed: {e}",
                "sentiment": "neutral",
                "priority": "medium"
            }

# =============================
# Restaurant RAG System
# =============================
class RestaurantRAGSystem:
    def __init__(self, api_key: str):
        self.gemini_service = GeminiRAGService(api_key)
        self.menu_vector_store = SimpleVectorStore(self.gemini_service)
        self.feedback_vector_store = SimpleVectorStore(self.gemini_service)
        self.order_history = []

    def initialize_sample_data(self):
        """Populate stores and orders. Safe to call once per session."""
        # Sample menu
        menu_items = [
            MenuItem("1", "Spicy Chicken Tikka", "Tender chicken marinated in aromatic spices and grilled to perfection",
                     "Main Course", 15.99, ["chicken", "yogurt", "spices"], ["gluten-free"], "hot"),
            MenuItem("2", "Vegetarian Buddha Bowl", "Fresh mixed vegetables with quinoa and tahini dressing",
                     "Healthy", 12.99, ["quinoa", "vegetables", "tahini"], ["vegetarian", "vegan"], "mild"),
            MenuItem("3", "Classic Margherita Pizza", "Traditional pizza with fresh mozzarella and basil",
                     "Pizza", 13.99, ["dough", "mozzarella", "tomatoes", "basil"], ["vegetarian"], "mild"),
            MenuItem("4", "Beef Burger Deluxe", "Juicy beef patty with premium toppings and fries",
                     "American", 16.99, ["beef", "lettuce", "tomato", "cheese"], [], "medium"),
            MenuItem("5", "Vegan Curry", "Rich coconut curry with mixed vegetables and rice",
                     "Indian", 14.99, ["coconut", "vegetables", "rice", "spices"], ["vegan", "gluten-free"], "hot")
        ]
        menu_docs, menu_meta = [], []
        for item in menu_items:
            doc = (
                f"{item.name} - {item.description} - Category: {item.category} - "
                f"Price: ${item.price} - Ingredients: {', '.join(item.ingredients)} - "
                f"Dietary: {', '.join(item.dietary_tags)} - Spice Level: {item.spice_level}"
            )
            menu_docs.append(doc)
            menu_meta.append({
                'id': item.id,
                'name': item.name,
                'price': item.price,
                'category': item.category,
                'dietary_tags': item.dietary_tags,
                'spice_level': item.spice_level
            })
        self.menu_vector_store.add_documents(menu_docs, menu_meta)

        # Sample feedback
        feedback_data = [
            CustomerFeedback("John Doe", "The chicken tikka was absolutely delicious! Perfect spice level.", 5, "2024-01-15"),
            CustomerFeedback("Jane Smith", "Buddha bowl was fresh but could use more dressing", 4, "2024-01-16"),
            CustomerFeedback("Mike Johnson", "Pizza was okay but took too long to arrive", 3, "2024-01-17"),
            CustomerFeedback("Sarah Wilson", "Loved the vegan curry! Great flavors and perfect texture", 5, "2024-01-18"),
            CustomerFeedback("Tom Brown", "Burger was overcooked and fries were cold", 2, "2024-01-19"),
        ]
        fb_docs, fb_meta = [], []
        for f in feedback_data:
            doc = f"Customer: {f.customer_name} - Rating: {f.rating}/5 - Feedback: {f.feedback}"
            fb_docs.append(doc)
            fb_meta.append({
                'customer_name': f.customer_name,
                'rating': f.rating,
                'feedback': f.feedback,
                'date': f.date
            })
        self.feedback_vector_store.add_documents(fb_docs, fb_meta)

        # Orders
        self.order_history = [
            {'date': '2024-01-15', 'time': '12:30', 'customer_id': 'C001', 'items': ['Spicy Chicken Tikka'], 'total': 15.99},
            {'date': '2024-01-16', 'time': '13:45', 'customer_id': 'C002', 'items': ['Vegetarian Buddha Bowl'], 'total': 12.99},
            {'date': '2024-01-17', 'time': '19:20', 'customer_id': 'C003', 'items': ['Classic Margherita Pizza'], 'total': 13.99},
            {'date': '2024-01-18', 'time': '18:30', 'customer_id': 'C004', 'items': ['Vegan Curry'], 'total': 14.99},
            {'date': '2024-01-19', 'time': '20:15', 'customer_id': 'C005', 'items': ['Beef Burger Deluxe'], 'total': 16.99},
        ]

    # ====== Features ======
    def get_menu_recommendations(self, customer_query: str) -> Dict:
        similar = self.menu_vector_store.search(customer_query, top_k=3)
        if not similar:
            return {"response": "No matching items found.", "recommendations": []}
        context = "Available Menu Items:\n"
        recs = []
        for item in similar:
            m = item['metadata']
            context += f"- {m['name']} (${m['price']}) - Category: {m['category']}, Dietary: {', '.join(m['dietary_tags'])}\n"
            recs.append(m)
        prompt = (
            f"Based on the customer's preferences: '{customer_query}', recommend the best menu items and explain why."
        )
        response = self.gemini_service.generate_response(prompt, context)
        return {"response": response, "recommendations": recs, "similarity_scores": [x['similarity'] for x in similar]}

    def process_customer_query(self, query: str) -> Dict:
        menu_results = self.menu_vector_store.search(query, top_k=3)
        feedback_results = self.feedback_vector_store.search(query, top_k=3)
        context = ""
        if menu_results:
            context += "Relevant Menu Information:\n"
            for it in menu_results:
                context += f"- {it['document']}\n"
        if feedback_results:
            context += "\nCustomer Feedback:\n"
            for it in feedback_results:
                context += f"- {it['document']}\n"
        response = self.gemini_service.generate_response(query, context)
        return {
            "response": response,
            "menu_context": [it['metadata'] for it in menu_results],
            "feedback_context": [it['metadata'] for it in feedback_results]
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
            "similar_feedback": [x['metadata'] for x in similar if x['metadata']['customer_name'] != customer_name]
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
            all_items.extend(order['items'])
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
        customer_orders = [o for o in self.order_history if o['customer_id'] == customer_id]
        if not customer_orders:
            return {"error": "Customer not found"}
        total_spent = float(sum(o['total'] for o in customer_orders))
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

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Restaurant RAG System", page_icon="🍽️", layout="wide")

# Log sink in session state
if "_logs" not in st.session_state:
    st.session_state._logs = []

# Sidebar — API Key and Init
st.sidebar.header("🔐 API Settings")
api_key = st.sidebar.text_input("GEMINI_API_KEY", value=DEFAULT_GEMINI_API_KEY, type="password")
init_clicked = st.sidebar.button("Initialize System", type="primary")

if init_clicked:
    if not api_key:
        st.sidebar.error("Please provide a GEMINI_API_KEY.")
    else:
        # Create system once per session
        st.session_state.rag = RestaurantRAGSystem(api_key)
        with st.spinner("Loading sample data..."):
            st.session_state.rag.initialize_sample_data()
        st.session_state.initialized = True
        st.sidebar.success("Initialized! Sample data loaded.")

# Main Title
st.title("🍽️ Restaurant RAG System Demo")

if not st.session_state.get("initialized"):
    st.info("Use the sidebar to initialize the system and load sample data.")
    st.stop()

rag: RestaurantRAGSystem = st.session_state.rag

# Columns for workflow
col1, col2 = st.columns(2)

with col1:
    st.subheader("1) Menu Recommendations")
    pref = st.text_input("Customer preference", value="I want something spicy and vegetarian")
    if st.button("Get Recommendations"):
        with st.spinner("Searching menu and generating response..."):
            rec = rag.get_menu_recommendations(pref)
        st.markdown("**Response:**")
        st.write(rec["response"])
        if rec["recommendations"]:
            st.markdown("**Recommended items:**")
            st.table(pd.DataFrame(rec["recommendations"]))

with col2:
    st.subheader("2) Customer Query (RAG)")
    q = st.text_input("Ask about menu or feedback", value="What do other customers think about your pizza?")
    if st.button("Process Query"):
        with st.spinner("Retrieving context and answering..."):
            ans = rag.process_customer_query(q)
        st.markdown("**Answer:**")
        st.write(ans["response"])
        with st.expander("Context used"):
            st.write({"menu": ans["menu_context"], "feedback": ans["feedback_context"]})

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.subheader("3) Analyze New Feedback")
    cust = st.text_input("Customer name", value="Alice Cooper")
    rating = st.slider("Rating", 1, 5, 4)
    fb = st.text_area("Feedback", value="The service was slow but the food was excellent!")
    if st.button("Analyze & Store Feedback"):
        with st.spinner("Analyzing sentiment and storing feedback..."):
            out = rag.analyze_customer_feedback(fb, cust, rating)
        st.markdown("**Sentiment Analysis:**")
        st.json(out["analysis"])
        if out["similar_feedback"]:
            st.markdown("**Similar Feedback:**")
            st.table(pd.DataFrame(out["similar_feedback"]))

with col4:
    st.subheader("4) Sales Report")
    if st.button("Generate Sales Report"):
        with st.spinner("Crunching numbers and generating insights..."):
            report = rag.generate_sales_report()
        if "error" in report:
            st.error(report["error"])
        else:
            st.markdown("**Metrics:**")
            st.json(report["metrics"])  # json keeps np types out
            st.markdown("**AI Insights:**")
            st.write(report["insights"])

st.divider()

st.subheader("5) Loyalty Points")
cid = st.text_input("Customer ID", value="C001")
if st.button("Calculate Loyalty"):
    with st.spinner("Calculating loyalty rewards..."):
        info = rag.calculate_loyalty_reward(cid)
    if "error" in info:
        st.error(info["error"])
    else:
        st.json(info)

# Developer logs (optional)
with st.expander("Developer logs"):
    for line in st.session_state._logs[-200:]:
        st.text(line)
