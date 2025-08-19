# pdf_chunking_fix.py
"""
Focused fix for PDF chunking and menu parsing issues in the restaurant RAG app.
This module provides improved PDF processing functions that can be integrated into the main app.
"""

import re
import json
from typing import List, Dict
from dataclasses import dataclass

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

class ImprovedMenuParser:
    """Improved menu parser with better chunking and parsing logic."""
    
    def __init__(self, gemini_service):
        self.gemini_service = gemini_service
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with better formatting preservation."""
        import fitz  # PyMuPDF
        
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
            print(f"Extracted {len(full_text)} characters from PDF")
            return full_text
            
        except Exception as e:
            print(f"Failed to read PDF: {e}")
            return ""
    
    def smart_chunk_text(self, text: str) -> List[str]:
        """Intelligently chunk text into menu items using multiple strategies."""
        chunks = []
        
        # Strategy 1: Split by price patterns (₹, $, etc.)
        price_pattern = r'[₹$]\s*\d+'
        price_splits = re.split(price_pattern, text)
        
        if len(price_splits) > 1:
            # Found price-based splits
            for i, chunk in enumerate(price_splits[1:], 1):  # Skip first empty chunk
                if len(chunk.strip()) > 10:
                    chunks.append(chunk.strip())
            print(f"Strategy 1 (price-based): Found {len(chunks)} chunks")
        
        # Strategy 2: Split by common menu separators
        if not chunks:
            separators = ['\n\n', '•', '▪', '▫', '○', '●', '◆', '◇']
            for sep in separators:
                if sep in text:
                    chunks = [c.strip() for c in text.split(sep) if len(c.strip()) > 10]
                    print(f"Strategy 2 (separator '{sep}'): Found {len(chunks)} chunks")
                    break
        
        # Strategy 3: Split by numbered items
        if not chunks:
            numbered_pattern = r'\d+\.\s*'
            numbered_splits = re.split(numbered_pattern, text)
            if len(numbered_splits) > 1:
                chunks = [c.strip() for c in numbered_splits[1:] if len(c.strip()) > 10]
                print(f"Strategy 3 (numbered): Found {len(chunks)} chunks")
        
        # Strategy 4: Split by line breaks and group related lines
        if not chunks:
            lines = text.split('\n')
            current_chunk = []
            for line in lines:
                line = line.strip()
                if line:
                    current_chunk.append(line)
                elif current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text) > 20:
                        chunks.append(chunk_text)
                    current_chunk = []
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) > 20:
                    chunks.append(chunk_text)
            print(f"Strategy 4 (line-based): Found {len(chunks)} chunks")
        
        # Filter and clean chunks
        cleaned_chunks = []
        for chunk in chunks:
            if len(chunk) > 20 and len(chunk) < 1000:  # Reasonable size
                # Remove common PDF artifacts
                chunk = re.sub(r'^\s*[0-9]+\s*', '', chunk)  # Remove page numbers
                chunk = re.sub(r'\s+', ' ', chunk)  # Normalize whitespace
                cleaned_chunks.append(chunk.strip())
        
        print(f"Final cleaned chunks: {len(cleaned_chunks)}")
        return cleaned_chunks
    
    def parse_menu_items(self, text: str) -> List[MenuItem]:
        """Parse text into structured menu items with improved LLM prompts."""
        chunks = self.smart_chunk_text(text)
        menu_items = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Improved prompt for better parsing
                parse_prompt = f"""
You are a menu parser. Parse this menu item text into a JSON object with these exact fields:
{{
    "name": "dish name (string, extract the main dish name)",
    "description": "description (string, can be empty if no description)",
    "category": "appetizer/main course/dessert/beverage/soup/salad/etc (string)",
    "price": price as number (float, extract numeric value only, 0.0 if not found),
    "ingredients": ["ingredient1", "ingredient2"] (array of strings, empty if not mentioned),
    "dietary_tags": ["vegetarian", "vegan", "gluten-free", etc] (array of strings, empty if not mentioned),
    "spice_level": "mild/medium/spicy/unknown" (string)
}}

Menu text: "{chunk}"

Return ONLY valid JSON, no other text or explanations:"""

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
                    
                    # Remove any non-JSON text before/after
                    response = re.sub(r'^[^{]*', '', response)
                    response = re.sub(r'[^}]*$', '', response)
                    
                    item_data = json.loads(response)
                    
                    # Create MenuItem with validation
                    menu_item = MenuItem(
                        id=str(i + 1),
                        name=item_data.get('name', chunk[:50]) or chunk[:50],
                        description=item_data.get('description', '') or '',
                        category=item_data.get('category', 'Uncategorized') or 'Uncategorized',
                        price=float(item_data.get('price', 0.0) or 0.0),
                        ingredients=item_data.get('ingredients', []) or [],
                        dietary_tags=item_data.get('dietary_tags', []) or [],
                        spice_level=item_data.get('spice_level', 'unknown') or 'unknown'
                    )
                    menu_items.append(menu_item)
                    print(f"Successfully parsed item {i+1}: {menu_item.name}")
                    
                except Exception as e:
                    print(f"LLM parsing failed for chunk {i}: {e}")
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
                    print(f"Fallback created item {i+1}: {menu_item.name}")
                    
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                continue
        
        print(f"Successfully parsed {len(menu_items)} menu items")
        return menu_items

def improved_process_customer_query(rag_system, query: str):
    """Improved customer query processing with better context and prompts."""
    # Search for relevant menu items and feedback
    menu_results = rag_system.menu_vector_store.search(query, top_k=5)
    feedback_results = rag_system.feedback_vector_store.search(query, top_k=3)
    
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
    
    answer = rag_system.gemini_service.generate_response(full_prompt)
    
    return {
        "response": answer,
        "menu_context": [m.get("metadata", {}) for m in menu_results],
        "feedback_context": [f.get("metadata", {}) for f in feedback_results]
    }

def improved_get_menu_recommendations(rag_system, customer_query: str):
    """Get personalized menu recommendations with better context."""
    similar = rag_system.menu_vector_store.search(customer_query, top_k=8)
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

    response = rag_system.gemini_service.generate_response(recommendation_prompt, context)
    
    return {
        "response": response, 
        "recommendations": [s['metadata'] for s in similar]
    }
