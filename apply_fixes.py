#!/usr/bin/env python3
"""
Script to apply PDF chunking and question answering fixes to the existing restaurant RAG app.
"""

import re
import os

def apply_fixes_to_app():
    """Apply the fixes to app(copy2).py"""
    
    # Read the current app file
    with open("app(copy2).py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Fix 1: Replace the initialize_sample_data method with improved version
    old_initialize_method = '''    def initialize_sample_data(self, pdf_path: str = "taj-amer-food-menu.pdf"):
        if not os.path.exists(pdf_path):
            _log(f"PDF not found at {pdf_path}")
            return

        # Extract text from PDF
        try:
            doc = fitz.open(pdf_path)
            full_text = "\\n\\n".join([p.get_text("text") for p in doc])
            doc.close()  # Close the document
            _log("Extracted text from PDF.")
        except Exception as e:
            _log(f"Failed to read PDF: {e}")
            return

        if not full_text.strip():
            _log("PDF appears to be empty or unreadable")
            return

        # Ask Gemini to parse menu into structured JSON (best-effort)
        parse_prompt = f"""
You are a menu parser. Convert the following restaurant menu text into a JSON array where each menu item has these fields:
- id: string (unique identifier)
- name: string (dish name)
- description: string (description of the dish)
- category: string (appetizer, main course, dessert, etc.)
- price: number (price in rupees, extract numeric value only)
- ingredients: array of strings (main ingredients)
- dietary_tags: array of strings (vegetarian, vegan, gluten-free, etc.)
- spice_level: string (mild, medium, spicy, or unknown)

Return ONLY a valid JSON array, no other text.

Menu text:
{{full_text[:4000]}}  
"""
        
        try:
            parsed = self.gemini_service.generate_response(parse_prompt)
            parsed = parsed.strip()
            
            # Clean up response - remove code fences if present
            if "```json" in parsed:
                parsed = parsed.split("```json")[1].split("```")[0].strip()
            elif "```" in parsed:
                parts = parsed.split("```")
                if len(parts) >= 3:
                    parsed = parts[1].strip()

            menu_items_data = json.loads(parsed)
            if not isinstance(menu_items_data, list):
                raise ValueError("LLM returned non-list")
            _log(f"Parsed {{len(menu_items_data)}} menu items from LLM.")
            
        except Exception as e:
            _log(f"Failed to parse JSON from LLM: {{e}}. Falling back to simple chunking.")
            # fallback: chunk the text
            chunks = [c.strip() for c in full_text.split("\\n\\n") if len(c.strip()) > 20][:50]
            menu_items_data = []
            for i, ch in enumerate(chunks):
                name = ch.split("\\n")[0][:80] if "\\n" in ch else ch[:80]
                menu_items_data.append({{
                    "id": str(i+1),
                    "name": name,
                    "description": ch.replace("\\n", " ")[:400],
                    "category": "Uncategorized",
                    "price": 0.0,
                    "ingredients": [],
                    "dietary_tags": [],
                    "spice_level": "unknown"
                }})
            _log(f"Fallback parsed {{len(menu_items_data)}} chunks.")

        # Prepare docs + meta
        docs = []
        meta = []
        for i, item in enumerate(menu_items_data):
            try:
                m_id = str(item.get("id", i+1))
                name = item.get("name", "Unnamed")
                desc = item.get("description", "")
                cat = item.get("category", "Uncategorized")
                price = float(item.get("price", 0.0) or 0.0)
                ingredients = item.get("ingredients", []) or []
                diet = item.get("dietary_tags", []) or []
                spice = item.get("spice_level", "unknown") or "unknown"
                
                # Ensure ingredients and diet are lists
                if isinstance(ingredients, str):
                    ingredients = [ingredients]
                if isinstance(diet, str):
                    diet = [diet]
                
                text = f"{{name}} - {{desc}} - Category: {{cat}} - Price: ₹{{price}} - Ingredients: {{', '.join(ingredients)}} - Dietary: {{', '.join(diet)}} - Spice: {{spice}}"
                docs.append(text)
                meta.append({{
                    "id": m_id,
                    "name": name,
                    "price": price,
                    "category": cat,
                    "dietary_tags": diet,
                    "spice_level": spice
                }})
            except Exception as ex:
                _log(f"Error preparing item #{{i}}: {{ex}}")

        if not docs:
            _log("No documents to ingest.")
            return

        _log(f"Ingesting {{len(docs)}} documents into vector store...")
        self.menu_vector_store.add_documents(docs, meta)
        _log("Menu ingestion complete.")

        # Seed feedback sample into feedback store (only when using SimpleVectorStore)
        if isinstance(self.feedback_vector_store, SimpleVectorStore) and len(self.feedback_vector_store.documents) == 0:
            sample_fb = [
                CustomerFeedback("John Doe", "Loved the lotus stem! A bit too salty though.", 4, "2024-01-15"),
                CustomerFeedback("Jane Smith", "Prawns were perfectly cooked. Excellent service!", 5, "2024-01-16"),
                CustomerFeedback("Mike Johnson", "The biryani was amazing but took too long to arrive.", 3, "2024-01-17"),
            ]
            fb_docs, fb_meta = [], []
            for f in sample_fb:
                fb_docs.append(f"Customer: {{f.customer_name}} - Rating: {{f.rating}}/5 - Feedback: {{f.feedback}}")
                fb_meta.append({{
                    "customer_name": f.customer_name, 
                    "rating": f.rating, 
                    "feedback": f.feedback, 
                    "date": f.date
                }})
            self.feedback_vector_store.add_documents(fb_docs, fb_meta)
            _log("Seeded feedback examples.")'''

    new_initialize_method = '''    def initialize_sample_data(self, pdf_path: str = "taj-amer-food-menu.pdf"):
        if not os.path.exists(pdf_path):
            _log(f"PDF not found at {pdf_path}")
            return

        # Extract and parse menu items with improved chunking
        text = self._extract_text_from_pdf(pdf_path)
        if not text.strip():
            _log("PDF appears to be empty or unreadable")
            return

        menu_items = self._parse_menu_items(text)
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

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with better formatting preservation."""
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Get text with better formatting
                text = page.get_text("text")
                # Clean up common PDF artifacts
                text = re.sub(r'\\s+', ' ', text)  # Normalize whitespace
                text = re.sub(r'([a-z])([A-Z])', r'\\1 \\2', text)  # Add space between camelCase
                text_parts.append(text.strip())
            
            doc.close()
            full_text = "\\n\\n".join(text_parts)
            _log(f"Extracted {len(full_text)} characters from PDF")
            return full_text
            
        except Exception as e:
            _log(f"Failed to read PDF: {e}")
            return ""
    
    def _smart_chunk_text(self, text: str) -> List[str]:
        """Intelligently chunk text into menu items using multiple strategies."""
        chunks = []
        
        # Strategy 1: Split by price patterns (₹, $, etc.)
        price_pattern = r'[₹$]\\s*\\d+'
        price_splits = re.split(price_pattern, text)
        
        if len(price_splits) > 1:
            # Found price-based splits
            for i, chunk in enumerate(price_splits[1:], 1):  # Skip first empty chunk
                if len(chunk.strip()) > 10:
                    chunks.append(chunk.strip())
            _log(f"Strategy 1 (price-based): Found {len(chunks)} chunks")
        
        # Strategy 2: Split by common menu separators
        if not chunks:
            separators = ['\\n\\n', '•', '▪', '▫', '○', '●', '◆', '◇']
            for sep in separators:
                if sep in text:
                    chunks = [c.strip() for c in text.split(sep) if len(c.strip()) > 10]
                    _log(f"Strategy 2 (separator '{sep}'): Found {len(chunks)} chunks")
                    break
        
        # Strategy 3: Split by numbered items
        if not chunks:
            numbered_pattern = r'\\d+\\.\\s*'
            numbered_splits = re.split(numbered_pattern, text)
            if len(numbered_splits) > 1:
                chunks = [c.strip() for c in numbered_splits[1:] if len(c.strip()) > 10]
                _log(f"Strategy 3 (numbered): Found {len(chunks)} chunks")
        
        # Strategy 4: Split by line breaks and group related lines
        if not chunks:
            lines = text.split('\\n')
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
            _log(f"Strategy 4 (line-based): Found {len(chunks)} chunks")
        
        # Filter and clean chunks
        cleaned_chunks = []
        for chunk in chunks:
            if len(chunk) > 20 and len(chunk) < 1000:  # Reasonable size
                # Remove common PDF artifacts
                chunk = re.sub(r'^\\s*[0-9]+\\s*', '', chunk)  # Remove page numbers
                chunk = re.sub(r'\\s+', ' ', chunk)  # Normalize whitespace
                cleaned_chunks.append(chunk.strip())
        
        _log(f"Final cleaned chunks: {len(cleaned_chunks)}")
        return cleaned_chunks
    
    def _parse_menu_items(self, text: str) -> List[Dict]:
        """Parse text into structured menu items with improved LLM prompts."""
        chunks = self._smart_chunk_text(text)
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
                    
                    # Create menu item with validation
                    menu_item = {
                        "id": str(i + 1),
                        "name": item_data.get('name', chunk[:50]) or chunk[:50],
                        "description": item_data.get('description', '') or '',
                        "category": item_data.get('category', 'Uncategorized') or 'Uncategorized',
                        "price": float(item_data.get('price', 0.0) or 0.0),
                        "ingredients": item_data.get('ingredients', []) or [],
                        "dietary_tags": item_data.get('dietary_tags', []) or [],
                        "spice_level": item_data.get('spice_level', 'unknown') or 'unknown'
                    }
                    menu_items.append(menu_item)
                    _log(f"Successfully parsed item {i+1}: {menu_item['name']}")
                    
                except Exception as e:
                    _log(f"LLM parsing failed for chunk {i}: {e}")
                    # Fallback: create basic item
                    menu_item = {
                        "id": str(i + 1),
                        "name": chunk[:50],
                        "description": chunk,
                        "category": "Uncategorized",
                        "price": 0.0,
                        "ingredients": [],
                        "dietary_tags": [],
                        "spice_level": "unknown"
                    }
                    menu_items.append(menu_item)
                    _log(f"Fallback created item {i+1}: {menu_item['name']}")
                    
            except Exception as e:
                _log(f"Error processing chunk {i}: {e}")
                continue
        
        _log(f"Successfully parsed {len(menu_items)} menu items")
        return menu_items'''

    # Fix 2: Replace the process_customer_query method with improved version
    old_process_method = '''    def process_customer_query(self, query: str):
        menu_results = self.menu_vector_store.search(query, top_k=3)
        feedback_results = self.feedback_vector_store.search(query, top_k=3)
        
        context_parts = []
        if menu_results:
            menu_context = "Relevant Menu Items:\\n" + "\\n".join([
                f"- {{m['metadata'].get('name','Unknown')}}: ₹{{m['metadata'].get('price','N/A')}} ({{m['metadata'].get('category','')}})" 
                for m in menu_results
            ])
            context_parts.append(menu_context)
        
        if feedback_results:
            feedback_context = "Relevant Customer Feedback:\\n" + "\\n".join([
                f"- {{f['metadata'].get('customer_name','Anonymous')}}: {{f['metadata'].get('feedback','')}}" 
                for f in feedback_results
            ])
            context_parts.append(feedback_context)
        
        context = "\\n\\n".join(context_parts)
        answer = self.gemini_service.generate_response(query, context)
        
        return {{
            "response": answer,
            "menu_context": [m.get("metadata", {{}}) for m in menu_results],
            "feedback_context": [f.get("metadata", {{}}) for f in feedback_results]
        }}'''

    new_process_method = '''    def process_customer_query(self, query: str):
        """Process customer queries with improved context and prompts."""
        # Search for relevant menu items and feedback
        menu_results = self.menu_vector_store.search(query, top_k=5)
        feedback_results = self.feedback_vector_store.search(query, top_k=3)
        
        # Build comprehensive context
        context_parts = []
        
        if menu_results:
            menu_context = "Available Menu Items:\\n" + "\\n".join([
                f"• {{m['metadata'].get('name','Unknown')}} (₹{{m['metadata'].get('price','N/A')}})\\n"
                f"  Category: {{m['metadata'].get('category','')}}\\n"
                f"  Description: {{m['metadata'].get('description','')}}\\n"
                f"  Ingredients: {{', '.join(m['metadata'].get('ingredients',[]))}}\\n"
                f"  Dietary: {{', '.join(m['metadata'].get('dietary_tags',[]))}}\\n"
                f"  Spice Level: {{m['metadata'].get('spice_level','unknown')}}"
                for m in menu_results
            ])
            context_parts.append(menu_context)
        
        if feedback_results:
            feedback_context = "Customer Feedback:\\n" + "\\n".join([
                f"• {{f['metadata'].get('customer_name','Anonymous')}} (Rating: {{f['metadata'].get('rating','N/A')}}/5): {{f['metadata'].get('feedback','')}}"
                for f in feedback_results
            ])
            context_parts.append(feedback_context)
        
        context = "\\n\\n".join(context_parts)
        
        # Use improved prompt for better responses
        system_prompt = """You are a helpful restaurant assistant. Based on the menu items and customer feedback provided, answer the customer's question accurately and helpfully. 

If the customer is asking about:
- Menu items: Provide details about the dishes, prices, ingredients, and dietary information
- Recommendations: Suggest items based on their preferences and popular choices
- Customer feedback: Share relevant customer experiences and ratings
- General questions: Provide helpful information about the restaurant

Be friendly, informative, and specific. If you don't have enough information, say so politely."""
        
        full_prompt = f"{{system_prompt}}\\n\\nContext:\\n{{context}}\\n\\nCustomer Question: {{query}}\\n\\nAssistant:"
        
        answer = self.gemini_service.generate_response(full_prompt)
        
        return {{
            "response": answer,
            "menu_context": [m.get("metadata", {{}}) for m in menu_results],
            "feedback_context": [f.get("metadata", {{}}) for f in feedback_results]
        }}'''

    # Apply the replacements
    content = content.replace(old_initialize_method, new_initialize_method)
    content = content.replace(old_process_method, new_process_method)
    
    # Write the fixed file
    with open("app_fixed.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ Applied fixes to app_fixed.py")
    print("📝 Changes made:")
    print("  1. Improved PDF text extraction with better formatting")
    print("  2. Added smart chunking with multiple strategies")
    print("  3. Enhanced menu item parsing with better LLM prompts")
    print("  4. Improved question answering with comprehensive context")
    print("  5. Better error handling and fallbacks")

if __name__ == "__main__":
    apply_fixes_to_app()
