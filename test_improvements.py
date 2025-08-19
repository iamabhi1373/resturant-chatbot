#!/usr/bin/env python3
"""
Test script to verify the PDF chunking and question answering improvements.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_pdf_chunking():
    """Test the improved PDF chunking functionality."""
    print("🧪 Testing PDF chunking improvements...")
    
    # Check if PDF exists
    pdf_path = "taj-amer-food-menu.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    try:
        # Import the improved functions
        from pdf_chunking_fix import ImprovedMenuParser
        
        # Mock Gemini service for testing
        class MockGeminiService:
            def generate_response(self, prompt):
                # Return a simple mock response
                return '{"name": "Test Dish", "description": "Test description", "category": "main course", "price": 15.0, "ingredients": ["test"], "dietary_tags": ["vegetarian"], "spice_level": "medium"}'
        
        # Test the parser
        parser = ImprovedMenuParser(MockGeminiService())
        
        # Extract text
        text = parser.extract_text_from_pdf(pdf_path)
        if not text:
            print("❌ Failed to extract text from PDF")
            return False
        
        print(f"✅ Extracted {len(text)} characters from PDF")
        
        # Test chunking
        chunks = parser.smart_chunk_text(text)
        if not chunks:
            print("❌ Failed to create chunks")
            return False
        
        print(f"✅ Created {len(chunks)} chunks")
        
        # Show sample chunks
        for i, chunk in enumerate(chunks[:3]):
            print(f"   Chunk {i+1}: {chunk[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing PDF chunking: {e}")
        return False

def test_question_answering():
    """Test the improved question answering functionality."""
    print("\n🧪 Testing question answering improvements...")
    
    try:
        # Import the improved functions
        from pdf_chunking_fix import improved_process_customer_query, improved_get_menu_recommendations
        
        # Mock RAG system for testing
        class MockRAGSystem:
            def __init__(self):
                self.menu_vector_store = MockVectorStore()
                self.feedback_vector_store = MockVectorStore()
                self.gemini_service = MockGeminiService()
        
        class MockVectorStore:
            def search(self, query, top_k=5):
                return [
                    {
                        "document": "Test menu item",
                        "metadata": {
                            "name": "Butter Chicken",
                            "price": 18.0,
                            "category": "main course",
                            "description": "Creamy and delicious",
                            "ingredients": ["chicken", "butter", "cream"],
                            "dietary_tags": [],
                            "spice_level": "medium"
                        },
                        "similarity": 0.8
                    }
                ]
        
        class MockGeminiService:
            def generate_response(self, prompt):
                return "This is a test response from the improved question answering system."
        
        # Test the functions
        rag_system = MockRAGSystem()
        
        # Test customer query processing
        result = improved_process_customer_query(rag_system, "What vegetarian options do you have?")
        if result and "response" in result:
            print("✅ Customer query processing works")
            print(f"   Response: {result['response'][:100]}...")
        else:
            print("❌ Customer query processing failed")
            return False
        
        # Test recommendations
        result = improved_get_menu_recommendations(rag_system, "I want something spicy")
        if result and "response" in result:
            print("✅ Menu recommendations work")
            print(f"   Response: {result['response'][:100]}...")
        else:
            print("❌ Menu recommendations failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing question answering: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Restaurant RAG Improvements")
    print("=" * 50)
    
    # Test PDF chunking
    chunking_ok = test_pdf_chunking()
    
    # Test question answering
    qa_ok = test_question_answering()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   PDF Chunking: {'✅ PASS' if chunking_ok else '❌ FAIL'}")
    print(f"   Question Answering: {'✅ PASS' if qa_ok else '❌ FAIL'}")
    
    if chunking_ok and qa_ok:
        print("\n🎉 All tests passed! The improvements are working correctly.")
        print("\n📝 Next steps:")
        print("   1. Use 'app_fixed.py' instead of 'app(copy2).py'")
        print("   2. Upload your menu PDF and test the chatbot")
        print("   3. The system should now properly chunk PDFs and answer questions better")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return chunking_ok and qa_ok

if __name__ == "__main__":
    main()
