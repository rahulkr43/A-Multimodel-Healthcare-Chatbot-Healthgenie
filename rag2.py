# healthgenie.py - Complete AI Healthcare Assistant

import os
import re
from typing import Dict, List, Optional
import spacy

# LangChain imports
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.llms import OpenAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

# Initialize NLP model
nlp = spacy.load("en_core_web_sm")

# Configuration
DATA_PATH = r"Data\medical_vault.txt"
CHROMA_DIR = r"vectorstores\chroma_index"

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Local LLM (local server or ollama)

llm = OpenAI(
    base_url="http://localhost:1234/v1/",
    api_key="not-needed",  # Can be any string since it's local
    model="qwen2.5-7b-instruct-1m",
    temperature=0.3
)

# ======================
# RAG Setup
# ======================

def build_or_load_vectorstore(force_rebuild: bool = False):
    """Build or load Chroma vectorstore."""
    if force_rebuild or not os.path.exists(CHROMA_DIR):
        print("[INFO] Building vectorstore from documents...")
        loader = TextLoader(DATA_PATH)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            docs, embedding_model, persist_directory=CHROMA_DIR
        )
        vectorstore.persist()
    else:
        print("[INFO] Loading existing vectorstore...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embedding_model
        )
    return vectorstore

# Initialize RAG components
vectorstore = build_or_load_vectorstore(force_rebuild=False)
retriever = vectorstore.as_retriever()

system_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are HealthGenie, a helpful AI healthcare assistant. Provide health-related information including symptoms, medications, and hospital resources.
Avoid giving direct diagnoses or prescriptions.

ALWAYS respond in this exact format:

🧠 Suggested Diagnosis: [Your diagnosis suggestion based on symptoms] or give necessary information about the diesease or query realted

💡 Health Advice:
- [First recommendation]
- [Second recommendation]
- [Third recommendation]

🔗 Best Buy:
- [Medicine recommendation link]

🧪 Recommended Tests (if applicable):
- [Test 1]
- [Test 2]

🩸 Blood Bank Information (if applicable):
Name of Unit/Center | Address | Contact Number | Email ID | Type | Services | Notes

Context:
{context}

Question:
{question}

Answer:"""
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": system_prompt}
)

# ======================
# Intent Detection
# ======================

def extract_medicine_info(query: str) -> Dict:
    """Extract medicine details from query."""
    doc = nlp(query)
    medicine = None
    attributes = {
        "price": False,
        "side_effects": False,
        "dosage": False,
        "availability": False,
        "alternatives": False
    }
    
    # Check for medicine entities
    for ent in doc.ents:
        if ent.label_ in ["DRUG", "CHEMICAL"]:
            medicine = ent.text
            break
    
    # If no NER found, try pattern matching
    if not medicine:
        med_match = re.search(r"(?:price|side effects|dosage) of (\w+)", query, re.I)
        if med_match:
            medicine = med_match.group(1)
    
    # Detect medicine-related attributes
    query_lower = query.lower()
    if "price" in query_lower or "cost" in query_lower:
        attributes["price"] = True
    if "side effect" in query_lower or "side effects" in query_lower:
        attributes["side_effects"] = True
    if "dosage" in query_lower or "dose" in query_lower:
        attributes["dosage"] = True
    if "available" in query_lower or "stock" in query_lower:
        attributes["availability"] = True
    if "alternative" in query_lower or "substitute" in query_lower:
        attributes["alternatives"] = True
    
    return {"medicine": medicine, "attributes": attributes}

def extract_blood_info(query: str) -> Dict:
    """Extract blood donation details from query."""
    # Blood group detection
    blood_group = re.search(r'\b(A|B|AB|O)[+-]\b', query, re.IGNORECASE)
    blood_group = blood_group.group(0).upper() if blood_group else None
    
    # Location extraction
    doc = nlp(query)
    locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
    
    # Default to Bihar if no location specified
    state = "Bihar"
    district = None
    
    if locations:
        if len(locations) > 1:
            district, state = locations[0], locations[1]
        else:
            district = locations[0]
    
    return {
        "blood_group": blood_group,
        "state": state,
        "district": district
    }

def detect_intent_and_entities(query: str) -> Dict:
    """Advanced intent detection with entity extraction."""
    query_lower = query.lower()
    
    # Check for blood intent first (most specific)
    blood_info = extract_blood_info(query)
    if blood_info["blood_group"]:
        return {
            "intent": "blood",
            "entities": blood_info
        }
    
    # Check for medicine intent
    med_info = extract_medicine_info(query)
    if med_info["medicine"]:
        return {
            "intent": "medicine",
            "entities": med_info
        }
    
    # Default to symptom/advice intent
    return {
        "intent": "symptom",
        "entities": {"symptoms": [token.text for token in nlp(query) if token.pos_ == "NOUN"]}
    }

# ======================
# Response Generators
# ======================

def generate_symptom_response(rag_result: str) -> Dict:
    """Improved parsing of symptom/advice responses with better fallbacks."""
    # Initialize with default values
    response = {
        "diagnosis": "No specific diagnosis available",
        "advice": [],
        "tests": []
    }
    
    # Try to extract diagnosis
    diagnosis_match = re.search(r"Suggested Diagnosis:\s*(.*?)(?:\n|💡|🧪|$)", rag_result, re.IGNORECASE | re.DOTALL)
    if diagnosis_match:
        response["diagnosis"] = diagnosis_match.group(1).strip()
    
    # Try to extract health advice
    advice_match = re.search(r"Health Advice:\s*([\s\S]*?)(?:\n🧪|\n💡|$)", rag_result, re.IGNORECASE)
    if advice_match:
        advice_text = advice_match.group(1).strip()
        # Extract bullet points
        response["advice"] = [line.strip() for line in re.findall(r"- (.*?)(?:\n|$)", advice_text) if line.strip()]
    
    # Try to extract recommended tests
    tests_match = re.search(r"Recommended Tests:\s*([\s\S]*?)(?:\n|$)", rag_result, re.IGNORECASE)
    if tests_match:
        tests_text = tests_match.group(1).strip()
        response["tests"] = [line.strip() for line in re.findall(r"- (.*?)(?:\n|$)", tests_text) if line.strip()]
    
    # If no advice was found, try to use the entire response as advice
    if not response["advice"]:
        response["advice"] = [rag_result.strip()] if rag_result.strip() else ["No specific advice available"]
    
    return {
        "type": "symptom",
        "response": response
    }

# ======================
# Main Query Handler
# ======================
def handle_query(user_query: str) -> Dict:
    """Main query handler with intelligent routing."""
    intent_data = detect_intent_and_entities(user_query)
    
    try:
        if intent_data["intent"] == "medicine":
            medicine_name = intent_data["entities"]["medicine"]
            attributes = intent_data["entities"]["attributes"]

            # Construct a combined RAG query
            rag_query = f"Tell me about {medicine_name}."
            if attributes["side_effects"]:
                rag_query += " Include side effects."
            if attributes["dosage"]:
                rag_query += " Include recommended dosage."
            if attributes["availability"]:
                rag_query += " Check availability."
            if attributes["alternatives"]:
                rag_query += " Suggest alternatives."
            if attributes["price"]:
                rag_query += " Mention the price."

            rag_response = rag_chain.invoke({"query": rag_query})
            return generate_symptom_response(rag_response["result"])
        
        elif intent_data["intent"] == "blood":
            blood_info = intent_data["entities"]
            blood_group = blood_info.get("blood_group", "Unknown")
            location = f"{blood_info.get('district', '')}, {blood_info.get('state', '')}"
            
            rag_query = f"Where can I find {blood_group} blood in {location}?"
            rag_response = rag_chain.invoke({"query": rag_query})
            return generate_symptom_response(rag_response["result"])
        
        else:
            rag_response = rag_chain.invoke({"query": user_query})
            return generate_symptom_response(rag_response["result"])
    
    except Exception as e:
        return {
            "type": "error",
            "response": f"An error occurred: {str(e)}"
        }

# ======================
# Example Usage
# ======================

if __name__ == "__main__":
    test_queries = [
        "I have headache and fever, what should I do?",
        "What is the price of Dolo 650?",
        "Where can I find O+ blood in Patna, Bihar?",
        "What are the side effects of Cetirizine?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = handle_query(query)
        print("Response:", response)