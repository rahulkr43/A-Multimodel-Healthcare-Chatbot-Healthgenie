# HealthGenie: AI-powered RAG-based Healthcare Assistant (Full Version with Comments)

# =================================
# STEP 1: SETUP & LIBRARY IMPORTS
# =================================

# Core Python Libraries
import os
from openai import OpenAI
from typing import List, Dict

# LangChain Imports
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Web scraping
from playwright.sync_api import sync_playwright

# NLP for intent detection
import spacy 


# =========================
# STEP 2: CONFIGURATION
# =========================

# Set OpenAI Key - make sure to set securely in .env or system variables
client = OpenAI(base_url="http://localhost:1234/v1/", api_key="not-needed")
os.environ["OPENAI_API_KEY"] = "not=needed"

# Initialize LLM and Embedding Model
from langchain_community.llms import OpenAI

llm = OpenAI(
    base_url="http://localhost:1234/v1/",
    api_key="not-needed",  # Can be any string since it's local
    model="qwen2.5-7b-instruct-1m",
    temperature=0.3
)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# ================================
# STEP 3: DOCUMENT LOADING & RAG
# ================================

# ============================
# VECTOR STORE & LLM SETUP
# ============================

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Directories
DATA_PATH = r"Data\medical_vault.txt"
CHROMA_DIR = r"vectorstores\chroma_index"

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Initialize Chat LLM
llm = ChatOpenAI(base_url="http://localhost:1234/v1/",model_name="dolphin-2.2.1-mistral-7b", temperature=0.3)


def build_or_load_vectorstore(force_rebuild: bool = False):
    """
    Builds Chroma vectorstore from documents or loads it from disk if exists.
    Set `force_rebuild=True` to regenerate embeddings from scratch.
    """
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
        vectorstore = Chroma(persist_directory=CHROMA_DIR,embedding_function=embedding_model)
    

    return vectorstore

# Load vectorstore and setup retriever
vectorstore = build_or_load_vectorstore(force_rebuild=False)
retriever = vectorstore.as_retriever()

# Custom system prompt
system_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are HealthGenie, a helpful AI healthcare assistant. Provide health-related information including symptoms, medications, and hospital resources.
Avoid giving direct diagnoses or prescriptions.

Context:
{context}

Question:
{question}

Answer:"""
)

# RAG chain setup
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": system_prompt}
)

# ===============================
# STEP 4: INTENT DETECTION LOGIC
# ===============================

def detect_intent(query: str) -> str:
    query_lower = query.lower()
    if any(bg in query.upper() for bg in ["A+", "B+", "O-", "AB-", "O+", "A-", "B-", "AB+"]):
        return "blood"
    elif any(keyword in query_lower for keyword in ["medicine", "buy", "price", "tablet"]):
        return "medicine"
    else:
        return "symptom"

# ============================
# STEP 5: WEB SCRAPER UTILS
# ============================

# Medicine scraper from Tata 1mg
def scrape_tata1mg(medicine_name: str):
    from playwright.sync_api import sync_playwright

    url = f"https://www.1mg.com/search/all?name={medicine_name}"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        try:
            page.goto(url, timeout=60000)
            page.wait_for_timeout(3000)  # give extra time to render JS

            # Print HTML for debugging
            html = page.content()
            with open("1mg_debug.html", "w", encoding="utf-8") as f:
                f.write(html)

            name_el = page.query_selector(".style__pro-title___3G3rr")
            price_el = page.query_selector(".style__price-tag___B2csA")
            link_el = page.query_selector(".style__product-box___3oEU6 a")

            if not name_el or not price_el or not link_el:
                raise ValueError("Some product info not found on page.")

            name = name_el.inner_text()
            price = price_el.inner_text()
            link = f"https://www.1mg.com{link_el.get_attribute('href')}"

            browser.close()
            return {"name": name, "price": price, "link": link}

        except Exception as e:
            browser.close()
            print("1mg scraping failed:", e)
            return {"error": "Could not extract medicine info."}

# Dummy function for blood availability
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

def scrape_blood_availability(state: str, district: str, blood_group: str) -> List[Dict]:
    """
    Scrapes blood unit availability from the eRaktKosh blood bank website for a given state, district, and blood group.
    """
    url = "https://eraktkosh.mohfw.gov.in/BLDAHIMS/bloodbank/stockAvailability.cnt"
    session = requests.Session()

    # Fetch the initial page to get any necessary cookies or tokens
    response = session.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Prepare the payload for the POST request
    payload = {
        'stateCode': state,
        'districtCode': district,
        'bloodGroup': blood_group,
        'bloodComponent': 'Whole Blood',  # Adjust if necessary
    }

    # Send the POST request
    post_response = session.post(url, data=payload)
    post_soup = BeautifulSoup(post_response.text, 'html.parser')

    # Parse the results
    results = []
    rows = post_soup.select("#stockDetails tbody tr")
    if not rows:
        return [{"message": "No data found for your query"}]

    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 7:
            bank_name = cols[0].get_text(strip=True)
            location = cols[1].get_text(strip=True)
            contact = cols[5].get_text(strip=True)
            units = cols[6].get_text(strip=True)
            results.append({
                "blood_bank": bank_name,
                "location": location,
                "contact": contact,
                "blood_group": blood_group,
                "available_units": units
            })

    return results

# ============================
# STEP 6: API SETUP
# ============================


# Future Enhancements:
# - Use spaCy NER for medicine/blood location extraction
# - OCR/TTS integrations
# - Add PDF loaders
# - Chat frontend UI
# - Session-based caching for performance
