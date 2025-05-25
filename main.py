from flask import Flask, request, jsonify, render_template
import json

from rag3 import (build_or_load_vectorstore,detect_intent_and_entities,scrape_tata1mg,
    scrape_blood_availability,
    generate_symptom_response,
    generate_medicine_response,
    generate_blood_response,handle_query,
    rag_chain
)

app = Flask(__name__)

# Initialize RAG components at startup
vectorstore = build_or_load_vectorstore(force_rebuild=False)
retriever = vectorstore.as_retriever()

@app.route("/")
@app.route("/home")
def home():
    """Render the main application page."""
    return render_template("home.html", title="HealthGenie")


@app.route("/chat", methods=["POST"])
def chat_with_bot():
    """Handle chat requests and route to appropriate response handlers."""
    data = request.get_json()
    
    # Validate request
    if not data or "query" not in data:
        return jsonify({
            "type": "error",
            "response": "Missing query field"
        }), 400

    query = data["query"].strip()
    
    try:
        # Step 1: Detect intent and extract entities
        intent_data = detect_intent_and_entities(query)
        
        # Step 2: Route to appropriate handler
        if intent_data["intent"] == "medicine":
            return handle_medicine_query(intent_data)
        elif intent_data["intent"] == "blood":
            return handle_blood_query(intent_data)
        else:
            return handle_symptom_query(query)
            
    except Exception as e:
        return jsonify({
            "type": "error",
            "response": f"An error occurred: {str(e)}"
        }), 500

def handle_symptom_query(query: str):
    """Handle symptom/advice queries using RAG."""
    rag_response = rag_chain.invoke({"query": query})
    structured_response = generate_symptom_response(rag_response["result"])
    return jsonify(structured_response)

def handle_medicine_query(intent_data: dict):
    """Handle medicine-related queries with scraping fallback."""
    medicine_name = intent_data["entities"]["medicine"]
    attributes = intent_data["entities"]["attributes"]
    
    # Try scraping first
    medicine_info = scrape_tata1mg(medicine_name)
    
    # If scraping fails, fall back to RAG
    if "error" in medicine_info:
        rag_query = f"Tell me about {medicine_name}"
        if attributes["side_effects"]:
            rag_query += " side effects"
        if attributes["dosage"]:
            rag_query += " recommended dosage"
        
        rag_response = rag_chain.invoke({"query": rag_query})
        return jsonify(generate_symptom_response(rag_response["result"]))
    
    # Enhance with additional info if requested
    if attributes["side_effects"] and "side_effects" not in medicine_info:
        rag_query = f"What are the side effects of {medicine_name}?"
        rag_response = rag_chain.invoke({"query": rag_query})
        medicine_info["side_effects"] = rag_response["result"]
    
    if attributes["dosage"] and "dosage" not in medicine_info:
        rag_query = f"What is the recommended dosage for {medicine_name}?"
        rag_response = rag_chain.invoke({"query": rag_query})
        medicine_info["dosage"] = rag_response["result"]
    
    return jsonify(generate_medicine_response(medicine_info))

def handle_blood_query(intent_data: dict):
    """Handle blood availability queries."""
    blood_info = intent_data["entities"]
    
    # Validate required fields
    if not blood_info["blood_group"]:
        return jsonify({
            "type": "error",
            "response": "Please specify a valid blood group (e.g., A+, B-, O+)"
        })
    
    # Default to Bihar if no state specified
    state = blood_info["state"] or "Bihar"
    district = blood_info["district"] or "Patna"  # Default district
    
    # Get blood availability data
    blood_data = scrape_blood_availability(
        state=state,
        district=district,
        blood_group=blood_info["blood_group"]
    )
    
    return jsonify(generate_blood_response(blood_data))


if __name__ == "__main__":
    app.run(debug=True, port=5000)