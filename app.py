from flask import Flask, request, jsonify, render_template
import json
from rag2 import (
    build_or_load_vectorstore,
    detect_intent_and_entities,
    generate_symptom_response,
    handle_query,
    rag_chain
)
from pdf import pdf_bp 

app = Flask(__name__)

# Initialize RAG components at startup
vectorstore = build_or_load_vectorstore(force_rebuild=False)
retriever = vectorstore.as_retriever()
# Register blueprint for pdf uploader
app.register_blueprint(pdf_bp)

@app.route("/")
@app.route("/home")
def home():
    """Render the main application page."""
    return render_template("home.html", title="HealthGenie")

@app.route("/chat", methods=["POST"])
def chat_with_bot():
    """Handle chat requests using the RAG system."""
    data = request.get_json()
    
    # Validate request
    if not data or "query" not in data:
        return jsonify({
            "type": "error",
            "response": "Missing query field"
        }), 400

    query = data["query"].strip()
    
    try:
        # Handle the query using the RAG system
        response = handle_query(query)
        return jsonify(response)
            
    except Exception as e:
        return jsonify({
            "type": "error",
            "response": f"An error occurred: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)