from flask import Flask, request, jsonify
from flask_cors import CORS
import datetime
import pdfplumber
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.callbacks.manager import get_openai_callback

# PDF path
pdf_path = "data.pdf"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to create embeddings and vector store
def create_vector_store(text):
    api_key = "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"  # Replace with your OpenAI API Key
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_texts(text.split("\n"), embeddings)
    return vector_store

# Answer function using LangChain
def answer(text, question):
    vector_store = create_vector_store(text)
    retriever = vector_store.as_retriever()
    llm = OpenAI(openai_api_key="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")  # Replace with your OpenAI API Key
    chain = ConversationalRetrievalChain.from_llm(llm, retriever)

    with get_openai_callback() as cb:
        response = chain({"question": question, "chat_history": []})
        print(f"Total Tokens: {cb.total_tokens}, Total Cost (USD): ${cb.total_cost:.6f}")
        return response["answer"]

# Extract text from the PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Flask app setup
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "ChatBot Flask Server is running!"

@app.route("/time")
def current_time():
    return jsonify(datetime.datetime.now().strftime("%H:%M:%S"))

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("text")
    question = request.body.get("text")
    response = answer(pdf_text, question)
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
