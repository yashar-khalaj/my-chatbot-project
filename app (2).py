from flask import Flask, render_template, request, jsonify
import openai
import json
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI as LLM

app = Flask(__name__)

# کلید API OpenAI خود را وارد کنید
openai.api_key = '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'

# ایجاد embedding و FAISS vector store با استفاده از OpenAIEmbeddings
def create_embedding(chunks, api_key):
    try:
        # استفاده از OpenAIEmbeddings برای ساخت embedding
        embedding = OpenAIEmbeddings(openai_api_key=api_key)
        # ایجاد FAISS vector store از متن‌ها
        vector_store = FAISS.from_texts(chunks, embedding)
        print(f"Vector store created successfully with {len(chunks)} chunks.")
        return vector_store
    except Exception as e:
        print(f"Error creating embedding and FAISS store: {e}")
        return None

# بارگذاری چانک‌های متنی
def load_chunks_from_json(json_filename):
    with open(json_filename, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return chunks

# بارگذاری چانک‌ها از فایل JSON
json_filename = "pdf_chunks.json"
loaded_chunks = load_chunks_from_json(json_filename)

# ایجاد FAISS vector store
api_key = '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'
vector_store = create_embedding(loaded_chunks, api_key)

# صفحه اصلی
@app.route("/")
def index():
    return render_template("index.html")

# پردازش سوال و پاسخ از چت‌بات
@app.route("/ask", methods=["POST"])
def ask():
    question = request.form["question"]  # گرفتن سوال از فرم
    print(f"Question received: {question}")

    if vector_store:
        # جستجو در FAISS vector store برای پیدا کردن نزدیکترین متن‌ها
        similar_doc = vector_store.similarity_search(question)

        # استفاده از مدل OpenAI برای پاسخ به سوال
        llm = LLM(openai_api_key=api_key)
        retriever = vector_store.as_retriever()
        chain = ConversationalRetrievalChain.from_llm(llm, retriever)

        # انجام پرسش و پاسخ
        response = chain({"question": question, "chat_history": []})
        
        answer = response["answer"]
    else:
        answer = "Error: No vector store available."

    return jsonify({"answer": answer})  # پاسخ را به صورت JSON برمی‌گرداند

if __name__ == "__main__":
    app.run(debug=True)
