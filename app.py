import gradio as gr
from langchain_groq import ChatGroq
import os
from langgraph.graph import StateGraph, START, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
import time
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

class State(TypedDict):
    query: str
    is_safe: bool
    is_relevant: bool
    company_description: str
    answer: str
    vectorstoredb: Chroma

class checker_class(BaseModel):
    is_relevant: bool = Field(description="Check whether the given query is relevant to the company.")

def invoke_llm(query):
    llm = ChatGroq(model='llama-3.3-70b-versatile')
    try:
        res = llm.invoke(query)
    except:
        time.sleep(60)
        res = llm.invoke(query)
    return res.content
    
def invoke_relevance_checker_llm(query):
    llm = ChatGroq(model='gemma2-9b-it')
    checker_llm = llm.with_structured_output(checker_class)
    try:
        res = checker_llm.invoke([HumanMessage(content=query)])
    except:
        time.sleep(60)
        res = checker_llm.invoke([HumanMessage(content=query)])
    return res.is_relevant

def safety_checker(state:State):
    llm = ChatGroq(model='meta-llama/llama-guard-4-12b')
    query = state['query']
    res = llm.invoke(query)
    if res.content == 'safe':
        return {'is_safe':True}
    else:
        return {'is_safe':False, 'answer':"<SAFETY CHECKER> That prompt was harmful, please try something else"}
        
def relevance_checker(state:State):
    prompt = "You are a lenient relevance-checking assistant. You will be given a user query and a company description. Your job is to decide whether the query is relevant to the company.\n✅ Approve most queries that are even loosely related.\n🚫 Only reject queries that are **clearly unrelated** or have **no connection at all**.\n\n"
    prompt += f"\nQuery: {state['query']}"
    prompt += f"\nDescription: {state['company_description']}"
    res = invoke_relevance_checker_llm(prompt)
    return {'is_relevant':res, 'answer':"Sorry! That doesn't seem to be relevant to us, please try something else."}

def agent(state:State):
    relevant_text = ""
    search_docs = state['vectorstoredb'].similarity_search(state['query'])
    for chunk in search_docs:
        relevant_text += f"\n{chunk.page_content}"
    prompt = f"You have to answer this query: {state['query']} based only on the following information: {relevant_text}. Reply only with the answer."
    try:
        res = invoke_llm(prompt)
    except:
        time.sleep(60)
        res = invoke_llm(prompt)
    finally:
        return {'answer':res}

def safety_assigner(state:State):
    if state['is_safe']:
        return 'relevant'
    else:
        return 'END'

def relevant_assigner(state:State):
    if state['is_relevant']:
        return 'Agent'
    else:
        return 'END'

def chat(query, vect, dec):
    yield gr.update(visible=True), ""
    mess = {'query':query, 'vectorstoredb': vect, 'company_description': dec}
    res = graph.invoke(mess)
    yield gr.update(visible=False), res['answer']

def setter(pdf_file, description, company_name):
    yield gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "", "", ""
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    consise_pdf = docs[1].page_content if len(docs) > 1 else docs[0].page_content
    consise_pdf = consise_pdf[:5555]
    full_pdf = ""
    for content in docs:
        full_pdf += f"\n{content.page_content}"
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = splitter.split_text(full_pdf)
    vector_db = Chroma.from_texts(chunks, embeddings)    
    prompt = "You are a company description generator assistant. "
    prompt += "You will be given the name of a company, a short description provided by the owner, "
    prompt += "and additional content extracted from a company file (such as a brochure or document). "
    prompt += "Using this information, generate a concise and professional 3–4 line description of the company. Also, reply in markdown\n\n"
    prompt += f"Company Name: {company_name}\n"
    prompt += f"Owner's Description: {description}\n"
    prompt += f"File Content: {consise_pdf}\n"
    prompt += "Final Description:"
    response = invoke_llm(prompt)
    yield gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), response, response, vector_db

builder = StateGraph(State)

builder.add_node("Safety Checker", safety_checker)
builder.add_node("Relevance Checker", relevance_checker)
builder.add_node("Agent", agent)

builder.add_edge(START, "Safety Checker")
builder.add_conditional_edges("Safety Checker", safety_assigner, {'relevant':"Relevance Checker", 'END': END})
builder.add_conditional_edges("Relevance Checker", relevant_assigner, {'Agent':"Agent", 'END':END})
builder.add_edge("Agent",END)

graph = builder.compile()

with gr.Blocks(css=".section {margin-bottom: 20px;}") as ui:

    vectorstore_db = gr.State()
    company_generated_description = gr.State()
    
    # 🌀 CSS + HTML animation injection
    header = gr.HTML("""
    <style>
        .fade-in {
            animation: fadeIn 1.2s ease-in;
        }
        .slide-up {
            animation: slideUp 0.8s ease-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to   { opacity: 1; }
        }
        @keyframes slideUp {
            from { transform: translateY(20px); opacity: 0; }
            to   { transform: translateY(0); opacity: 1; }
        }
    </style>
    <div class='fade-in'>
        <h1 style="text-align:center; font-size: 2.4em;">👋 Welcome to Your Personalized AI Agent Demo ✨</h1>
        <p style="text-align:center; font-size: 1.2em;">🚀 Automate marketing, save time, and scale smartly using AI Agents</p>
    </div>
    """, visible=True)

    with gr.Column(visible=True) as setup_page:
        
        with gr.Group(elem_classes=["slide-up", "section"]):
            gr.Markdown("### 💼 What’s the name of your company/service?")
            company_name = gr.Textbox(lines=1, placeholder="e.g., SwiftSync AI")

        with gr.Group(elem_classes=["slide-up", "section"]):
            gr.Markdown("### 📝 Tell us briefly what your company does:")
            company_desc = gr.Textbox(lines=3, placeholder="We provide AI-driven automation tools...")

        with gr.Group(elem_classes=["slide-up", "section"]):
            gr.Markdown("### 📄 Got a business PDF? Upload it here to make your AI Agent smarter:")
            pdf_file = gr.File(file_types=[".pdf"], label="Upload your PDF")

        with gr.Group(elem_classes=["slide-up"]):
            setup_submit = gr.Button("✨ Build My Agent Now")

    with gr.Column(visible=False) as processing_page:
        processing_msg = gr.HTML("""
        <style>
        @keyframes spin {
            0%   { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes fade {
            0%, 100% { opacity: 0.2; }
            50% { opacity: 1; }
        }
        .loader {
            border: 6px solid #e0e0e0;
            border-top: 6px solid #00bcd4;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            box-shadow: 0 0 10px rgba(0,188,212,0.4);
        }
        .processing-text {
            font-size: 1.1em;
            margin-top: 15px;
            font-weight: 500;
            color: #555;
            animation: fade 2s infinite ease-in-out;
        }
        </style>
    
        <div style="display: flex; flex-direction: column; align-items: center; margin-top: 40px;">
            <div class="loader"></div>
            <div class="processing-text">🧠 Building your AI Agent...</div>
        </div>
        """, visible=True)

    with gr.Column(visible=False) as agent_page:
        # Header Section
        gr.HTML("""
        <style>
            .title-box {
                text-align: center;
                padding: 15px 0;
                background: linear-gradient(90deg, #007bff 0%, #00c2ff 100%);
                color: white;
                border-radius: 12px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            }
            .info-card {
                background: #f9f9f9;
                border-left: 4px solid #007bff;
                padding: 12px 20px;
                border-radius: 8px;
                font-size: 15px;
                margin-bottom: 20px;
                color: #333;
            }
            .query-area {
                padding: 20px;
                border-radius: 12px;
                background: #fff;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }
            .footer-note {
                text-align: center;
                color: #888;
                font-size: 13.5px;
                padding: 15px 0;
                margin-top: 20px;
            }
        </style>
    
        <div class="title-box">
            <h1>🧠 Your Personalized AI Agent</h1>
            <p style="margin-top: -10px;">Supercharged for Safety, Relevance, and Results</p>
        </div>
        """)
        gr.HTML("""
    <style>
    .built-by-card {
        margin-top: 30px;
        padding: 15px;
        background: #f0f4ff;
        color: #333;
        text-align: center;
        border-radius: 12px;
        font-size: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .built-by-card:hover {
        box-shadow: 0 4px 14px rgba(0,0,0,0.1);
        background: #e6f0ff;
    }
    </style>
    
    <div class="built-by-card">
        🚀 Built with ❤️ by <strong>Darsh Tayal</strong>
    </div>
    """, visible = True)

    
        # Company Description
        comp_descri = gr.Markdown("")
    
        # Agent Info Features
        gr.HTML("""
        <div class="info-card">
            ✅ This agent uses a <strong>relevance checker</strong> to block off-topic questions.<br>
            🔒 It also runs a <strong>safety filter</strong> to protect users from harmful content.<br>
            🕒 <em>Saving your time while keeping things secure.</em>
        </div>
        """)
    
        # Query Section
        gr.HTML("<div class='query-area'>")
        gr.Markdown("### 💬 Ask something related to your business/service:")
        query = gr.Textbox(lines=2, placeholder="e.g., What are the top 3 features of our service?")
        agent_submit = gr.Button("🚀 Submit Query")
        loading_spinner = gr.HTML("""
    <style>
        @keyframes spin {
            0%   { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .loader {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #00bcd4;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        .loading-text {
            margin-top: 8px;
            color: #666;
            font-size: 14px;
            animation: pulse 1.8s infinite ease-in-out;
        }
        @keyframes pulse {
            0%, 100% { opacity: 0.4; }
            50% { opacity: 1; }
        }
    </style>
    <div style="display:flex; flex-direction:column; align-items:center; margin-top: 10px;" id="spinner">
        <div class="loader"></div>
        <div class="loading-text">Thinking... generating magic ✨</div>
    </div>
""", visible=False)

        answer = gr.TextArea(label='🤖 AI Response', lines=4, interactive=False)
        gr.HTML("</div>")  # Close .query-area div
    
        # Footer CTA
        gr.HTML("""
        <div class="footer-note">
            💡 This was just a general demo. Want a version tailored to your business?<br>
            👉 Email me at <strong>darshtayal8@gmail.com</strong><br>
            📈 We can connect this agent to whatsapp, or any other marketing channel you use<br>
            ⚙️ Start automating, or get left behind.
        </div>
        """)
    setup_submit.click(fn=setter, inputs=[pdf_file, company_desc, company_name], outputs=[setup_page, processing_page, agent_page, header, comp_descri, company_generated_description, vectorstore_db])
    agent_submit.click(fn=chat, inputs=[query, vectorstore_db, company_generated_description], outputs=[loading_spinner, answer])
ui.launch()
