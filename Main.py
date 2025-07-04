import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv
import numpy as np
import tensorflow as tf
from datetime import datetime
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from tensorflow.keras.utils import img_to_array
from pinecone import Pinecone as pc, ServerlessSpec
from ultralytics import YOLO
from PIL import Image
from functions import data_load_url, data_load_pdf, text_split, huggingface_embeddings

load_dotenv(dotenv_path='.env')
GROQ_API = os.getenv('GROQ_API')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

st.set_page_config(layout="wide", page_title="MedAI Assistant", page_icon="🏥")
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .header { background: linear-gradient(90deg, #00c9ff 0%, #92fe9d 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center; }
    .chat-user { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0; margin-left: 2rem; }
    .chat-bot { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0; margin-right: 2rem; }
    .status-card { background: rgba(255, 255, 255, 0.1); padding: 1.5rem; border-radius: 12px; margin: 1rem 0; backdrop-filter: blur(10px); }
    .stButton > button { background: linear-gradient(45deg, #667eea, #764ba2); border: none; border-radius: 10px; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "image_context" not in st.session_state:
    st.session_state["image_context"] = "No image analysis performed"

st.markdown("""
<div class="header">
    <h1>🏥 MedAI Assistant</h1>
    <p>Advanced Medical AI with Document Analysis & X-ray Detection</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔧 Data Source")
    data_type = st.selectbox("Choose Analysis Type:", ["📚 Medical Documents", "🔍 X-ray Analysis"])

def setup_qa_system(data, index_name='data'):
    progress_bar = st.progress(0)
    for key in list(st.session_state.keys()):
        if key not in ["chat_history", "image_context"]:
            del st.session_state[key]
    progress_bar.progress(25)
    embeddings = huggingface_embeddings()
    vector = embeddings.embed_query('test')
    dimensions = len(vector)
    pine = pc(api_key=PINECONE_API_KEY)
    if index_name in pine.list_indexes().names():
        pine.delete_index(name=index_name)
    progress_bar.progress(50)
    cloud = os.environ.get('PINECONE_CLOUD', 'aws')
    region = os.environ.get('PINECONE_REGION', 'us-east-1')
    spec = ServerlessSpec(cloud=cloud, region=region)
    pine.create_index(name=index_name, dimension=dimensions, metric="cosine", spec=spec)
    while not pine.describe_index(index_name).status['ready']:
        time.sleep(1)
    progress_bar.progress(75)
    from langchain.docstore.document import Document
    documents = [Document(page_content=txt, metadata={}) for txt in data]
    text_chunks = text_split(documents, chunk_size=1000, chunk_overlap=100)
    Pinecone.from_documents(text_chunks, index_name=index_name, embedding=embeddings)
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    prompt_template = """Use the given context to answer the question. If you don't know the answer, say "I don't know".
Context: {context}
Question: {query}
Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])
    llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=GROQ_API, temperature=0.3)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 8}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    st.session_state['qa'] = qa
    progress_bar.progress(100)
    time.sleep(1)
    progress_bar.empty()
    st.success("✅ AI Assistant Ready!")

def analyze_fracture_xray(image_file):
    try:
        fracture_model = YOLO("best.pt")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image_file.save(temp_file.name)
        results = fracture_model.predict(source=temp_file.name, conf=0.01, save=False)
        boxes = results[0].boxes
        if len(boxes) > 0:
            detections = []
            for box in boxes:
                cls_id = int(box.cls[0].item())
                confidence = box.conf[0].item()
                detections.append(f"**{fracture_model.names[cls_id]}** (confidence: {confidence:.2f})")
            context = f"Fracture Analysis Results:\n" + "\n".join(detections)
            img_with_boxes = Image.fromarray(results[0].plot())
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(img_with_boxes, caption="🔍 Fracture Detection Results", width=400)
            st.success(f"✅ Detected {len(boxes)} fracture(s)")
            return context
        else:
            st.info("No fractures detected")
            return "No fractures detected in the X-ray image"
    except Exception as e:
        st.error(f"Error in fracture analysis: {str(e)}")
        return "Error occurred during fracture analysis"
    finally:
        try:
            os.unlink(temp_file.name)
        except:
            pass

def analyze_covid_xray(image_file):
    try:
        covid_model = tf.keras.models.load_model("covid_xray_detection.h5")
        image_file = image_file.resize((64, 64))
        img = img_to_array(image_file) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = covid_model.predict(img)
        covid_classes = ['COVID-19', 'Normal', 'Pneumonia']
        predicted_class = covid_classes[np.argmax(prediction)]
        confidence = np.max(prediction)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image_file, caption="📷 Uploaded X-ray", width=400)
        if predicted_class == 'COVID-19':
            st.error(f"⚠️ COVID-19 Detected (Confidence: {confidence:.2f})")
        elif predicted_class == 'Normal':
            st.success(f"✅ Normal Chest X-ray (Confidence: {confidence:.2f})")
        else:
            st.warning(f"⚠️ {predicted_class} Detected (Confidence: {confidence:.2f})")
        return f"COVID-19 Analysis Results:\nPrediction: {predicted_class}\nConfidence: {confidence:.2f}"
    except Exception as e:
        st.error(f"Error in COVID-19 analysis: {str(e)}")
        return "Error occurred during COVID-19 analysis"

col1, col2 = st.columns([3, 1])

with col1:
    if data_type == "📚 Medical Documents":
        st.markdown("### 📚 Medical Document Analysis")
        doc_option = st.radio("Choose document source:", ["📄 Upload PDF", "🌐 URL Analysis", "📋 Default Database"])
        if doc_option == "📄 Upload PDF":
            uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
            if uploaded_files and st.button("🚀 Process PDFs"):
                with st.spinner("Processing documents..."):
                    all_data = []
                    for file in uploaded_files:
                        data = data_load_pdf(file)
                        all_data.extend(data)
                    setup_qa_system(all_data, 'pdfurl')
        elif doc_option == "🌐 URL Analysis":
            urls_input = st.text_area("Enter URLs (one per line):")
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            if urls and st.button("🚀 Process URLs"):
                with st.spinner("Processing URLs..."):
                    data = data_load_url(urls)
                    setup_qa_system(data, 'pdfurl')
        elif doc_option == "📋 Default Database":
            if st.button("🚀 Load Default Database"):
                with st.spinner("Loading medical database..."):
                    loader = PyPDFLoader('Medical_PDF.pdf')
                    data = loader.load()
                    setup_qa_system(data, 'data')
    elif data_type == "🔍 X-ray Analysis":
        st.markdown("### 🔍 X-ray Analysis")
        xray_option = st.radio("Choose analysis type:", ["🦴 Fracture Detection", "🫁 COVID-19 Detection"])
        uploaded_file = st.file_uploader("Upload X-ray image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📷 Uploaded X-ray",  use_container_width=True)
            if st.button("🔍 Analyze X-ray"):
                positive_flag = False
                precautions = ""
                if xray_option == "🦴 Fracture Detection":
                    context = analyze_fracture_xray(image)
                    if "No fractures detected" not in context:
                        positive_flag = True
                        precautions = """Precautions for Fracture:
- Immobilize the affected area.
- Avoid putting weight on it.
- Use cold packs to reduce swelling.
- Seek immediate medical attention."""
                elif xray_option == "🫁 COVID-19 Detection":
                    context = analyze_covid_xray(image)
                    if "COVID-19" in context:
                        positive_flag = True
                        precautions = """Precautions for COVID-19:
- Isolate yourself from others.
- Wear a mask at all times.
- Monitor oxygen levels and temperature.
- Stay hydrated and rest.
- Contact healthcare services."""
                st.session_state['image_context'] = context
                if positive_flag:
                    st.success("⚠️ Positive case detected. AI assistant activated for precautions & follow-up questions.")
                    setup_qa_system([precautions], 'data')
                else:
                    st.info("✅ No abnormalities detected. AI not activated as there is no condition requiring follow-up.")

with col2:
    st.markdown("### 📊 System Status")
    if 'qa' in st.session_state:
        st.success("🤖 AI Ready")
    else:
        st.warning("⏳ Load Data")
    if st.session_state['chat_history']:
        st.metric("Questions Asked", len(st.session_state['chat_history']))

st.markdown("---")
st.markdown("## 💬 Ask Your Questions")
user_input = st.text_input("💭 Type your question here...", key="chat_input")
if st.button("🚀 Ask", type="primary") and user_input:
    if 'qa' in st.session_state:
        with st.spinner("🤔 Thinking..."):
            qa = st.session_state['qa']
            result = qa.invoke({"query": user_input})
            st.session_state['chat_history'].append({
                "question": user_input,
                "answer": result['result'],
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            st.session_state.chat_input = ""
            st.experimental_rerun()
    else:
        st.warning("⚠️ Please load data first or ensure a positive case was detected.")

if st.session_state['chat_history']:
    st.markdown("### 📝 Chat History")
    if st.button("🗑️ Clear History"):
        st.session_state['chat_history'] = []
        st.experimental_rerun()
    for i, chat in enumerate(reversed(st.session_state['chat_history'][-5:])):
        st.markdown(f"""<div class="chat-user"><strong>👤 You ({chat['timestamp']}):</strong><br>{chat['question']}</div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="chat-bot"><strong>🤖 MedAI:</strong><br>{chat['answer']}</div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""<div style="text-align: center; color: #888; padding: 1rem;">
    <p>🏥 MedAI Assistant - For Educational Purposes Only</p></div>""", unsafe_allow_html=True)
