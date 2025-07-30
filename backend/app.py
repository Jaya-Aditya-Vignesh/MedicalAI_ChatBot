import os
import time
import tempfile
import uuid
import re  # Import the regular expression module
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import base64 # Import for image encoding
import io # Import for in-memory byte streams
# --- Load environment variables from .env file at the very start ---
load_dotenv()
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = ''


# --- Prevent unintended GPU usage by HuggingFace transformers ---
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# LangChain and Pinecone Imports
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import Pinecone
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Imports for X-Ray Analysis ---
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from ultralytics import YOLO
from PIL import Image

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================

# Initialize Flask App
app = Flask(__name__)

# --- IMPORTANT ---
# Set a secret key for session management. This is required for CORS with credentials.
# In a production environment, use a more secure, randomly generated key from environment variables.
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a-very-secure-and-random-default-secret-key')

# --- More explicit CORS configuration for local development ---
# This explicitly allows the necessary headers and methods that the browser
# pre-flight check (`OPTIONS`) will verify before sending the actual request.
CORS(
    app,
    supports_credentials=True,
    origins=["http://localhost:3000", "http://localhost:3000"],  # Directly specifying all allowed frontend origins
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)

# --- Environment Variable Check ---
# Ensure necessary API keys are set in your environment after loading .env
if 'PINECONE_API_KEY' not in os.environ:
    raise ValueError("PINECONE_API_KEY environment variable not set.")
if 'GROQ_API' not in os.environ:
    raise ValueError("GROQ_API_KEY environment variable not set.")


# ==============================================================================
# 2. DOCUMENT PROCESSING & EMBEDDING UTILS
# ==============================================================================

def data_load_pdf(file_storage):
    """
    Loads a PDF file from a Flask FileStorage object, saves it to a temporary
    directory, and processes it with DirectoryLoader.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = secure_filename(file_storage.filename)
        temp_file_path = os.path.join(temp_dir, filename)
        file_storage.save(temp_file_path)

        loader = DirectoryLoader(temp_dir, glob="*.pdf", loader_cls=PyPDFLoader, silent_errors=True)
        documents = loader.load()
        return documents


def data_load_url(urls):
    """Loads content from a list of URLs."""
    if isinstance(urls, str):
        urls = [urls]
    loader = UnstructuredURLLoader(urls=urls, ssl_verify_certificate=False)
    return loader.load()


def text_split(data, chunk_size=1000, chunk_overlap=100):
    """Splits loaded documents into smaller chunks for processing."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(data)


def huggingface_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Initializes and returns HuggingFace embeddings, explicitly setting the device
    to 'cpu' to prevent torch distributed/GPU errors.
    """
    model_kwargs = {'device': 'cpu'}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )


# ==============================================================================
# 3. X-RAY ANALYSIS LOGIC
# ==============================================================================

def analyze_covid_xray(image_file):
    """Analyzes a chest X-ray image for COVID-19 using a TensorFlow model."""
    try:
        # Load the pre-trained model
        model = tf.keras.models.load_model("models/covid_xray_detection.h5")

        # Preprocess the image
        image_file = image_file.resize((64, 64))
        img = img_to_array(image_file) / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        prediction = model.predict(img)
        covid_classes = ['COVID-19', 'Normal', 'Pneumonia']
        predicted_class = covid_classes[np.argmax(prediction)]
        confidence = np.max(prediction)

        return {"prediction": predicted_class, "confidence": float(confidence)}
    except Exception as e:
        return {"error": str(e)}


def analyze_fracture_xray(image_file):
    """
    Analyzes an X-ray for fractures using a YOLO model and returns detections
    with bounding box coordinates.
    """
    try:
        # Load the pre-trained YOLO model
        model = YOLO("models/best.pt")

        os.makedirs("backend", exist_ok=True)

        # Fixed filename
        save_path = os.path.join("backend", "xray_input.png")

        # Save the uploaded image with the same name
        image_file.save(save_path)

        # Run YOLO prediction on this image
        results = model.predict(source=save_path, conf=0.01, save=False)
        annotated_image_bgr = results[0].plot()
        annotated_image_rgb = annotated_image_bgr[..., ::-1]
        result_image = Image.fromarray(annotated_image_rgb)

        # Save the PIL image to an in-memory buffer
        buf = io.BytesIO()
        result_image.save(buf, format="JPEG")
        # Get the Base64 string of the image
        base64_string = base64.b64encode(buf.getvalue()).decode("utf-8")
        boxes = results[0].boxes
        if len(boxes) == 0:
            return {"resultText": "No fractures detected", "resultImage": base64_string}

        detections = []
        for box in boxes:
            class_name = model.names[int(box.cls[0])]
            confidence = box.conf[0]
            coords = [int(c) for c in box.xyxy[0].tolist()]
            detections.append(f"{class_name} (confidence: {confidence:.2f}) at coordinates {coords}")

        return {"resultText": "\n".join(detections), "resultImage": base64_string}

    except Exception as e:
        return {"error": f"Error in fracture analysis: {str(e)}"}


# ==============================================================================
# 4. CORE QA ENGINE LOGIC
# ==============================================================================

def setup_qa_system(data, index_name_prefix='doc'):

    index_name = f"{index_name_prefix}"

    embeddings = huggingface_embeddings()
    pine_client = PineconeClient(api_key=os.getenv('PINECONE_API_KEY'))
    existing_indexes = [index_info["name"] for index_info in pine_client.list_indexes()]
    if index_name in ['pdf','url']:
        if index_name in existing_indexes:
            pine_client.delete_index(index_name)

        vector_dim = len(embeddings.embed_query("test query"))
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        pine_client.create_index(
            name=index_name,
            dimension=vector_dim,
            metric="cosine",
            spec=spec
        )

        while not pine_client.describe_index(index_name).status['ready']:
            time.sleep(1)

        text_chunks = text_split(data)
        Pinecone.from_documents(text_chunks, embeddings, index_name=index_name)

        return index_name
    else:
        if index_name in existing_indexes:
            return index_name

        vector_dim = len(embeddings.embed_query("test query"))
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        pine_client.create_index(
            name=index_name,
            dimension=vector_dim,
            metric="cosine",
            spec=spec
        )

        while not pine_client.describe_index(index_name).status['ready']:
            time.sleep(1)
        loader = PyPDFLoader('documents/Medical_data.pdf')
        data = loader.load()
        text_chunks = text_split(data)
        Pinecone.from_documents(text_chunks, embeddings, index_name=index_name)
        return index_name





def get_qa_chain(index_name):
    """
    Reconstructs the QA chain for a given query using an existing session index.
    """
    embeddings = huggingface_embeddings()

    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=os.getenv('GROQ_API'))

    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain


# ==============================================================================
# 5. FLASK API ENDPOINTS
# ==============================================================================

@app.route('/')
def index():
    """Root endpoint for a simple health check."""
    return "MedAI Backend is running!"


@app.route('/api/setup-qa', methods=['POST'])
def setup_qa_endpoint():
    """API endpoint to set up the QA engine."""
    try:
        source_type = request.form.get('type')
        documents = []
        index_prefix = 'default'

        if source_type == 'pdf':
            if 'file' not in request.files:
                return jsonify({"error": "No PDF file provided"}), 400
            file = request.files['file']
            documents = data_load_pdf(file)
            index_prefix = 'pdf'

        elif source_type == 'url':
            url = request.form.get('url')
            if not url:
                return jsonify({"error": "No URL provided"}), 400
            documents = data_load_url(url)
            index_prefix = 'url'

        elif source_type == 'default':
            default_content = "The MedAI Assistant is a helpful AI tool designed to provide information based on medical documents. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified doctor for any health concerns."
            documents = [Document(page_content=default_content)]

        else:
            return jsonify({"error": "Invalid source type specified"}), 400

        if not documents:
            return jsonify({"error": "Failed to load any content from the provided source."}), 500

        index_name = setup_qa_system(documents, source_type)
        session['qa_index_name'] = index_name

        return jsonify({"status": "ready", "message": f"QA engine ready with index '{index_name}'."})

    except Exception as e:
        app.logger.exception("An error occurred in /api/setup-qa")
        error_message = str(e) if app.debug else "An internal server error occurred."
        return jsonify({"error": error_message}), 500


@app.route('/api/ask', methods=['POST'])
def ask_question_endpoint():
    """API endpoint to ask a question."""
    try:
        data = request.get_json()
        query = data.get('query')
        index_name = session.get('qa_index_name')

        if not query:
            return jsonify({"error": "No query was provided"}), 400
        if not index_name:
            return jsonify({
                               "error": "Your session has expired or the QA engine was not set up. Please provide a document first."}), 400

        qa_chain = get_qa_chain(index_name)
        result = qa_chain.invoke({"query": query})

        return jsonify({"answer": result['result']})

    except Exception as e:
        app.logger.exception("An error occurred in /api/ask")
        error_message = str(e) if app.debug else "An internal server error occurred."
        return jsonify({"error": error_message}), 500


@app.route('/api/xray/<string:endpoint>', methods=['POST'])
def xray_analysis_endpoint(endpoint):
    """
    Live API endpoint for X-ray analysis.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        image_file = request.files['file']
        image = Image.open(image_file.stream)

        if endpoint == 'covid':
            # Call the COVID analysis function
            analysis_result = analyze_covid_xray(image)
            # The frontend expects a single 'result' string. We format the dict here.
            if 'error' in analysis_result:
                return jsonify({"result": f"Error: {analysis_result['error']}"})
            else:
                return jsonify({
                                   "result": f"Prediction: {analysis_result['prediction']} (Confidence: {analysis_result['confidence']:.2f})"})

        elif endpoint == 'fracture':
            # Call the fracture analysis function
            analysis_result = analyze_fracture_xray(image)
            if 'error' in analysis_result:
                return jsonify({"result": f"Error: {analysis_result['error']}"})
            else:
                return jsonify(analysis_result)  # This function already returns the desired format

        else:
            return jsonify({"error": "Invalid X-ray analysis endpoint"}), 404

    except Exception as e:
        app.logger.exception("An error occurred during X-ray analysis")
        return jsonify({"result": f"An unexpected error occurred: {str(e)}"}), 500


# ==============================================================================
# 6. APPLICATION RUNNER
# ==============================================================================

if __name__ == '__main__':
    # To run this application:
    # 1. Ensure you have the correct library versions installed.
    #    pip install python-dotenv
    #    pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
    #    pip install sentence-transformers==2.2.2 transformers==4.38.2
    #    pip install Flask Flask-Cors langchain langchain-community langchain-groq pinecone-client pypdf unstructured "unstructured[pdf]"
    #    pip install tensorflow ultralytics Pillow numpy
    # 2. Create a '.env' file in the same directory with your API keys:
    #    PINECONE_API_KEY='your_pinecone_api_key'
    #    GROQ_API_KEY='your_groq_api_key'
    # 3. Place your models in 'backend/models/covid_xray_detection.h5' and 'backend/models/best.pt'
    # 4. Run the script:
    #    python app.py
    app.run(port=5000, debug=True)
