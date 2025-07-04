<h1 align="center">🏥 MedAI Assistant</h1>

<p align="center">
  <img src="https://img.shields.io/badge/built%20with-Streamlit-ff4b4b?logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue.svg">
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN%20%7C%20YOLO-ff6f00">
  <img src="https://img.shields.io/badge/license-MIT-green">
</p>

<p align="center"><b>Note:</b> This project is for educational and prototyping purposes only.<br>Not intended for clinical use without proper validation.</p>

<hr>

<h2>🚀 Overview</h2>

<p>
MedAI Assistant is an interactive platform that combines <strong>Retrieval-Augmented Generation (RAG)</strong> for intelligent medical document Q&A with <strong>deep learning–based image classification</strong> to detect fractures and COVID-19 from X-ray images.  
It provides doctors, researchers, and students with an intuitive interface to explore medical documents, analyze X-rays, and get context-aware answers.
</p>

<hr>

<h2>✨ Features</h2>

<ul>
  <li>✅ <strong>RAG-Based Document QA</strong>: Upload medical PDFs, analyze web URLs, or use the default medical database to get contextual answers to natural language questions.</li>
  <li>✅ <strong>Deep Learning–Based Image Analysis</strong>: 
    <ul>
      <li>🦴 <strong>Fracture Detection</strong>: Uses object detection to highlight fractures on X-rays.</li>
      <li>🫁 <strong>COVID-19 Detection</strong>: Classifies X-rays as COVID-19, Normal, or Pneumonia.</li>
    </ul>
  </li>
  <li>✅ <strong>Integrated Context</strong>: X-ray analysis results feed directly into document Q&A for richer multi-modal insights.</li>
  <li>✅ <strong>Modern Streamlit UI</strong>: Responsive with gradients, glassmorphism cards, and conversational chat bubbles.</li>
</ul>

<hr>

<h2>🏗️ Project Structure</h2>

<pre>
MedAI-Assistant/
├── app.py
├── functions.py
├── best.pt
├── covid_xray_detection.h5
├── Medical_PDF.pdf
├── requirements.txt
├── .env
└── README.md
</pre>

<hr>

<h2>⚙️ Installation</h2>

<h4>1️⃣ Clone the repository and set up virtual environment</h4>

<pre><code>
git clone https://github.com/yourusername/MedAI-Assistant.git
cd MedAI-Assistant
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
</code></pre>

<h4>2️⃣ Configure environment variables</h4>

Create a <code>.env</code> file with:

<pre><code>
GROQ_API=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
</code></pre>

<hr>

<h2>🚀 Running the App</h2>

<pre><code>
streamlit run app.py
</code></pre>

Visit <code>http://localhost:8501</code> to start exploring.

<hr>

<h2>💡 Usage Guide</h2>

<h3>🔍 Medical Document Q&A</h3>
<ul>
  <li>Select 📚 <strong>Medical Documents</strong> in the sidebar.</li>
  <li>Choose:
    <ul>
      <li>📄 Upload PDFs</li>
      <li>🌐 Enter URLs (one per line)</li>
      <li>📋 Load the provided database</li>
    </ul>
  </li>
  <li>Click 🚀 <strong>Process</strong> and ask questions in the chat input below.</li>
</ul>

<h3>🩻 X-ray Analysis</h3>
<ul>
  <li>Select 🔍 <strong>X-ray Analysis</strong> in the sidebar.</li>
  <li>Choose:
    <ul>
      <li>🦴 Fracture Detection</li>
      <li>🫁 COVID-19 Detection</li>
    </ul>
  </li>
  <li>Upload an image, click 🔍 <strong>Analyze X-ray</strong>, view results and use them in Q&A.</li>
</ul>

<hr>

<h2>🔬 Under the Hood</h2>

<table>
<tr><th>Component</th><th>Description</th></tr>
<tr><td>RAG QA</td><td>Embeds and stores document chunks in Pinecone, retrieves context for answering medical questions.</td></tr>
<tr><td>Fracture Detection</td><td>YOLO object detection loads from <code>best.pt</code> and marks fractures.</td></tr>
<tr><td>COVID Detection</td><td>CNN model loaded from <code>covid_xray_detection.h5</code> predicts COVID-19, Normal, or Pneumonia.</td></tr>
</table>

<hr>

<h2>🚑 Troubleshooting</h2>

<ul>
  <li><strong>Pinecone slow to initialize:</strong> Index creation may take ~10-20 sec on first run.</li>
  <li><strong>Model files missing:</strong> Ensure <code>best.pt</code> and <code>covid_xray_detection.h5</code> are present.</li>
  <li><strong>Streamlit glitches:</strong> Try clearing cache:
    <pre><code>streamlit cache clear</code></pre>
  </li>
</ul>

<hr>

<h2>🚀 Future Enhancements</h2>

<ul>
  <li>Integrate MRI/CT scan analysis.</li>
  <li>Add attention visualizations for explainable AI.</li>
  <li>Implement user authentication and data security features.</li>
  <li>Deploy on Docker/Kubernetes for cloud scale.</li>
</ul>

<hr>

<h2>🤝 Contributing</h2>

<p>We welcome contributions! Fork this repo, create a branch, and submit a pull request. 🙌</p>

<hr>

<h2>📜 License</h2>

<p>This project is under the MIT License.<br>
⚠️ For research and educational use only. Not for real-world diagnosis without regulatory validation.</p>

<hr>

<h2>🙌 Acknowledgments</h2>

<p>Thanks to the open-source community for frameworks and models powering this project!</p>
e.
