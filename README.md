# MediScan AI: Medical Chatbot & X-ray Analysis Assistant

## 🧠 Overview

MediScan AI is a full-stack web application designed to provide intelligent assistance for medical queries and X-ray analysis. It features a user-friendly interface built with React, a robust backend powered by Flask, and leverages large language models (LLMs) for conversational AI and specialized models for medical image analysis.

This application aims to streamline access to medical information and preliminary diagnostic support, serving as a helpful tool for users seeking quick insights.

---

## 🌟 Features

### 💬 Medical Chatbot Assistance
- **Multiple Upload Options**: Interact with the chatbot by uploading PDF documents, using a default text input, or providing a URL for content analysis.
- **LLM-Powered Conversations**: Engage in natural language conversations with an AI assistant for medical inquiries (via Gemini API).

### 🩻 X-ray Analysis
- **Chest X-ray Analysis**: Upload chest X-ray images for automated analysis.
- **Fracture X-ray Analysis**: Upload X-ray images for fracture detection and analysis.
- **Result Display**: View preliminary analysis results directly within the application.

### 🖥️ Responsive UI & Architecture
- **React + Tailwind CSS** frontend for a clean and intuitive interface.
- **Modular Project Structure**: Clear separation between frontend (React) and backend (Flask).

---

## 📁 Folder Structure

```
Project/
├── backend/
│   ├── backend/                  # Internal Flask modules
│   ├── documents/                # Medical documents (e.g., PDFs)
│   ├── models/                   # ML models (.pt, .h5)
│   ├── .env                      # Environment variables
│   └── app.py                    # Flask application entry
├── frontend/
│   ├── node_modules/             # Node dependencies
│   ├── public/                   # Static assets
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatBox.css
│   │   │   └── ChatBox.js
│   │   ├── App.js, index.js, etc.
│   └── package.json              # Frontend scripts and dependencies
├── MedicalAI_ChatBot/            # Optional module for chatbot logic
└── README.md                     # This file
```

---

## 🛠️ Technologies Used

### Frontend
- React.js
- Tailwind CSS

### Backend
- Flask
- Python
- Flask-CORS
- python-dotenv
- google-generativeai (Gemini API)
- Machine Learning Libraries: TensorFlow/Keras or PyTorch

---

## ⚙️ Setup Instructions

### ✅ Prerequisites
Ensure you have the following installed:
- Node.js & npm
- Python 3.8+
- pip

---

### 📥 1. Clone the Repository

```bash
git clone https://github.com/your-username/Project.git
cd Project
```

---

### 🐍 2. Backend Setup

```bash
cd backend
python -m venv venv

# Windows:
.\venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

#### Example `requirements.txt`

```
Flask
Flask-CORS
python-dotenv
google-generativeai
tensorflow
scikit-learn
opencv-python
```

#### Create `.env` in `backend/`

```env
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

---

### 💻 3. Frontend Setup

```bash
cd ../frontend
npm install
```

#### Canvas Environment Notes
- If deployed in a Canvas environment, global Firebase variables are injected.
- Outside of Canvas, Firebase runs in mock mode (see `App.js`).

---

### 🚀 Running the Application

#### ▶️ Start Backend

```bash
cd Project/backend
# Activate venv
python app.py
# Runs on http://127.0.0.1:5000
```

#### ▶️ Start Frontend

```bash
cd Project/frontend
npm start
# Opens at http://localhost:3000
```

---

## 🧪 Usage

### 💬 Medical Chatbot
- Navigate to **"Medical Chatbot Assistance"**
- Choose: PDF Upload / Default Upload / URL Upload
- Ask questions in the chat interface
- Responses powered by Gemini-based LLMs

### 🩻 X-ray Analysis
- Navigate to **"X-ray Analysis"**
- Select **Chest X-ray** or **Fracture X-ray**
- Upload an image and view the AI-generated result

---

## 🤝 Contribution

Feel free to fork this repository, submit pull requests, or report issues.

