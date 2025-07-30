import React, { useState, useEffect, useRef } from 'react';
import './App.css'; // Import the external CSS file

// --- Helper Components ---

const Spinner = () => <div className="spinner" aria-label="Loading..."></div>;
const Icon = ({ icon, className = '' }) => <span className={className} aria-hidden="true">{icon}</span>;

const ErrorDisplay = ({ message }) => {
  if (!message) return null;
  return (
    <div className="error-display">
      <Icon icon="⚠️" />
      <p><strong>Error:</strong> {message}</p>
    </div>
  );
};

// --- Core Components ---

const ChatBox = ({ messages, onSendMessage, isProcessing, isContextReady }) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = () => {
    if (input.trim() && !isProcessing) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') handleSend();
  };

  return (
    <div className="chatbox">
      <div className="chatbox-header">
        <div>
          <div className="chatbox-icon"><Icon icon="🩺" /></div>
          <h3>Medical Chat Assistant</h3>
        </div>
        <div className="status-indicator">
          <div className={`status-dot ${isContextReady ? 'ready' : 'waiting'}`}></div>
          <span className="status-text">{isContextReady ? 'Ready' : 'Waiting'}</span>
        </div>
      </div>
      <div className="chatbox-content">
        <div className="message-area">
          {messages.map((msg, index) => (
            <div key={index} className={`message-bubble ${msg.sender}`}><p>{msg.text}</p></div>
          ))}
          {isProcessing && (
            <div className="message-bubble ai">
              <div className="typing-indicator"><span></span><span></span><span></span></div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <div className="chat-input-container">
          <input
            type="text"
            placeholder={isContextReady ? "Ask a question..." : "Please set up a data source first."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            className="chat-input"
            disabled={!isContextReady || isProcessing}
          />
          <button onClick={handleSend} className="send-button" disabled={!isContextReady || isProcessing}>
            {isProcessing ? <Spinner /> : <Icon icon="➤" />}
          </button>
        </div>
      </div>
    </div>
  );
};

function App() {
  // UI State
  const [section, setSection] = useState(null);
  const [chatOption, setChatOption] = useState(null);
  const [xrayType, setXrayType] = useState(null);

  // Data State
  const [file, setFile] = useState(null);
  const [url, setUrl] = useState('');
  const [uploadedImageUrl, setUploadedImageUrl] = useState(null);
  const [resultImageUrl, setResultImageUrl] = useState(null);
  const [xrayResultText, setXrayResultText] = useState(null);

  // Logic State
  const [isProcessing, setIsProcessing] = useState(false);
  const [isContextReady, setIsContextReady] = useState(false);
  const [messages, setMessages] = useState([]);
  const [error, setError] = useState(null);

  // --- API Communication ---

  const setupQAEngine = async (type, data) => {
    setIsProcessing(true);
    setError(null);
    setMessages([]);
    try {
      const body = new FormData();
      body.append('type', type);
      if (type === 'pdf' && data) body.append('file', data);
      if (type === 'url' && data) body.append('url', data);

      const response = await fetch('http://localhost:5000/api/setup-qa', {
        method: 'POST',
        credentials: 'include',
        body: body,
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || `Server error: ${response.status}`);
      }
      const result = await response.json();
      if (result.status === 'ready') {
        setIsContextReady(true);
        setMessages([{ sender: 'ai', text: 'Context is ready. You can now ask questions.' }]);
      } else {
        throw new Error('Failed to set up QA engine.');
      }
    } catch (err) {
      setError(err.message);
      setIsContextReady(false);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSendMessage = async (query) => {
    setMessages(prev => [...prev, { sender: 'user', text: query }]);
    setIsProcessing(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:5000/api/ask', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || `Server error: ${response.status}`);
      }
      const data = await response.json();
      setMessages(prev => [...prev, { sender: 'ai', text: data.answer }]);
    } catch (err) {
      setError(err.message);
      setMessages(prev => [...prev, { sender: 'ai', text: `Sorry, I ran into an error: ${err.message}` }]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleXrayUpload = async (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;

    setFile(uploadedFile);
    setIsProcessing(true);
    setError(null);
    setXrayResultText(null);
    setResultImageUrl(null);
    setUploadedImageUrl(URL.createObjectURL(uploadedFile));

    try {
      const formData = new FormData();
      formData.append('file', uploadedFile);
      const endpoint = xrayType === 'chest' ? 'covid' : 'fracture';

      const res = await fetch(`http://localhost:5000/api/xray/${endpoint}`, {
        method: 'POST',
        credentials: 'include',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || `Server error: ${res.status}`);
      }

      const data = await res.json();
      console.log("Received data from server:", data);

      if (data.error) throw new Error(data.error);

      if (endpoint === 'fracture') {
        if (data.resultText && data.resultImage) {
          setXrayResultText(data.resultText);
          setResultImageUrl(`data:image/jpeg;base64,${data.resultImage}`);
        } else {
          throw new Error("Received an invalid or unexpected response from the server for fracture analysis.");
        }
      } else {
        if (data.result) {
          setXrayResultText(data.result);
          setResultImageUrl(null);
        } else {
           throw new Error("Received an invalid or unexpected response from the server for chest X-ray analysis.");
        }
      }

    } catch (err) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  // --- Event Handlers & Reset Logic ---

  const handlePDFSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setupQAEngine('pdf', selectedFile);
    }
  };

  const handleUrlSubmit = (e) => {
    e.preventDefault();
    if (url.trim()) setupQAEngine('url', url);
  };

  const handleDefaultSelect = () => {
    setChatOption('default');
    setupQAEngine('default');
  };

  const resetState = () => {
    setFile(null);
    setUrl('');
    setIsContextReady(false);
    setMessages([]);
    setError(null);
    setXrayResultText(null);
    setUploadedImageUrl(null);
    setResultImageUrl(null);
  };

  const resetToHome = () => {
    setSection(null);
    setChatOption(null);
    setXrayType(null);
    resetState();
  };

  const resetSection = () => {
    if (section === 'chat') setChatOption(null);
    if (section === 'xray') setXrayType(null);
    resetState();
  };

  return (
    <div className="App enhanced-ui">
      <div className="header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon"><Icon icon="❤️" /></div>
            <h1 className="logo-text">MedAI Assistant</h1>
          </div>
          {(section || chatOption || xrayType) && (
            <button onClick={section && !chatOption && !xrayType ? resetToHome : resetSection} className="back-button">
              <Icon icon="←" /> Back
            </button>
          )}
        </div>
      </div>

      <div className="main-content">
        {!section && (
          <div className="welcome-section">
            <h2 className="welcome-title">Welcome to Your AI Medical Assistant</h2>
            <p className="welcome-description">Get instant medical insights through our advanced AI-powered chatbot or analyze X-ray images with cutting-edge computer vision technology.</p>
            <div className="main-cards">
              <div onClick={() => setSection('chat')} className="main-card chat">
                <div className="card-icon chat-icon"><Icon icon="🩺" /></div>
                <h3 className="card-title">Medical Chatbot</h3>
                <p className="card-description">Chat with our AI for medical questions, document analysis, and health guidance.</p>
              </div>
              <div onClick={() => setSection('xray')} className="main-card xray">
                <div className="card-icon xray-icon"><Icon icon="🩻" /></div>
                <h3 className="card-title">X-ray Analysis</h3>
                <p className="card-description">Upload X-ray images for AI-powered analysis and diagnostic insights.</p>
              </div>
            </div>
          </div>
        )}

        {section === 'chat' && !chatOption && (
          <div className="options-container">
            <div className="section-title">
              <h2>Choose Your Chat Method</h2>
              <p>Select how you'd like to interact with the medical AI assistant.</p>
            </div>
            <div className="option-list">
              <div onClick={() => setChatOption('pdf')} className="option-item">
                <div className="option-content"><div className="option-icon"><Icon icon="📄" /></div><div className="option-text"><h3>Upload PDF Document</h3><p>Upload medical documents, reports, or research papers for analysis.</p></div></div>
              </div>
              <div onClick={handleDefaultSelect} className="option-item">
                <div className="option-content"><div className="option-icon"><Icon icon="🧠" /></div><div className="option-text"><h3>Use Default Knowledge</h3><p>Chat directly with our AI using its built-in medical knowledge base.</p></div></div>
              </div>
              <div onClick={() => setChatOption('url')} className="option-item">
                <div className="option-content"><div className="option-icon"><Icon icon="🔗" /></div><div className="option-text"><h3>Import from URL</h3><p>Provide a URL to medical content for analysis and discussion.</p></div></div>
              </div>
            </div>
          </div>
        )}

        {section === 'chat' && chatOption && (
          <div className="content-section">
            <div className="content-card">
              <div className="content-header chat-header"><h2>Medical Chat Assistant</h2></div>
              <div className="content-body">
                <ErrorDisplay message={error} />
                {!isContextReady && !isProcessing && (
                  <div className="upload-section">
                    {chatOption === 'pdf' && (
                      <div>
                        <label className="upload-label">Upload PDF Document</label>
                        <input type="file" accept="application/pdf" onChange={handlePDFSelect} className="hidden" id="pdf-upload" />
                        <label htmlFor="pdf-upload" className="upload-area">
                          <Icon icon="📤" className="upload-icon" />
                          <div className="upload-text"><p>{file ? file.name : 'Click to upload PDF'}</p><p>Your analysis will begin automatically.</p></div>
                        </label>
                      </div>
                    )}
                    {chatOption === 'url' && (
                      <div>
                        <label className="upload-label">Enter URL</label>
                        <form onSubmit={handleUrlSubmit} className="url-input-form">
                          <input type="url" placeholder="Enter medical document URL..." className="url-input" value={url} onChange={(e) => setUrl(e.target.value)} />
                          <button type="submit" className="submit-button" disabled={!url.trim()}>Submit</button>
                        </form>
                      </div>
                    )}
                  </div>
                )}
                {isProcessing && !isContextReady && (<div className="upload-area uploading"><Spinner /><span>Setting up QA Engine...</span></div>)}
                <ChatBox messages={messages} onSendMessage={handleSendMessage} isProcessing={isProcessing} isContextReady={isContextReady} />
              </div>
            </div>
          </div>
        )}

        {section === 'xray' && !xrayType && (
          <div className="options-container">
            <div className="section-title">
              <h2>Choose X-ray Analysis Type</h2>
              <p>Select the type of X-ray analysis you need.</p>
            </div>
            <div className="xray-grid">
              <div onClick={() => setXrayType('chest')} className="xray-card">
                <div className="xray-card-icon chest-icon"><Icon icon="🫁" /></div>
                <h3 className="card-title">Chest X-ray Analysis</h3>
                <p className="card-description">COVID-19 detection and respiratory analysis.</p>
              </div>
              <div onClick={() => setXrayType('fracture')} className="xray-card">
                <div className="xray-card-icon fracture-icon"><Icon icon="🦴" /></div>
                <h3 className="card-title">Fracture Detection</h3>
                <p className="card-description">Bone fracture identification and assessment.</p>
              </div>
            </div>
          </div>
        )}

        {section === 'xray' && xrayType && (
          <div className="content-section">
            <div className="content-card">
              <div className={`content-header xray-header ${xrayType}`}>
                <h2>{xrayType === 'chest' ? 'Chest X-ray Analysis' : 'Fracture Detection'}</h2>
              </div>
              <div className="content-body">
                <ErrorDisplay message={error} />
                <div className="upload-section">
                  <label className="upload-label">Upload X-ray Image</label>
                  <div>
                    <input type="file" accept="image/*" onChange={handleXrayUpload} className="hidden" id="xray-upload" disabled={isProcessing} />
                    <label htmlFor="xray-upload" className={`upload-area ${isProcessing ? 'uploading' : ''}`}>
                      {isProcessing ? (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', color: '#cbd5e1' }}><Spinner /><span>Analyzing image...</span></div>
                      ) : (
                        <><Icon icon="📤" className="upload-icon" /><div className="upload-text"><p>{file ? file.name : 'Click to upload X-ray image'}</p><p>JPEG, PNG, or other image formats</p></div></>
                      )}
                    </label>
                  </div>
                </div>

                {/* --- NEW FRACTURE ANALYSIS LAYOUT --- */}
                {xrayType === 'fracture' && (uploadedImageUrl || resultImageUrl) && (
                  <>
                    <div className="analysis-grid">
                      {uploadedImageUrl && (
                        <div className="image-display">
                          <div className="image-display-header">Uploaded Image</div>
                          <img src={uploadedImageUrl} alt="Uploaded X-ray" className="uploaded-image" />
                        </div>
                      )}
                      {resultImageUrl && (
                        <div className="image-display">
                          <div className="image-display-header">Result Image</div>
                          <img src={resultImageUrl} alt="Analyzed X-ray" className="uploaded-image" />
                        </div>
                      )}
                    </div>
                    {xrayResultText && (
                      <div className="result-section">
                        <div className="result-header">
                          <Icon icon="🔬" className="result-icon" />
                          <div className="result-content">
                            <h3>Analysis Details</h3>
                            <p>{xrayResultText}</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                )}

                {/* --- ORIGINAL COVID/OTHER ANALYSIS LAYOUT --- */}
                {xrayType !== 'fracture' && xrayResultText && (
                  <div className="result-section">
                    <div className="result-header">
                      <Icon icon="🔬" className="result-icon" />
                      <div className="result-content">
                        <h3>Analysis Result</h3>
                        <p>{xrayResultText}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;