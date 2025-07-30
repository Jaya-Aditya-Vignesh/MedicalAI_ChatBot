import React, { useState } from 'react';
import './ChatBox.css';

function ChatBox() {
  const [userInput, setUserInput] = useState('');
  const [chat, setChat] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!userInput.trim()) return;
    setChat([...chat, { sender: 'user', text: userInput }]);
    setLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:5000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userInput }),
      });

      const data = await response.json();
      setChat(prev => [...prev, { sender: 'bot', text: data.answer }]);
    } catch (err) {
      setChat(prev => [...prev, { sender: 'bot', text: 'Error connecting to backend.' }]);
    }

    setUserInput('');
    setLoading(false);
  };

  return (
    <div className="chat-box">
      <div className="chat-messages">
        {chat.map((msg, idx) => (
          <div key={idx} className={`chat-message ${msg.sender}`}>
            <span>{msg.text}</span>
          </div>
        ))}
        {loading && <div className="chat-message bot"><span>Typing...</span></div>}
      </div>
      <div className="chat-input">
        <input
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="Ask your medical question..."
        />
        <button onClick={handleSend}>Send</button>
      </div>
    </div>
  );
}

export default ChatBox;
