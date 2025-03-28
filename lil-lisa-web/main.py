from flask import Flask, request, jsonify, Response
import os
import re
import uuid
import json
import traceback
import requests
from dotenv import load_dotenv, dotenv_values
import logging
import time
from html import escape

# Load environment variables from the specified .env file
load_dotenv('lil-lisa-web.env')

lil_lisa_env = dotenv_values('lil-lisa-web.env')

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Set the base URL for API requests, defaulting to localhost if not specified
BASE_URL = os.getenv("LIL_LISA_SERVER_URL", lil_lisa_env.get("LIL_LISA_SERVER_URL"))

if not BASE_URL:
    raise ValueError("LIL_LISA_SERVER_URL environment variable is not set")
else:
    BASE_URL = BASE_URL.rstrip('/')  # Remove trailing slash to avoid endpoint issues
    logger.info(f"Using LIL_LISA_SERVER_URL: {BASE_URL}")

# Initialize the Flask application
app = Flask(__name__)

# Function to generate a unique session ID
def generate_unique_session_id() -> str:
    """Generate a unique session ID using UUID version 4."""
    return str(uuid.uuid4())

# Function to apply formatting to individual lines (for bold text and links)
def apply_formatting(text):
    """Apply formatting to convert markdown-like syntax to HTML while preserving text after ':'."""
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\$\$\s*(.+?)\s*\$\$\s*\$\$\s*(https?://\S+)\s*\$\$', r'<a href="\2">\1</a>', text)
    text = re.sub(r'(https?://\S+)', r'<a href="\1">\1</a>', text)
    return text

# Function to format LLM responses into HTML without special code block styling.
def format_response(response: str) -> str:
    """
    Convert LLM text to HTML by transforming markdown-like lists, bold text, links, and headings.
    Any occurrence of triple backticks (```) is removed. If a line becomes empty after removal,
    it is skipped. This way, text originally inside the backticks is displayed normally.
    """
    lines = response.split("\n")
    formatted = ""
    in_list = False
    for line in lines:
        # Remove all triple backticks if present
        if "```" in line:
            line = line.replace("```", "")
            if not line.strip():
                continue  # Skip line if nothing remains
        if line.strip().startswith("###"):
            heading = line.strip()[3:].strip()
            formatted += f"<h3>{heading}</h3>"
        elif line.strip().startswith("-") or line.strip().startswith("*"):
            if not in_list:
                formatted += "<ul>"
                in_list = True
            item = line.strip()[1:].strip()
            formatted_item = apply_formatting(item)
            formatted += f"<li>{formatted_item}</li>"
        else:
            if in_list:
                formatted += "</ul>"
                in_list = False
            formatted_line = apply_formatting(line)
            formatted += formatted_line + "<br>"
    if in_list:
        formatted += "</ul>"
    return formatted

# Generator to stream text with a typing delay.
def stream_with_delay(text, char_delay=0.005):
    """
    Generator that yields HTML tags immediately and text characters with a delay.
    """
    i = 0
    while i < len(text):
        if text[i] == "<":
            j = text.find(">", i)
            if j == -1:
                j = len(text) - 1
            tag = text[i:j+1]
            yield tag
            i = j + 1
        else:
            yield text[i]
            time.sleep(char_delay)
            i += 1

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Handle non-streaming chat requests."""
    data = request.get_json()
    session_id = data.get("session_id")
    product = data.get("product")
    nl_query = data.get("query")
    locale = data.get("locale", "en-US")
    is_expert = data.get("is_expert", False)
    
    if not session_id:
        session_id = generate_unique_session_id()

    try:
        invoke_url = f"{BASE_URL}/invoke/"
        params = {
            "session_id": session_id,
            "locale": locale,
            "product": product,
            "nl_query": nl_query,
            "is_expert_answering": is_expert
        }
        resp = requests.post(invoke_url, params=params, timeout=60)
        resp.raise_for_status()
        formatted_answer = format_response(resp.text)
    except Exception as e:
        traceback.print_exc()
        formatted_answer = f"Internal error: {str(e)}"

    return jsonify({"answer": formatted_answer, "session_id": session_id})

@app.route("/api/stream_chat", methods=["POST"])
def api_stream_chat():
    """Handle streaming chat requests with persistent session ID and typing delay."""
    data = request.get_json()
    session_id = data.get("session_id")
    product = data.get("product")
    nl_query = data.get("query")
    locale = data.get("locale", "en-US")
    is_expert = data.get("is_expert", False)

    if not session_id:
        session_id = generate_unique_session_id()
        
    try:
        stream_url = f"{BASE_URL}/invoke_stream/"
        params = {
            "session_id": session_id,
            "locale": locale,
            "product": product,
            "nl_query": nl_query,
            "is_expert_answering": is_expert
        }
        r = requests.post(stream_url, params=params, stream=True, timeout=60)
        r.raise_for_status()

        def generate():
            in_list = False
            buffer = ""
            for chunk in r.iter_content(chunk_size=128):
                if chunk:
                    buffer += chunk.decode("utf-8")
                    lines = buffer.split("\n")
                    for line in lines[:-1]:
                        # Remove triple backticks if present in the line
                        if "```" in line:
                            line = line.replace("```", "")
                            if not line.strip():
                                continue
                        if line.strip().startswith("###"):
                            heading = line.strip()[3:].strip()
                            yield from stream_with_delay(f"<h3>{heading}</h3>")
                        elif line.strip().startswith("-") or line.strip().startswith("*"):
                            if not in_list:
                                yield from stream_with_delay("<ul>")
                                in_list = True
                            item = line.strip()[1:].strip()
                            formatted_item = apply_formatting(item)
                            yield from stream_with_delay("<li>")
                            yield from stream_with_delay(formatted_item)
                            yield from stream_with_delay("</li>")
                        else:
                            if in_list:
                                yield from stream_with_delay("</ul>")
                                in_list = False
                            formatted_line = apply_formatting(line)
                            yield from stream_with_delay(formatted_line)
                            yield from stream_with_delay("<br>")
                    buffer = lines[-1]
            if buffer:
                if "```" in buffer:
                    buffer = buffer.replace("```", "")
                if buffer.strip().startswith("###"):
                    heading = buffer.strip()[3:].strip()
                    yield from stream_with_delay(f"<h3>{heading}</h3>")
                elif buffer.strip().startswith("-") or buffer.strip().startswith("*"):
                    if not in_list:
                        yield from stream_with_delay("<ul>")
                    item = buffer.strip()[1:].strip()
                    formatted_item = apply_formatting(item)
                    yield from stream_with_delay("<li>")
                    yield from stream_with_delay(formatted_item)
                    yield from stream_with_delay("</li>")
                    yield from stream_with_delay("</ul>")
                else:
                    if in_list:
                        yield from stream_with_delay("</ul>")
                    yield from stream_with_delay(apply_formatting(buffer))
                    yield from stream_with_delay("<br>")

        response = Response(generate(), mimetype="text/html")
        response.headers['X-Session-ID'] = session_id
        return response
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "session_id": session_id})

@app.route("/api/thumbsup", methods=["POST"])
def api_thumbsup():
    """Record a thumbs-up endorsement for a session."""
    data = request.get_json()
    session_id = data.get("session_id")
    try:
        thumb_url = f"{BASE_URL}/record_endorsement/"
        params = {"session_id": session_id, "is_expert": False, "thumbs_up": True}
        resp = requests.post(thumb_url, params=params, timeout=10)
        resp.raise_for_status()
        result = resp.text
    except Exception as e:
        traceback.print_exc()
        result = f"Error: {str(e)}"
    return jsonify({"result": result})

@app.route("/api/thumbsdown", methods=["POST"])
def api_thumbsdown():
    """Record a thumbs-down endorsement for a session."""
    data = request.get_json()
    session_id = data.get("session_id")
    try:
        thumb_url = f"{BASE_URL}/record_endorsement/"
        params = {"session_id": session_id, "is_expert": False, "thumbs_up": False}
        resp = requests.post(thumb_url, params=params, timeout=10)
        resp.raise_for_status()
        result = resp.text
    except Exception as e:
        traceback.print_exc()
        result = f"Error: {str(e)}"
    return jsonify({"result": result})

@app.route("/", methods=["GET"])
def index():
    """Serve the main chatbot interface."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>RAG Pipeline Chatbot</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
  <style>
    body {
      margin: 0;
      padding: 0;
      font-family: 'Montserrat', sans-serif;
      background: linear-gradient(135deg, #f0f4f8, #d9e2ec);
      color: #334e68;
    }
    header.header {
      background: #1d2d44;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 30px;
      height: 70px;
    }
    nav.nav {
      display: flex;
      align-items: center;
    }
    nav.nav img.logo-image {
      height: 60px;
      width: auto;
      margin-right: 25px;
    }
    nav.nav a {
      color: #ffffff;
      margin-right: 25px;
      text-decoration: none;
      font-weight: 500;
      transition: color 0.3s;
    }
    nav.nav a:hover {
      color: #ffc107;
    }
    button.request-trial {
      background: #ffc107;
      border: none;
      border-radius: 30px;
      padding: 10px 25px;
      font-weight: 700;
      color: #1d2d44;
      transition: background 0.3s;
    }
    button.request-trial:hover {
      background: #e0a800;
    }
    .main-content {
      max-width: 1200px;
      margin: 40px auto;
      padding: 20px;
      text-align: center;
    }
    section.hero h1 {
      font-size: 2.5rem;
      font-weight: 700;
      margin-bottom: 20px;
      color: #1d2d44;
    }
    section.hero p {
      font-size: 1.2rem;
      line-height: 1.6;
      margin-bottom: 40px;
      color: #334e68;
    }
    #chrome-container {
      transition: all 0.3s ease-in-out;
      background: #ffffff;
      box-shadow: 0 10px 30px rgba(0,0,0,0.1);
      border-radius: 15px;
      padding: 40px;
      margin-bottom: 40px;
    }
    .chrome-tabs {
      display: flex;
      align-items: center;
      justify-content: flex-start;
      margin-bottom: 30px;
      flex-wrap: wrap;
    }
    .chrome-tab {
      position: relative;
      background: #e9ecef;
      color: #495057;
      margin: 10px;
      padding: 10px 20px;
      border-radius: 30px;
      border: none;
      cursor: pointer;
      transition: background 0.3s, transform 0.2s;
      font-size: 1rem;
    }
    .chrome-tab.active {
      background: #007bff;
      color: #fff;
      transform: translateY(-3px);
    }
    .chrome-tab:hover {
      background: #ced4da;
    }
    .chrome-tab .tooltip {
      visibility: hidden;
      opacity: 0;
      width: 280px;
      background-color: rgba(0, 0, 0, 0.85);
      color: #fff;
      text-align: left;
      border-radius: 8px;
      padding: 10px;
      position: absolute;
      z-index: 1000;
      bottom: 110%;
      left: 50%;
      transform: translateX(-50%);
      transition: opacity 0.3s;
      font-size: 0.85rem;
      box-shadow: 0px 2px 10px rgba(0,0,0,0.3);
    }
    .chrome-tab:hover .tooltip {
      visibility: visible;
      opacity: 1;
    }
    .chrome-tab .tooltip::after {
      content: "";
      position: absolute;
      top: 100%;
      left: 50%;
      margin-left: -5px;
      border-width: 5px;
      border-style: solid;
      border-color: rgba(0, 0, 0, 0.85) transparent transparent transparent;
    }
    #global-language-container {
      margin-left: auto;
      margin-right: 20px;
      display: flex;
      align-items: center;
    }
    #global-language-container label {
      margin-left: 10px;
      margin: 0;
      font-size: 1rem;
      color: #495057;
    }
    #start-new-chat-button {
      background: #17a2b8;
      border: none;
      border-radius: 30px;
      padding: 10px 20px;
      font-weight: 600;
      color: #fff;
      cursor: pointer;
      transition: background 0.3s, transform 0.2s;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      display: none;
      margin-left: 15px;
    }
    #start-new-chat-button:hover {
      background: #0056b3;
      transform: translateY(-2px);
    }
    .grok-container {
      background: #ffffff;
      border-radius: 15px;
      padding: 25px;
      position: relative;
      box-shadow: 0 5px 15px rgba(0,0,0,0.1);
      margin-bottom: 20px;
      text-align: left;
    }
    .grok-messages {
      margin-bottom: 10px;
    }
    .grok-message {
      margin: 12px 0;
      padding: 15px 20px;
      border-radius: 15px;
      font-size: 0.95rem;
      line-height: 1.4;
      word-wrap: break-word;
    }
    .user-message {
      background: #f1f3f5;
      text-align: right;
    }
    .assistant-message {
      background: #e9ecef;
      text-align: left;
      border-left: 4px solid #17a2b8;
      color: #343a40;
    }
    .system-message {
      background: #fff3cd;
      text-align: center;
      font-style: italic;
    }
    #common-input-area {
      position: fixed;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%);
      width: 85%;
      display: none;
      gap: 10px;
      justify-content: center;
      align-items: center;
      z-index: 1000;
    }
    #common-input-area input[type="text"] {
      flex: 1;
      padding: 15px 20px;
      border: 2px solid #ced4da;
      border-radius: 50px;
      outline: none;
      font-size: 1rem;
      transition: border-color 0.3s;
    }
    #common-input-area input[type="text"]:focus {
      border-color: #17a2b8;
    }
    #common-input-area button {
      padding: 15px 25px;
      background: #17a2b8;
      color: #fff;
      border: none;
      border-radius: 50px;
      cursor: pointer;
      font-size: 1rem;
      transition: background 0.3s;
    }
    #common-input-area button:hover {
      background: #138496;
    }
    .loading-text {
      margin-top: 5px;
      font-size: 0.9em;
    }
    .feedback-container { 
      text-align: right; 
      margin-top: 5px; 
    }
    .thumb-btn { 
      background: none; 
      border: none; 
      cursor: pointer; 
      font-size: 1.2em; 
      margin-left: 5px;
    }
    .feedback-popup {
      position: fixed;
      bottom: 80px;
      right: 20px;
      background: #333;
      color: #fff;
      padding: 10px 20px;
      border-radius: 5px;
      opacity: 0.9;
      font-size: 0.9em;
      z-index: 2000;
    }
    #doc-links-container {
      margin: 20px 0;
      font-size: 1rem;
      text-align: left;
    }
    .checkbox-container {
      position: relative;
      display: inline-block;
      margin-left: 10px;
    }
    .checkbox-container .tooltip {
      visibility: hidden;
      background-color: rgba(0, 0, 0, 0.8);
      color: #fff;
      text-align: center;
      border-radius: 4px;
      padding: 5px;
      position: absolute;
      bottom: 125%;
      left: 50%;
      transform: translateX(-50%);
      opacity: 0;
      transition: opacity 0.3s ease-in-out 0.5s;
      white-space: nowrap;
      z-index: 1000;
    }
    .checkbox-container:hover .tooltip {
      visibility: visible;
      opacity: 1;
    }
    .checkbox-container input[type="checkbox"] {
      transform: scale(1.2);
      vertical-align: middle;
      margin: 0;
      cursor: pointer;
    }
    #chat-area {
      display: none; 
    }
    pre {
      background-color: #f4f4f4;
      padding: 10px;
      border-radius: 5px;
      font-family: monospace;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <header class="header">
    <nav class="nav">
      <img src="https://raw.githubusercontent.com/Vezingg/rocknbot/main/LilLisa_Server/src/RadiantLogic.png"
           alt="Radiant Logic Logo"
           class="logo-image">
      <a href="https://marketplace.radiantlogic.com/" target="_blank">Marketplace</a>
      <a href="https://support.radiantlogic.com/hc/en-us" target="_blank">Support</a>
    </nav>
    <button class="request-trial"><a href="https://www.radiantlogic.com/request-a-trial/" target="_blank" style="text-decoration: none; color: #1d2d44;">Request Trial</a></button>
  </header>
  <div class="main-content">
    <section class="hero">
      <h1>Welcome to Radiant Logic Documentation Assistant</h1>
      <p>
        Ask your product-related questions below. Answers are drawn from our product documentation and include citations. Provide the product version for version-specific responses.
      </p>
    </section>
    <div id="chrome-container">
      <div class="chrome-tabs" id="product-tabs">
        <button class="chrome-tab" id="tab-ida" onclick="selectProduct('IDA')">
          Identity Analytics
          <span class="tooltip">
            <strong>Identity Analytics</strong><br>
            Respond to audit recommendations and automate controls to ensure compliance.
          </span>
        </button>
        <button class="chrome-tab" id="tab-iddm" onclick="selectProduct('IDDM')">
          Identity Data Management
          <span class="tooltip">
            <strong>Identity Data Management</strong><br>
            Manage identity data across silos with context-driven views.
          </span>
        </button>
        <button class="chrome-tab" id="tab-eoc" onclick="selectProduct('EOC')">
          Environment Operations Center
          <span class="tooltip">
            <strong>Environment Operations Center</strong><br>
            Monitor and manage your SaaS applications from a unified control plane.
          </span>
        </button>
        <div id="global-language-container">
          <label>
            <input type="checkbox" id="global-language-checkbox" onchange="handleGlobalLanguageChange(this)"> French
          </label>
        </div>
        <button id="start-new-chat-button" onclick="newChat(currentProduct)">Start New Chat</button>
      </div>
      <div id="chat-area"></div>
      <div id="common-input-area">
        <input type="text" id="query-input" placeholder="Type Your Question Here..." autocomplete="off">
        <button id="send-btn" onclick="sendQuery()">Send Query</button>
        <div class="checkbox-container" id="start-new-chat-checkbox-container" style="display: inline-block;">
          <input type="checkbox" id="start-new-chat-checkbox">
          <span class="tooltip">Start New Chat</span>
        </div>
      </div>
      <button id="scroll-to-bottom-btn" style="display: none; position: fixed; bottom: 80px; right: 20px; z-index: 1000; background: #17a2b8; color: #fff; border: none; border-radius: 50%; width: 40px; height: 40px; font-size: 20px; cursor: pointer;">⬇️</button>
      <div id="doc-links-container"></div>
    </div>
  </div>
  <script>
    let currentProduct = null;
    const sessions = {
      "IDA": { sessionId: null, firstQuerySent: false },
      "IDDM": { sessionId: null, firstQuerySent: false },
      "EOC": { sessionId: null, firstQuerySent: false }
    };
    const chatTemplates = {
      "IDA": `<div id="chat-IDA" class="grok-container">
                <div id="messages-IDA" class="grok-messages"></div>
                <div class="scroll-padding" style="height: 100px;"></div>
              </div>`,
      "IDDM": `<div id="chat-IDDM" class="grok-container">
                <div id="messages-IDDM" class="grok-messages"></div>
                <div class="scroll-padding" style="height: 100px;"></div>
              </div>`,
      "EOC": `<div class="static-info">
                <p>The Environment Operations Center does not support chatbot functionality yet. Please refer to the documentation below for details.</p>
              </div>`
    };
    let autoScrollEnabled = true;

    function ensureChatContainer() {
      const chatArea = document.getElementById('chat-area');
      if (!document.getElementById('chat-' + currentProduct)) {
        chatArea.innerHTML = chatTemplates[currentProduct];
      }
      chatArea.style.display = "block";
    }

    function appendMessage(content, sender) {
      if (!currentProduct) return;
      const messagesDiv = document.getElementById('messages-' + currentProduct);
      if (!messagesDiv) {
        ensureChatContainer();
      }
      const realMessagesDiv = document.getElementById('messages-' + currentProduct);
      if (!realMessagesDiv) return;

      const msgDiv = document.createElement('div');
      msgDiv.classList.add('grok-message');
      if (sender === 'user') {
        msgDiv.classList.add('user-message');
        msgDiv.innerText = content;
      } else if (sender === 'assistant') {
        msgDiv.classList.add('assistant-message');
        msgDiv.innerHTML = content;
        if (!content.startsWith("[Error")) {
          const feedbackDiv = document.createElement('div');
          feedbackDiv.classList.add('feedback-container');
          feedbackDiv.innerHTML = '<button class="thumb-btn up">👍</button>' +
                                  '<button class="thumb-btn down">👎</button>';
          msgDiv.appendChild(feedbackDiv);
          const upBtn = feedbackDiv.querySelector('.thumb-btn.up');
          const downBtn = feedbackDiv.querySelector('.thumb-btn.down');
          upBtn.addEventListener('click', function() {
            if (feedbackDiv.classList.contains("locked")) return;
            feedbackDiv.classList.add("locked");
            upBtn.disabled = true;
            downBtn.disabled = true;
            submitFeedback('thumbsup');
          });
          downBtn.addEventListener('click', function() {
            if (feedbackDiv.classList.contains("locked")) return;
            feedbackDiv.classList.add("locked");
            downBtn.disabled = true;
            upBtn.disabled = true;
            submitFeedback('thumbsdown');
          });
        }
      } else {
        msgDiv.classList.add('system-message');
        msgDiv.innerText = content;
      }
      realMessagesDiv.appendChild(msgDiv);
      if (autoScrollEnabled) {
        window.scrollTo(0, document.body.scrollHeight);
      }
    }

    function selectProduct(product) {
      currentProduct = product;
      document.querySelectorAll('.chrome-tab').forEach(tab => tab.classList.remove('active'));
      const clickedTab = document.getElementById('tab-' + product.toLowerCase());
      if (clickedTab) clickedTab.classList.add('active');
      const chatArea = document.getElementById('chat-area');
      const docLinksContainer = document.getElementById('doc-links-container');
      const inputArea = document.getElementById('common-input-area');
      const startNewChatBtn = document.getElementById('start-new-chat-button');
      const newChatCheckboxContainer = document.getElementById('start-new-chat-checkbox-container');

      if (product === "EOC") {
        chatArea.innerHTML = chatTemplates["EOC"];
        chatArea.style.display = "block";
        docLinksContainer.innerHTML =
          'See admin documentation <a href="https://developer.radiantlogic.com/eoc/latest/#0" target="_blank">Admin Documentation</a><br>' +
          'Environment Operations Center provides a unified control plane for managing your RadiantOne SaaS applications.';
        if (inputArea) inputArea.style.display = "none";
        if (startNewChatBtn) startNewChatBtn.style.display = "none";
        if (newChatCheckboxContainer) newChatCheckboxContainer.style.display = "none";
        document.getElementById('scroll-to-bottom-btn').style.display = 'none';
        return;
      }

      chatArea.innerHTML = "";
      chatArea.style.display = "none";
      docLinksContainer.innerHTML = "";
      if (inputArea) inputArea.style.display = "flex";
      if (startNewChatBtn) startNewChatBtn.style.display = "none";
      if (newChatCheckboxContainer) newChatCheckboxContainer.style.display = "inline-block";
      if (product === "IDDM") {
        docLinksContainer.innerHTML =
          'See admin documentation <a href="https://developer.radiantlogic.com/idm/v8.1/#0" target="_blank">Admin Documentation</a><br>' +
          'See developer documentation <a href="https://developer.radiantlogic.com/idm/v8.1/#1" target="_blank">Developer Documentation</a><br>' +
          'Simplify the management of identity data across distributed silos and speed up your identity projects.';
      } else if (product === "IDA") {
        docLinksContainer.innerHTML =
          'See user guide <a href="https://developer.radiantlogic.com/ia/iap-3.2/#0" target="_blank">User Guide</a><br>' +
          'See developer documentation <a href="https://developer.radiantlogic.com/ia/descartes/#1" target="_blank">Developer Documentation</a><br>' +
          'Quickly respond to audit recommendations and automate controls for compliant access rights.';
      }
      document.getElementById('scroll-to-bottom-btn').style.display = 'none';
    }

    function newChat(product) {
      sessions[product].sessionId = null;
      sessions[product].firstQuerySent = false;
      const chatArea = document.getElementById('chat-area');
      chatArea.innerHTML = "";
      chatArea.style.display = "none";
      const startNewChatBtn = document.getElementById('start-new-chat-button');
      if (startNewChatBtn) startNewChatBtn.style.display = "none";
      document.getElementById('scroll-to-bottom-btn').style.display = 'none';
    }

    async function sendQuery() {
      if (!currentProduct) return;
      const queryInput = document.getElementById('query-input');
      if (!queryInput || !queryInput.value.trim()) return;
      
      autoScrollEnabled = true;
      
      const newChatCheckbox = document.getElementById('start-new-chat-checkbox');
      if (newChatCheckbox && newChatCheckbox.checked) {
          newChat(currentProduct);
          newChatCheckbox.checked = false;
      }
      const userQuery = queryInput.value.trim();
      appendMessage(userQuery, 'user');
      queryInput.value = '';

      appendMessage("", 'assistant');
      const messagesDiv = document.getElementById('messages-' + currentProduct);
      const assistantMsgDiv = messagesDiv.lastChild;
      
      let locale = 'en-US';
      const langCheckbox = document.getElementById('global-language-checkbox');
      if (langCheckbox && langCheckbox.checked) locale = 'fr-FR';
      
      document.getElementById('scroll-to-bottom-btn').style.display = 'block';
      
      try {
        const response = await fetch('/api/stream_chat', {
          method: 'POST',
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
             session_id: sessions[currentProduct].sessionId,
             product: currentProduct,
             query: userQuery,
             locale: locale,
             is_expert: false
          })
        });
        
        const sessionId = response.headers.get('X-Session-ID');
        if (sessionId) {
            sessions[currentProduct].sessionId = sessionId;
        }
        if (!sessions[currentProduct].firstQuerySent) {
          sessions[currentProduct].firstQuerySent = true;
          document.getElementById('start-new-chat-button').style.display = 'inline-block';
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let assistantResponse = "";
        
        function readStream() {
          reader.read().then(({done, value}) => {
              if (done) {
                  if (!assistantResponse.startsWith("[Error")) {
                      const feedbackDiv = document.createElement('div');
                      feedbackDiv.classList.add('feedback-container');
                      feedbackDiv.innerHTML = '<button class="thumb-btn up">👍</button>' +
                                              '<button class="thumb-btn down">👎</button>';
                      assistantMsgDiv.appendChild(feedbackDiv);
                      const upBtn = feedbackDiv.querySelector('.thumb-btn.up');
                      const downBtn = feedbackDiv.querySelector('.thumb-btn.down');
                      upBtn.addEventListener('click', function() {
                          if (feedbackDiv.classList.contains("locked")) return;
                          feedbackDiv.classList.add("locked");
                          upBtn.disabled = true;
                          downBtn.disabled = true;
                          submitFeedback('thumbsup');
                      });
                      downBtn.addEventListener('click', function() {
                          if (feedbackDiv.classList.contains("locked")) return;
                          feedbackDiv.classList.add("locked");
                          downBtn.disabled = true;
                          upBtn.disabled = true;
                          submitFeedback('thumbsdown');
                      });
                  }
                  document.getElementById('scroll-to-bottom-btn').style.display = 'none';
                  return;
              }
              const chunk = decoder.decode(value, {stream: true});
              assistantResponse += chunk;
              assistantMsgDiv.innerHTML = assistantResponse;
              if (autoScrollEnabled) {
                window.scrollTo(0, document.body.scrollHeight);
              }
              readStream();
          }).catch(error => console.error("Error reading stream", error));
        }
        readStream();
      } catch (err) {
          console.error("Error:", err);
          appendMessage("[Error: Failed to get response]", "assistant");
          document.getElementById('scroll-to-bottom-btn').style.display = 'none';
      }
    }

    async function submitFeedback(type) {
      try {
        const response = await fetch('/api/' + type, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessions[currentProduct].sessionId })
        });
        await response.json();
      } catch (err) {
        console.error('Feedback submission error:', err);
      }
      showFeedbackPopup();
    }

    function showFeedbackPopup() {
      const popup = document.createElement('div');
      popup.classList.add('feedback-popup');
      popup.innerText = 'Thank you for your feedback!';
      document.body.appendChild(popup);
      setTimeout(() => popup.remove(), 3000);
    }

    function handleGlobalLanguageChange(checkboxElem) {
      setCookie("languagePreference", checkboxElem.checked ? "fr-FR" : "en-US", 7);
    }

    function setCookie(name, value, days) {
      const d = new Date();
      d.setTime(d.getTime() + (days*24*60*60*1000));
      let expires = "expires="+ d.toUTCString();
      document.cookie = name + "=" + value + ";" + expires + ";path=/";
    }

    function getCookie(name) {
      let cname = name + "=";
      let decodedCookie = decodeURIComponent(document.cookie);
      let ca = decodedCookie.split(';');
      for(let i = 0; i < ca.length; i++) {
        let c = ca[i].trim();
        if (c.indexOf(cname) == 0) {
          return c.substring(cname.length, c.length);
        }
      }
      return "";
    }

    window.addEventListener('scroll', function() {
      if ((window.innerHeight + window.scrollY) < document.body.scrollHeight - 10) {
        autoScrollEnabled = false;
      }
    });

    window.onload = function() {
      const langPref = getCookie("languagePreference");
      const globalCheckbox = document.getElementById('global-language-checkbox');
      if (globalCheckbox) globalCheckbox.checked = (langPref === "fr-FR");
      selectProduct('IDDM');
      const queryInput = document.getElementById('query-input');
      if (queryInput) {
        queryInput.addEventListener('keypress', function(e) {
          if (e.key === 'Enter') {
            e.preventDefault();
            sendQuery();
          }
        });
      }
      document.getElementById('scroll-to-bottom-btn').addEventListener('click', function() {
        window.scrollTo(0, document.body.scrollHeight);
      });
    };
  </script>
</body>
</html>
"""

# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
