import json
import logging
import os
import re
import traceback
import uuid

import requests
from dotenv import dotenv_values, load_dotenv
from flask import Flask, Response, jsonify, render_template, request

# Load environment variables
load_dotenv('lil-lisa-web.env')
lil_lisa_env = dotenv_values('lil-lisa-web.env')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

BASE_URL = os.getenv("LIL_LISA_SERVER_URL", lil_lisa_env.get("LIL_LISA_SERVER_URL"))
if not BASE_URL:
    raise ValueError("LIL_LISA_SERVER_URL environment variable is not set")
else:
    BASE_URL = BASE_URL.rstrip('/')
    logger.info(f"Using LIL_LISA_SERVER_URL: {BASE_URL}")

app = Flask(__name__)

def generate_unique_session_id() -> str:
    return str(uuid.uuid4())

@app.route("/api/chat_cot", methods=["POST"])
def api_chat_cot():
    """
    Proxies the request to /invoke_stream_html and streams the chunks back to the browser as chunked text.
    """
    data = request.get_json()
    session_id = data.get("session_id") or generate_unique_session_id()
    product = data.get("product")
    nl_query = data.get("query")
    locale = data.get("locale", "en-US")
    is_expert = data.get("is_expert", False)

    invoke_url = f"{BASE_URL}/invoke_stream_html/"
    params = {
        "session_id": session_id,
        "locale": locale,
        "product": product,
        "nl_query": nl_query,
        "is_expert_answering": is_expert
    }
    try:
        resp = requests.post(invoke_url, params=params, timeout=90, stream=True)
        resp.raise_for_status()

        def generate():
            for chunk in resp.iter_content(chunk_size=128):
                if chunk:
                    yield chunk
        return Response(generate(), mimetype='text/html')
    except requests.exceptions.Timeout as e:
        logger.error(f"Request timed out: {str(e)}")
        return "Request timed out", 504, {'Content-Type': 'text/plain'}
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error while retrieving CoT response: {str(e)}")
        return f"Failed to communicate with the server: {str(e)}", 503, {'Content-Type': 'text/plain'}
    except Exception as e:
        logger.error(f"Internal error while retrieving CoT response: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Internal server error: {e}", 500, {'Content-Type': 'text/plain'}

@app.route("/api/with_nodes", methods=["POST"])
def api_with_nodes():
    """
    Proxies the request to /invoke_with_nodes and returns the final answer with top 10 nodes.
    The markdown response is formatted as HTML for consistent browser rendering.
    """
    data = request.get_json()
    session_id = data.get("session_id") or generate_unique_session_id()
    product = data.get("product")
    nl_query = data.get("query")
    locale = data.get("locale", "en-US")
    is_expert = data.get("is_expert", False)
    include_nodes = data.get("include_nodes", True)

    invoke_url = f"{BASE_URL}/invoke_with_nodes/"
    params = {
        "session_id": session_id,
        "locale": locale,
        "product": product,
        "nl_query": nl_query,
        "is_expert_answering": is_expert,
        "include_nodes": include_nodes
    }
    try:
        # Increased timeout from 90 to 180 seconds
        resp = requests.post(invoke_url, params=params, timeout=180)
        resp.raise_for_status()
        
        # Get JSON response from the server
        response_data = resp.json()
        return jsonify(response_data)
    except requests.exceptions.Timeout as e:
        logger.error(f"Request timed out: {str(e)}")
        # Custom error message for timeout
        return jsonify({"error": "Unable to get the response. We are trying to fix the server. Please try again later."}), 504
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error while retrieving nodes response: {str(e)}")
        return jsonify({"error": f"Failed to communicate with the server: {str(e)}"}), 503
    except Exception as e:
        logger.error(f"Internal error while retrieving nodes response: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Internal server error: {e}"}), 500

@app.route("/api/thumbsdown", methods=["POST"])
def api_thumbsdown():
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
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
