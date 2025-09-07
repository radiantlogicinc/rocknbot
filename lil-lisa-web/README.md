# LilLisa Web Interface

A web application interface for the LilLisa Server providing a user-friendly frontend to interact with the Radiant Logic Documentation Assistant.

## Overview

LilLisa Web is a Flask-based web application that serves as the frontend for the LilLisa Server. It provides a user interface for users to interact with the AI documentation assistant, ask questions, and receive responses with relevant source information. The application streams responses in real-time and allows users to provide feedback on the responses and individual source chunks.

## Features

- **Real-time Streaming Responses**: Get answers as they're generated
- **Source Reference Display**: View the source documents used to generate responses
- **Feedback System**: Provide thumbs-up/thumbs-down feedback on both overall responses and individual source chunks
- **Multi-product Support**: Switch between different product documentation (IDA/IDDM)
- **Session Management**: Maintain conversation history across interactions

## Architecture

The application acts as a proxy between the user's browser and the LilLisa Server:

1. **Frontend**: HTML/CSS/JavaScript UI in `templates/index.html`
2. **Backend**: Flask server in `main.py` that communicates with the LilLisa Server
3. **Data Flow**: Browser → Flask App → LilLisa Server → Flask App → Browser

## API Endpoints

The application exposes the following API endpoints:

- **GET `/`**: Serves the main web interface
- **POST `/api/stream_with_nodes`**: Streams responses from the LilLisa Server to the browser
- **POST `/api/thumbsup`**: Records positive feedback for a response
- **POST `/api/thumbsdown`**: Records negative feedback for a response
- **POST `/api/thumbsfeedback`**: Records feedback for individual source chunks

## Setup and Configuration

### Prerequisites

- Python 3.8+
- Access to a running LilLisa Server instance

### Environment Variables

Create a `lil-lisa-web.env` file with the following variables:

```
LIL_LISA_SERVER_URL=http://your-lillisa-server-url
```

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   conda env create -f environment.yml
   ```
   or
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up the environment variables as described above

### Running the Application

```bash
python main.py
```

The application will be available at `http://localhost:8080` by default.

## Key Components

### main.py

The main Flask application file that:
- Sets up the web server
- Handles API requests
- Proxies requests to the LilLisa Server
- Manages streaming responses
- Processes feedback

### templates/index.html

The frontend UI that:
- Provides a chat interface
- Handles session management
- Displays responses and source information
- Manages feedback submission
- Offers product switching

## Integration with LilLisa Server

The web interface interacts with two main endpoints on the LilLisa Server:

1. **`/invoke_stream_with_nodes/`**: Used to get streaming responses with source information
2. **`/record_endorsement/`**: Used to record user feedback on responses and source chunks

## Support

Reach out to us if you have questions:
- Dhar Rawal (Slack: @Dhar Rawal, Email: drawal@radiantlogic.com)

## Authors and acknowledgment

- Carlos Escobar
- Dhar Rawal
- Unsh Rawal
- Nico Guyot
- Priyanshu Jani

## License

This project is currently closed source

## Project status

Under active development
