import uuid
from datetime import datetime
from typing import Dict, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our custom modules
from app.config import settings
from app.ai_brain import get_ai_response
from app.lead_capture import send_lead_email

# --- FastAPI Models & Setup ---
app = FastAPI(title=settings.APP_NAME, version=settings.VERSION)

# CORS middleware for allowing requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (use a database like Redis for production)
sessions: Dict[str, dict] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    confidence: float
    requires_action: Optional[str] = None

def get_session(session_id: str) -> dict:
    """Get or create a new user session."""
    if session_id not in sessions:
        sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "message_history": [],
            "awaiting_action": None,
            "user_data": {}
        }
    return sessions[session_id]

def detect_lead_intent(message: str) -> bool:
    """Detect if the user's message matches any lead capture phrases."""
    return any(phrase in message.lower() for phrase in settings.LEAD_CAPTURE_PHRASES)

# --- Main Chatbot Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request_data: ChatRequest):
    """
    Main endpoint to handle all chat messages.
    It orchestrates the conversation flow, lead capture, and RAG responses.
    """
    session_id = request_data.session_id or str(uuid.uuid4())
    session = get_session(session_id)
    
    # Store the user's message in the session history
    session["message_history"].append({
        "role": "user",
        "content": request_data.message,
        "timestamp": datetime.now().isoformat()
    })
    
    response_text = ""
    confidence = 0.7  # Default confidence
    requires_action = None

    # --- MAIN LOGIC FLOW ---
    # 1. Check if the bot is in a lead capture state
    if session.get("awaiting_action") == "get_name":
        session["user_data"]["name"] = request_data.message
        session["awaiting_action"] = "get_email"
        response_text = f"Thanks {request_data.message}! What's your email address?"
        requires_action = "get_email"
        confidence = 0.9
    elif session.get("awaiting_action") == "get_email":
        if "@" in request_data.message and "." in request_data.message:
            session["user_data"]["email"] = request_data.message
            session["awaiting_action"] = None
            response_text = f"Perfect! We'll contact you at {request_data.message} within 24 hours! 🎉"
            send_lead_email(session["user_data"], session["message_history"]) # Call the email function
            confidence = 1.0
        else:
            response_text = "That doesn't look like a valid email. Can you please try again?"
            session["awaiting_action"] = "get_email"
            confidence = 0.5
            requires_action = "get_email"
    # 2. If not in a state, check for new lead intent
    elif detect_lead_intent(request_data.message):
        session["awaiting_action"] = "get_name"
        response_text = "Awesome! Let's get you started. What's your name?"
        confidence = 0.9
        requires_action = "get_name"
    # 3. If no lead intent, use the RAG system
    else:
        ai_response = get_ai_response(request_data.message)
        response_text = ai_response["response"]
        confidence = ai_response["confidence"]
    
    # Store AI response in history
    session["message_history"].append({
        "role": "assistant",
        "content": response_text,
        "timestamp": datetime.now().isoformat()
    })
    
    return ChatResponse(
        response=response_text,
        session_id=session_id,
        confidence=confidence,
        requires_action=requires_action
    )

# --- Other Endpoints (for monitoring/debugging) ---
@app.get("/health")
async def health_check():
    """Health endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "sessions_active": len(sessions)
    }

@app.get("/session/{session_id}")
async def get_session_data(session_id: str):
    """Get session data (for debugging)."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]

# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
