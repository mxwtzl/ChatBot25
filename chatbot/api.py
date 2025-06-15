from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from Onkel_Host2 import christmasAgent, LogWriter
import re

app = FastAPI()

# CORS für lokale Entwicklung
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, christmasAgent] = {}

def validate_userid(userid: str) -> bool:
    pattern = r"^[a-zA-Z0-9_]{3,20}$"
    return bool(re.match(pattern, userid))

def get_agent(userid: str = "anonymous"):
    if not validate_userid(userid):
        raise HTTPException(status_code=400, detail="Invalid userid: must be 3-20 alphanumeric characters or underscores")
    if userid not in sessions:
        sessions[userid] = christmasAgent()
    return sessions[userid]

agent = christmasAgent()

logger = LogWriter()

class SetUserIdRequest(BaseModel):
    userid: str

class SetLanguageRequest(BaseModel):
    userid: str = "anonymous"
    language: str

class ChatMessage(BaseModel):
    message: str
    chat_history: List[str] = []
    userid: str = "anonymous"
    language: str = "de"

class ChatResponse(BaseModel):
    response: str
    state: str
    log_message: Dict[str, Any]
    round_count: int

@app.post("/set-userid")
async def set_userid(request: SetUserIdRequest):
    try:
        if not validate_userid(request.userid):
            raise HTTPException(status_code=400, detail="Invalid userid: must be 3-20 alphanumeric characters or underscores")
        sessions[request.userid] = christmasAgent()
        return {"message": f"User ID set to {request.userid}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/set-language")
async def set_language(request: SetLanguageRequest):
    try:
        if request.language not in ["de", "en"]:
            raise HTTPException(status_code=400, detail="Language must be 'de' or 'en'")
        agent = get_agent(request.userid)
        agent.language = request.language
        return {"message": f"Language set to {request.language} for user {request.userid}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage, agent: christmasAgent = Depends()):
    try:
        # Auswahl des ChatAgent über UserID
        agent = get_agent(chat_message.userid)
        agent.language = chat_message.language if chat_message.language in ["de", "en"] else agent.language
        response, log_message = agent.get_response(
            chat_message.message,
            chat_message.chat_history,
            chat_message.userid,
        )
        logger.write(log_message)
        return ChatResponse(
            response=response,
            state=agent.state,
            log_message=log_message,
            round_count=agent.round_count,
            language=agent.language,
            userid=chat_message.userid,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 