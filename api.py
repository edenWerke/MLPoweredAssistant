from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
import hashlib
import json
from main import ChatbotAssistant, get_stocks, file_hash
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

assistant = ChatbotAssistant('intents.json', function_mappings={'stocks': get_stocks})
assistant.parse_intents()

model_exists = os.path.exists('chatbot_model.pkl')
intents_hash_path = 'intents.hash'
current_hash = file_hash('intents.json')
previous_hash = None
if os.path.exists(intents_hash_path):
    with open(intents_hash_path, 'r') as f:
        previous_hash = f.read().strip()

needs_retrain = (not model_exists) or (previous_hash != current_hash)
if needs_retrain:
    assistant.prepare_data()
    assistant.train_model(batch_size=8, lr=0.001, epochs=100)
    assistant.save_pickle('chatbot_model.pkl')
    with open(intents_hash_path, 'w') as f:
        f.write(current_hash)
    assistant.load_pickle('chatbot_model.pkl')
else:
    assistant.load_pickle('chatbot_model.pkl')

@app.post('/chat')
async def chat_endpoint(request: ChatRequest):
    response = assistant.process_message(request.message)
    return {"response": response} 