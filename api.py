from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import sys
import pickle
import torch
import torch.nn as nn

# Garantir que o diretório atual está no path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Diretório onde os modelos treinados do Main.py são salvos
MODELS_DIR = os.path.join(current_dir, "models")

app = FastAPI(
    title="Luna Chat API",
    description="API para interagir com o modelo de IA Luna (sem contexto)",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SimpleModel replicado (igual ao usado no Main.py)
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Helper: listar modelos (pastas em models/)
def listar_modelos():
    if not os.path.exists(MODELS_DIR):
        return []
    return [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]

# Wrapper para carregar vectorizer + modelo torch
class LunaWrapper:
    def __init__(self, model_name: str):
        model_path = os.path.join(MODELS_DIR, model_name)
        vectorizer_file = os.path.join(model_path, "vectorizer.pkl")
        model_file = os.path.join(model_path, "torch_model.pt")

        if not os.path.exists(vectorizer_file) or not os.path.exists(model_file):
            raise FileNotFoundError("Modelo ou vetorizador não encontrados para: " + model_name)

        with open(vectorizer_file, "rb") as f:
            vectorizer, resposta2idx, idx2resposta = pickle.load(f)

        input_dim = len(vectorizer.get_feature_names_out())
        output_dim = len(resposta2idx)

        model = SimpleModel(input_dim, output_dim)
        model.load_state_dict(torch.load(model_file, map_location="cpu"))
        model.eval()

        self.vectorizer = vectorizer
        self.idx2resposta = idx2resposta
        self.model = model

    def predict(self, message: str) -> str:
        X = self.vectorizer.transform([message])
        X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
        with torch.no_grad():
            out = self.model(X_tensor)
        idx = torch.argmax(out, dim=1).item()
        return self.idx2resposta.get(idx, "")

# Request/response models (sem contexto)
class ChatRequest(BaseModel):
    message: str
    model_name: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    model: str

# Store active model instances
active_models = {}

# Dependency: obter/instanciar modelo (global)
async def get_model(model_name: Optional[str] = None) -> LunaWrapper:
    # Forçar uso do modelo chamado "luna"
    model_name = "Luna"
    user_id = "global"

    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Create folder: {model_path}")

    if user_id not in active_models or active_models[user_id]["name"] != model_name:
        try:
            wrapper = LunaWrapper(model_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model '{model_name}': {e}")
        active_models[user_id] = {"name": model_name, "model": wrapper}

    return active_models[user_id]["model"]

# Routes
@app.get("/api/models", response_model=List[str])
async def get_models():
    """Retorna apenas 'luna' se existir"""
    if os.path.exists(os.path.join(MODELS_DIR, "luna")):
        return ["luna"]
    return []

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to Luna and get a response (no context)"""
    model = await get_model(request.model_name)
    resp = model.predict(request.message)
    return ChatResponse(response=resp, model=active_models["global"]["name"])

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Luna API (stateless) is running"}