import csv
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# CONFIGURAÇÃO
CSV_FILE = "Datasets.csv"
MODEL_FILE = "torch_model.pt"
VECTORIZER_FILE = "vectorizer.pkl"

# -------------------------------
# 1 — Ler dataset CSV
def load_dataset():
    perguntas = []
    respostas = []
    contexto = []
    with open(CSV_FILE, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # pular header
        for row in reader:
            perguntas.append(row[0] + " " + row[2])  # juntar contexto
            respostas.append(row[1])
    return perguntas, respostas

# 2 — Criar embeddings simples com TF-IDF
def train_vectorizer(perguntas):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(perguntas)
    return vectorizer, X

# 3 — Criar modelo Torch simples (Logistic Regression)
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# 4 — Treinar modelo
def train_model(X, respostas):
    respostas_set = list(set(respostas))
    resposta2idx = {resp: idx for idx, resp in enumerate(respostas_set)}
    idx2resposta = {idx: resp for resp, idx in resposta2idx.items()}
    y = torch.tensor([resposta2idx[r] for r in respostas])

    model = SimpleModel(X.shape[1], len(respostas_set))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)

    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch} Loss: {loss.item():.4f}")

    return model, resposta2idx, idx2resposta

# 5 — Salvar modelo
def save_model(model, vectorizer, resposta2idx, idx2resposta):
    torch.save(model.state_dict(), MODEL_FILE)
    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump((vectorizer, resposta2idx, idx2resposta), f)

# 6 — Carregar modelo
def load_model(vectorizer):
    perguntas, respostas = load_dataset()
    X = vectorizer.transform(perguntas)
    respostas_set = list(set(respostas))
    model = SimpleModel(X.shape[1], len(respostas_set))
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()
    with open(VECTORIZER_FILE, "rb") as f:
        _, resposta2idx, idx2resposta = pickle.load(f)
    return model, idx2resposta

# 7 — Fazer pergunta
def ask_question(model, vectorizer, idx2resposta, pergunta, contexto=""):
    query = pergunta + " " + contexto
    X_query = vectorizer.transform([query])
    X_tensor = torch.tensor(X_query.toarray(), dtype=torch.float32)
    with torch.no_grad():
        output = model(X_tensor)
    idx = torch.argmax(output, dim=1).item()
    return idx2resposta[idx]

# -------------------------------
if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        perguntas, respostas = load_dataset()
        vectorizer, X = train_vectorizer(perguntas)
        model, resposta2idx, idx2resposta = train_model(X, respostas)
        save_model(model, vectorizer, resposta2idx, idx2resposta)
    else:
        with open(VECTORIZER_FILE, "rb") as f:
            vectorizer, resposta2idx, idx2resposta = pickle.load(f)
        model = SimpleModel(len(vectorizer.get_feature_names_out()), len(resposta2idx))
        model.load_state_dict(torch.load(MODEL_FILE))
        model.eval()

    print("=== Chat Simples com Luna === (digite 'sair' para encerrar)")
    contexto = ""
    while True:
        pergunta = input("Você: ")
        if pergunta.lower() == "sair":
            break
        resposta = ask_question(model, vectorizer, idx2resposta, pergunta, contexto)
        print(f"Luna: {resposta}")
        contexto = pergunta[-20:]  # influência leve do contexto
