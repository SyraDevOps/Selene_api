import csv
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sqlite3

# --- CONFIGURAÇÕES GLOBAIS ---
MODELS_DIR = "models"

# --- FUNÇÕES AUXILIARES ---
def criar_diretorio_modelos():
    """Cria o diretório 'models' se ele não existir."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

def listar_modelos():
    """Lista todos os modelos disponíveis no diretório 'models'."""
    criar_diretorio_modelos()
    return [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]

def db_connect(model_name):
    """Conecta ao banco de dados SQLite de um modelo específico."""
    db_path = os.path.join(MODELS_DIR, model_name, "data.db")
    return sqlite3.connect(db_path)

# --- 1. GERENCIAMENTO DE MODELOS ---
def criar_novo_modelo():
    """Cria a estrutura de pastas e o banco de dados para um novo modelo."""
    model_name = input("Digite o nome para o novo modelo: ")
    model_path = os.path.join(MODELS_DIR, model_name)

    if os.path.exists(model_path):
        print(f"\n[ERRO] O modelo '{model_name}' já existe.")
        return

    os.makedirs(model_path)
    
    # Cria o banco de dados
    conn = db_connect(model_name)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dataset (
        pergunta TEXT,
        resposta TEXT,
        contexto TEXT
    )
    """)
    conn.commit()
    conn.close()
    
    print(f"\n[SUCESSO] Modelo '{model_name}' criado em '{model_path}'.")

# --- 2. TREINAMENTO ---
def carregar_dados_csv_para_db(model_name, csv_file):
    """Carrega dados de um arquivo CSV para o banco de dados do modelo."""
    conn = db_connect(model_name)
    cursor = conn.cursor()
    
    # Limpa dados antigos para novo treinamento
    cursor.execute("DELETE FROM dataset")

    with open(csv_file, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Pular cabeçalho
        for row in reader:
            cursor.execute("INSERT INTO dataset (pergunta, resposta, contexto) VALUES (?, ?, ?)", (row[0], row[1], row[2]))
    
    conn.commit()
    conn.close()
    print(f"\n[INFO] Dados do arquivo '{csv_file}' carregados para o modelo '{model_name}'.")

def carregar_dados_do_db(model_name):
    """Carrega perguntas e respostas do banco de dados do modelo."""
    perguntas = []
    respostas = []
    conn = db_connect(model_name)
    cursor = conn.cursor()
    cursor.execute("SELECT pergunta, resposta, contexto FROM dataset")
    
    for row in cursor.fetchall():
        # Concatena pergunta e contexto para o treinamento
        perguntas.append(row[0] + " " + row[2])
        respostas.append(row[1])
        
    conn.close()
    return perguntas, respostas

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def treinar_modelo():
    """Função principal para treinar um modelo existente."""
    modelos = listar_modelos()
    if not modelos:
        print("\n[AVISO] Nenhum modelo encontrado. Crie um modelo primeiro.")
        return

    print("\nModelos disponíveis:")
    for i, name in enumerate(modelos):
        print(f"{i + 1}. {name}")
    
    try:
        choice = int(input("Escolha o número do modelo para treinar: ")) - 1
        model_name = modelos[choice]
    except (ValueError, IndexError):
        print("\n[ERRO] Escolha inválida.")
        return
        
    csv_file = input("Digite o caminho para o arquivo CSV de treinamento: ")
    if not os.path.exists(csv_file):
        print(f"\n[ERRO] Arquivo '{csv_file}' não encontrado.")
        return

    try:
        epochs = int(input("Digite o número de épocas para o treinamento (ex: 200): "))
    except ValueError:
        print("\n[ERRO] Número de épocas inválido.")
        return

    # Processo de treinamento
    print("\n--- INICIANDO TREINAMENTO ---")
    carregar_dados_csv_para_db(model_name, csv_file)
    perguntas, respostas = carregar_dados_do_db(model_name)

    # 1. Vetorização com TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(perguntas)
    
    # 2. Preparação dos dados para o PyTorch
    respostas_set = sorted(list(set(respostas)))
    resposta2idx = {resp: idx for idx, resp in enumerate(respostas_set)}
    idx2resposta = {idx: resp for resp, idx in resposta2idx.items()}
    y = torch.tensor([resposta2idx[r] for r in respostas], dtype=torch.long)
    
    # 3. Definição do modelo, critério e otimizador
    model = SimpleModel(X.shape[1], len(respostas_set))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)

    # 4. Loop de treinamento
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Época {epoch+1}/{epochs} | Perda (Loss): {loss.item():.4f}")

    # 5. Salvar o modelo e o vetorizador
    model_path = os.path.join(MODELS_DIR, model_name)
    torch.save(model.state_dict(), os.path.join(model_path, "torch_model.pt"))
    with open(os.path.join(model_path, "vectorizer.pkl"), "wb") as f:
        pickle.dump((vectorizer, resposta2idx, idx2resposta), f)
        
    print(f"\n[SUCESSO] Treinamento do modelo '{model_name}' concluído!")

# --- 3. CHAT ---
def iniciar_chat():
    """Carrega um modelo treinado e inicia uma sessão de chat."""
    modelos = listar_modelos()
    if not modelos:
        print("\n[AVISO] Nenhum modelo treinado encontrado.")
        return

    print("\nModelos disponíveis para carregar:")
    for i, name in enumerate(modelos):
        print(f"{i + 1}. {name}")

    try:
        choice = int(input("Escolha o número do modelo para carregar: ")) - 1
        model_name = modelos[choice]
    except (ValueError, IndexError):
        print("\n[ERRO] Escolha inválida.")
        return
        
    model_path = os.path.join(MODELS_DIR, model_name)
    model_file = os.path.join(model_path, "torch_model.pt")
    vectorizer_file = os.path.join(model_path, "vectorizer.pkl")

    if not os.path.exists(model_file) or not os.path.exists(vectorizer_file):
        print(f"\n[ERRO] O modelo '{model_name}' não foi treinado ainda.")
        return

    # Carregar modelo e vetorizador
    with open(vectorizer_file, "rb") as f:
        vectorizer, resposta2idx, idx2resposta = pickle.load(f)

    input_dim = len(vectorizer.get_feature_names_out())
    output_dim = len(resposta2idx)
    
    model = SimpleModel(input_dim, output_dim)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    print(f"\n=== Chat com '{model_name.upper()}' ===")
    print("(digite 'sair' para encerrar)")
    contexto = ""
    while True:
        pergunta = input("Você: ")
        if pergunta.lower() == "sair":
            break

        # Fazer a predição
        query = pergunta + " " + contexto
        X_query = vectorizer.transform([query])
        X_tensor = torch.tensor(X_query.toarray(), dtype=torch.float32)
        with torch.no_grad():
            output = model(X_tensor)
        
        idx = torch.argmax(output, dim=1).item()
        resposta = idx2resposta[idx]
        
        print(f"{model_name.capitalize()}: {resposta}")
        contexto = pergunta[-30:] # Atualiza o contexto com o final da última pergunta

# --- MENU PRINCIPAL ---
def menu_principal():
    """Exibe o menu principal e gerencia a navegação do usuário."""
    criar_diretorio_modelos()
    while True:
        print("\n" + "="*30)
        print("    MENU PRINCIPAL - LUNA BOT")
        print("="*30)
        print("1. Criar um novo modelo")
        print("2. Treinar um modelo existente")
        print("3. Carregar modelo e iniciar chat")
        print("4. Sair")
        
        escolha = input("\nEscolha uma opção: ")

        if escolha == "1":
            criar_novo_modelo()
        elif escolha == "2":
            treinar_modelo()
        elif escolha == "3":
            iniciar_chat()
        elif escolha == "4":
            print("\nAté logo!")
            break
        else:
            print("\n[ERRO] Opção inválida. Tente novamente.")

if __name__ == "__main__":
    menu_principal()