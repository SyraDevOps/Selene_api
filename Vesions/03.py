import csv
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader
import sqlite3
import logging
import shutil

# --- CONFIGURAÇÕES GLOBAIS ---
MODELS_DIR = "models"
LOG_FILE = "luna_bot.log"

# --- 1. CONFIGURAÇÃO DE LOGS ---
def setup_logging():
    """Configura o logging para registrar eventos em um arquivo e no console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler() # Exibe logs no terminal também
        ]
    )

# --- CLASSES E FUNÇÕES AUXILIARES ---
class TextDataset(Dataset):
    """Dataset customizado para PyTorch compatível com matrizes esparsas TF-IDF."""
    def __init__(self, X, y):
        self.X = X.tocsr() # Converte para CSR para fatiamento eficiente
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Converte a linha esparsa para um tensor denso para o modelo
        x_sample = torch.tensor(self.X[idx].toarray().squeeze(), dtype=torch.float32)
        y_sample = self.y[idx]
        return x_sample, y_sample

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def criar_diretorio_modelos():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        logging.info(f"Diretório de modelos '{MODELS_DIR}' criado.")

def listar_modelos():
    criar_diretorio_modelos()
    return [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]

def db_connect(model_name):
    db_path = os.path.join(MODELS_DIR, model_name, "data.db")
    return sqlite3.connect(db_path)

# --- 2. GERENCIAMENTO DE DADOS E MODELOS ---
def criar_novo_modelo():
    model_name = input("Digite o nome para o novo modelo: ").strip()
    if not model_name:
        print("\n[ERRO] O nome do modelo não pode ser vazio.")
        logging.warning("Tentativa de criar modelo com nome vazio.")
        return

    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(model_path):
        print(f"\n[ERRO] O modelo '{model_name}' já existe.")
        logging.warning(f"Tentativa de criar um modelo já existente: {model_name}")
        return

    os.makedirs(model_path)
    conn = db_connect(model_name)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dataset (
        pergunta TEXT, resposta TEXT, contexto TEXT, UNIQUE(pergunta, contexto)
    )""")
    conn.commit()
    conn.close()
    print(f"\n[SUCESSO] Modelo '{model_name}' criado.")
    logging.info(f"Modelo '{model_name}' criado com sucesso em '{model_path}'.")

def carregar_dados_csv_para_db(model_name, csv_files):
    conn = db_connect(model_name)
    cursor = conn.cursor()
    count = 0
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            logging.warning(f"Arquivo CSV '{csv_file}' não encontrado. Pulando.")
            continue
        with open(csv_file, newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                # Inserir ignorando duplicatas (baseado na constraint UNIQUE)
                cursor.execute("INSERT OR IGNORE INTO dataset (pergunta, resposta, contexto) VALUES (?, ?, ?)", (row[0], row[1], row[2]))
                count += cursor.rowcount
    conn.commit()
    conn.close()
    logging.info(f"{count} novas linhas inseridas no banco de dados do modelo '{model_name}'.")

def carregar_dados_do_db(model_name):
    perguntas, respostas = [], []
    conn = db_connect(model_name)
    cursor = conn.cursor()
    cursor.execute("SELECT pergunta, resposta, contexto FROM dataset")
    for row in cursor.fetchall():
        perguntas.append(row[0] + " " + row[2])
        respostas.append(row[1])
    conn.close()
    return perguntas, respostas

# --- 3. TREINAMENTO (COM MELHORIAS) ---
def treinar_modelo():
    modelos = listar_modelos()
    if not modelos:
        print("\n[AVISO] Nenhum modelo encontrado. Crie um primeiro.")
        return

    print("\nModelos disponíveis:")
    for i, name in enumerate(modelos): print(f"{i + 1}. {name}")
    try:
        choice = int(input("Escolha o modelo para treinar: ")) - 1
        model_name = modelos[choice]
    except (ValueError, IndexError):
        print("\n[ERRO] Escolha inválida.")
        return

    csv_paths_input = input("Digite os caminhos para os arquivos CSV (separados por vírgula): ")
    csv_files = [path.strip() for path in csv_paths_input.split(',')]

    model_path = os.path.join(MODELS_DIR, model_name)
    model_file = os.path.join(model_path, "torch_model.pt")
    
    train_mode = 'scratch'
    if os.path.exists(model_file):
        mode_choice = input("Um modelo treinado já existe. Deseja [C]ontinuar o treinamento ou [R]e-treinar do zero? (C/R): ").lower()
        if mode_choice == 'r':
            train_mode = 'scratch'
        elif mode_choice == 'c':
            train_mode = 'continue'
        else:
            print("\n[ERRO] Opção inválida.")
            return

    try:
        epochs = int(input("Digite o número de épocas: "))
        batch_size = int(input("Digite o tamanho do batch (ex: 16, 32): "))
    except ValueError:
        print("\n[ERRO] Número de épocas ou batch size inválido.")
        return

    logging.info(f"Iniciando treinamento para o modelo '{model_name}'. Modo: {train_mode}.")
    
    # Limpar DB se for re-treino
    if train_mode == 'scratch':
        conn = db_connect(model_name)
        conn.cursor().execute("DELETE FROM dataset")
        conn.commit()
        conn.close()
        logging.info("Banco de dados limpo para re-treino do zero.")

    carregar_dados_csv_para_db(model_name, csv_files)
    perguntas, respostas = carregar_dados_do_db(model_name)
    
    if not perguntas:
        print("\n[ERRO] Nenhum dado encontrado para treinamento.")
        logging.error(f"Treinamento cancelado para '{model_name}': nenhum dado disponível.")
        return

    # Vetorização
    vectorizer_file = os.path.join(model_path, "vectorizer.pkl")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(perguntas)
    
    # Preparação dos dados Torch
    respostas_set = sorted(list(set(respostas)))
    resposta2idx = {resp: idx for idx, resp in enumerate(respostas_set)}
    idx2resposta = {idx: resp for resp, idx in resposta2idx.items()}
    y = torch.tensor([resposta2idx[r] for r in respostas], dtype=torch.long)
    
    # DataLoader
    dataset = TextDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Modelo
    model = SimpleModel(X.shape[1], len(respostas_set))
    if train_mode == 'continue':
        try:
            model.load_state_dict(torch.load(model_file))
            logging.info(f"Estado do modelo anterior carregado para '{model_name}'.")
        except Exception as e:
            logging.error(f"Falha ao carregar modelo existente. Treinando do zero. Erro: {e}")
            print("\n[AVISO] Não foi possível carregar o modelo anterior. Treinando do zero.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("\n--- INICIANDO TREINAMENTO ---")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in data_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f"Época {epoch+1}/{epochs} | Perda Média: {avg_loss:.4f}")
        logging.info(f"Modelo '{model_name}' - Época {epoch+1}/{epochs} - Perda Média: {avg_loss:.4f}")

    # Salvar tudo
    torch.save(model.state_dict(), model_file)
    with open(vectorizer_file, "wb") as f:
        pickle.dump((vectorizer, resposta2idx, idx2resposta), f)
        
    print(f"\n[SUCESSO] Treinamento do modelo '{model_name}' concluído!")
    logging.info(f"Modelo '{model_name}' e vetorizador salvos com sucesso.")


# --- 4. CHAT E OUTRAS UTILIDADES ---
def iniciar_chat():
    modelos = listar_modelos()
    if not modelos:
        print("\n[AVISO] Nenhum modelo treinado encontrado.")
        return

    print("\nModelos disponíveis para carregar:")
    for i, name in enumerate(modelos): print(f"{i + 1}. {name}")
    try:
        choice = int(input("Escolha o modelo para carregar: ")) - 1
        model_name = modelos[choice]
    except (ValueError, IndexError):
        print("\n[ERRO] Escolha inválida.")
        return

    model_path = os.path.join(MODELS_DIR, model_name)
    model_file = os.path.join(model_path, "torch_model.pt")
    vectorizer_file = os.path.join(model_path, "vectorizer.pkl")
    if not all(os.path.exists(f) for f in [model_file, vectorizer_file]):
        print(f"\n[ERRO] O modelo '{model_name}' não foi treinado ainda.")
        logging.warning(f"Tentativa de carregar o modelo não treinado '{model_name}'.")
        return

    with open(vectorizer_file, "rb") as f:
        vectorizer, resposta2idx, idx2resposta = pickle.load(f)

    model = SimpleModel(len(vectorizer.get_feature_names_out()), len(resposta2idx))
    model.load_state_dict(torch.load(model_file))
    model.eval()
    
    logging.info(f"Iniciando sessão de chat com o modelo '{model_name}'.")
    print(f"\n=== Chat com '{model_name.upper()}' ===")
    print("(digite 'sair' para encerrar)")
    contexto = ""
    while True:
        pergunta = input("Você: ")
        if pergunta.lower() == "sair": break
        query = pergunta + " " + contexto
        X_query = vectorizer.transform([query])
        X_tensor = torch.tensor(X_query.toarray(), dtype=torch.float32)
        with torch.no_grad():
            output = model(X_tensor)
        idx = torch.argmax(output, dim=1).item()
        resposta = idx2resposta[idx]
        print(f"{model_name.capitalize()}: {resposta}")
        contexto = pergunta[-30:]

def exportar_modelo():
    modelos = listar_modelos()
    if not modelos:
        print("\n[AVISO] Nenhum modelo para exportar.")
        return

    print("\nModelos disponíveis para exportar:")
    for i, name in enumerate(modelos): print(f"{i + 1}. {name}")
    try:
        choice = int(input("Escolha o modelo para exportar: ")) - 1
        model_name = modelos[choice]
    except (ValueError, IndexError):
        print("\n[ERRO] Escolha inválida.")
        return

    model_path = os.path.join(MODELS_DIR, model_name)
    output_filename = f"{model_name}_export"
    
    try:
        shutil.make_archive(output_filename, 'zip', model_path)
        print(f"\n[SUCESSO] Modelo '{model_name}' exportado para '{output_filename}.zip'.")
        logging.info(f"Modelo '{model_name}' exportado para '{output_filename}.zip'.")
    except Exception as e:
        print(f"\n[ERRO] Falha ao exportar o modelo. Detalhes: {e}")
        logging.error(f"Erro ao exportar o modelo '{model_name}': {e}")

def importar_modelo():
    zip_path = input("Digite o caminho para o arquivo .zip do modelo: ")
    if not os.path.exists(zip_path) or not zip_path.endswith('.zip'):
        print("\n[ERRO] Arquivo não encontrado ou não é um .zip válido.")
        return
        
    model_name = os.path.basename(zip_path).replace('_export.zip', '')
    extract_path = os.path.join(MODELS_DIR, model_name)

    if os.path.exists(extract_path):
        overwrite = input(f"O modelo '{model_name}' já existe. Deseja sobrescrevê-lo? (s/n): ").lower()
        if overwrite != 's':
            print("Importação cancelada.")
            return

    try:
        shutil.unpack_archive(zip_path, extract_path)
        print(f"\n[SUCESSO] Modelo '{model_name}' importado com sucesso.")
        logging.info(f"Modelo importado de '{zip_path}' para '{extract_path}'.")
    except Exception as e:
        print(f"\n[ERRO] Falha ao importar o modelo. Detalhes: {e}")
        logging.error(f"Erro ao importar de '{zip_path}': {e}")


# --- 5. MENU PRINCIPAL ---
def menu_principal():
    setup_logging()
    criar_diretorio_modelos()
    logging.info("Aplicação iniciada.")
    while True:
        print("\n" + "="*35)
        print("      MENU PRINCIPAL - LUNA BOT")
        print("="*35)
        print("1. Criar um novo modelo")
        print("2. Treinar/Continuar treinamento")
        print("3. Carregar modelo e iniciar chat")
        print("4. Exportar um modelo (.zip)")
        print("5. Importar um modelo (.zip)")
        print("6. Sair")
        
        escolha = input("\nEscolha uma opção: ")

        if escolha == "1": criar_novo_modelo()
        elif escolha == "2": treinar_modelo()
        elif escolha == "3": iniciar_chat()
        elif escolha == "4": exportar_modelo()
        elif escolha == "5": importar_modelo()
        elif escolha == "6":
            logging.info("Aplicação encerrada.")
            print("\nAté logo!")
            break
        else:
            print("\n[ERRO] Opção inválida. Tente novamente.")

if __name__ == "__main__":
    menu_principal()