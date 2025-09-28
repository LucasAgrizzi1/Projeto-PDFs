import os
import logging
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings

# --- Configuração Inicial ---
# 1. Configurar a chave da API da OpenAI
# Recomenda-se definir a chave como uma variável de ambiente (OPENAI_API_KEY)
# Exemplo: os.environ["OPENAI_API_KEY"] = "SUA_CHAVE_OPENAI_AQUI"

# 2. Configurar o nível de logging
logging.basicConfig(level=logging.INFO)

# 3. Definir o diretório onde estão seus arquivos PDF
DATA_DIR = "data"

# 4. Configurações do Modelo (LLM) e Embeddings
# Usando gpt-3.5-turbo para chat e o modelo de embedding padrão da OpenAI
Settings.llm = OpenAI(model="gpt-3.5-turbo")
# O modelo de embedding (para a busca vetorial) será o padrão (text-embedding-ada-002)

def inicializar_sistema_rag():
    """
    Carrega os documentos, cria o índice vetorial e o motor de consulta.
    """
    print("--- 1. Carregando Documentos PDF ---")
    try:
        # Carrega todos os arquivos do diretório 'data'
        documentos = SimpleDirectoryReader(DATA_DIR).load_data()
        print(f"✅ Documentos carregados: {len(documentos)} arquivos encontrados em '{DATA_DIR}'.")
    except Exception as e:
        print(f"❌ Erro ao carregar documentos. Verifique se a pasta '{DATA_DIR}' existe e contém PDFs.")
        print(f"Detalhes: {e}")
        return None

    print("\n--- 2. Criando Índice Vetorial (Busca) ---")
    # Cria o índice vetorial a partir dos documentos.
    # Esta etapa processa o texto e cria embeddings (vetores) que permitem
    # a busca semântica (por significado).
    index = VectorStoreIndex.from_documents(documentos)
    print("✅ Índice Vetorial criado com sucesso.")

    # Cria o motor de consulta que orquestra a Recuperação e a Geração (RAG)
    query_engine = index.as_query_engine(
        similarity_top_k=3, # Recuperar os 3 trechos de texto mais relevantes
        # Adicionar instruções para o modelo base
        system_prompt=(
            "Você é um assistente de pesquisa para um estudante de TCC. "
            "Sua tarefa é responder a perguntas estritamente com base nas informações fornecidas "
            "nos documentos recuperados. Seja conciso e cite as fontes sempre que possível."
        )
    )
    
    return query_engine

def iniciar_chat(query_engine):
    """
    Inicia o loop de chat interativo.
    """
    print("\n" + "="*50)
    print("        SISTEMA DE CHAT RAG PARA TCC INICIADO")
    print("="*50)
    print("Faça perguntas sobre o conteúdo dos seus PDFs. Digite 'sair' para encerrar.")
    
    while True:
        pergunta = input("\nVocê (TCC): ")
        if pergunta.lower() == 'sair':
            print("\nObrigado por usar o assistente de TCC. Até a próxima!")
            break
        
        if not pergunta.strip():
            continue
            
        print("AI (Assistente): Pensando...")

        # O motor de consulta executa o RAG
        # 1. Recuperação: Busca os trechos mais relevantes do índice.
        # 2. Geração: Envia os trechos e a pergunta para o LLM.
        try:
            resposta = query_engine.query(pergunta)
            
            # Exibe a resposta e as fontes usadas para a fundamentação
            print("\n--- Resposta Fundamentada ---")
            print(resposta.response)
            
            print("\n--- Fontes Utilizadas ---")
            # Lista os metadados dos trechos recuperados (geralmente nome do arquivo)
            for node in resposta.source_nodes:
                # O 'file_name' é um metadado padrão do SimpleDirectoryReader
                file_name = node.metadata.get('file_name', 'Fonte Desconhecida')
                print(f"-> Arquivo: {file_name}")
                # Opcional: Mostrar o trecho (chunk) recuperado
                # print(f"   Trecho: {node.text[:150]}...") 
                
        except Exception as e:
            print(f"Ocorreu um erro durante a consulta: {e}")
            
# --- Execução Principal ---
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("ATENÇÃO: A variável de ambiente OPENAI_API_KEY não está definida.")
        print("Por favor, defina a chave da API para que o modelo de IA possa ser usado.")
    else:
        # Cria a pasta 'data' se não existir
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            print(f"Pasta '{DATA_DIR}' criada. Coloque seus PDFs nela e execute novamente.")
        else:
            engine = inicializar_sistema_rag()
            if engine:
                iniciar_chat(engine)
