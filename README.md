# Projeto-PDFs

MODO DE USO:

instale a biblioteca necessária:

Bash:
pip install llama-index pypdf openai.

Crie uma pasta chamada data e coloque seus arquivos PDF de TCC dentro dela. 

Adicione Seus PDFs: Coloque todos os seus artigos e documentos de TCC dentro da pasta data.

Defina a Chave: Certifique-se de que sua chave da API da OpenAI esteja configurada como uma variável de ambiente (OPENAI_API_KEY).

Execute o Script:

Bash
python tcc_rag_chat.py

Interaja: O sistema irá carregar, indexar seus PDFs e iniciar o chat, permitindo que você faça perguntas complexas sobre o conteúdo dos seus documentos.
