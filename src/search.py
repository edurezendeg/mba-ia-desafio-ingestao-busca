"""Busca semantica no banco vetorial + geracao da resposta pela LLM.

Concentra a configuracao compartilhada (embeddings, LLM e vector store) para
que `ingest.py` e `chat.py` reutilizem a mesma definicao de conexao/colecao.

`search_prompt()` monta e retorna um chain (LCEL) que, ao ser invocado com a
pergunta do usuario:
  1. Vetoriza a pergunta e busca os 10 chunks mais relevantes (k=10).
  2. Concatena o conteudo recuperado no bloco CONTEXTO.
  3. Aplica o prompt (regras de "responder somente pelo contexto").
  4. Chama a LLM e devolve a resposta em texto.
"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_postgres import PGVector

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "openai").lower()
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/rag",
)
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_embeddings():
    """Retorna o modelo de embeddings de acordo com o provedor configurado."""
    if PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model)

    from langchain_openai import OpenAIEmbeddings

    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)


def get_llm():
    """Retorna a LLM (chat model) de acordo com o provedor configurado."""
    if PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite")
        return ChatGoogleGenerativeAI(model=model, temperature=0)

    from langchain_openai import ChatOpenAI

    model = os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano")
    return ChatOpenAI(model=model)


def get_vector_store(pre_delete_collection: bool = False) -> PGVector:
    """Cria o vector store PGVector apontando para a colecao configurada.

    `pre_delete_collection=True` limpa a colecao antes de usar (usado na
    ingestao para tornar o processo idempotente ao reexecutar).
    """
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
        pre_delete_collection=pre_delete_collection,
    )


def _retrieve_context(store: PGVector, question: str) -> str:
    """Busca os 10 chunks mais relevantes e concatena no bloco CONTEXTO."""
    results = store.similarity_search_with_score(question, k=10)
    return "\n\n".join(document.page_content for document, _score in results)


def search_prompt(question=None):
    """Monta o chain de RAG (retrieval + prompt + LLM).

    - Sem argumento (`search_prompt()`) retorna o chain para ser reutilizado
      pelo CLI: `chain.invoke("sua pergunta")` -> resposta em texto.
    - Com `question`, ja executa e retorna a resposta (atalho de conveniencia).
    """
    store = get_vector_store()
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {
            "contexto": RunnableLambda(lambda q: _retrieve_context(store, q)),
            "pergunta": RunnablePassthrough(),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    if question is not None:
        return chain.invoke(question)
    return chain


if __name__ == "__main__":
    # Teste rapido: python src/search.py "sua pergunta"
    import sys

    pergunta = " ".join(sys.argv[1:]).strip() or input("PERGUNTA: ").strip()
    print(search_prompt(pergunta))
