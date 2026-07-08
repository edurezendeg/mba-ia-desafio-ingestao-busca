# Desafio MBA Engenharia de Software com IA - Full Cycle

Ingestão e busca semântica (**RAG**) com **LangChain** e **PostgreSQL + pgVector**:

1. **Ingestão** — lê um PDF, divide em *chunks*, gera *embeddings* e armazena os
   vetores no banco.
2. **Busca** — recebe perguntas via **CLI** e responde **somente** com base no
   conteúdo do PDF. Se a informação não estiver no documento, retorna:
   `Não tenho informações necessárias para responder sua pergunta.`

## Estrutura

```
.
├── docker-compose.yml     # PostgreSQL + pgVector (cria a extensão vector)
├── requirements.txt       # Dependências (versões fixas)
├── .env.example           # Template das variáveis de ambiente
├── document.pdf           # PDF utilizado na ingestão
├── src/
│   ├── ingest.py          # Ingestão do PDF -> banco vetorial
│   ├── search.py          # Config compartilhada + chain de busca (retrieval + prompt + LLM)
│   └── chat.py            # CLI de perguntas e respostas
└── README.md
```

## Pré-requisitos

- Python 3.12+
- Docker e Docker Compose
- Uma **API Key** da OpenAI **ou** do Google Gemini

## Passo a passo

### 1. Ambiente virtual e dependências

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Ubuntu/Debian/WSL:** se `python3 -m venv` falhar com
> `ensurepip is not available`, instale antes: `sudo apt install python3.12-venv`.

### 2. Variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env`:

- **OpenAI (padrão):** deixe `PROVIDER=openai` e preencha `OPENAI_API_KEY`.
- **Gemini:** troque para `PROVIDER=gemini` e preencha `GOOGLE_API_KEY`.

A `DATABASE_URL` já vem apontando para o banco do `docker-compose.yml`.

### 3. Subir o banco de dados

```bash
docker compose up -d
```

Sobe o PostgreSQL com pgVector na porta `5432`. O serviço `bootstrap_vector_ext`
cria a extensão `vector` automaticamente após o banco ficar saudável.

### 4. Ingestão do PDF

```bash
python src/ingest.py
```

O PDF é dividido em *chunks* de **1000** caracteres com **150** de *overlap*,
convertido em *embeddings* e armazenado no banco. A ingestão é **idempotente**:
reexecutar limpa a coleção antes de inserir novamente.

### 5. Rodar o chat

```bash
python src/chat.py
```

## Exemplo de uso

```
Faça sua pergunta:

PERGUNTA: Qual o faturamento da empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

---

Faça sua pergunta:

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

> Para testar uma pergunta única sem abrir o chat:
> `python src/search.py "Qual o faturamento da SuperTechIABrazil?"`

## Como funciona (fluxo da busca)

1. A pergunta é vetorizada com o mesmo modelo de *embeddings* da ingestão.
2. `similarity_search_with_score(query, k=10)` recupera os 10 *chunks* mais
   relevantes do pgVector.
3. O conteúdo recuperado é concatenado no bloco `CONTEXTO` de um prompt com
   regras estritas (responder somente pelo contexto).
4. A LLM gera a resposta final, exibida no CLI.

## Trocar de provedor (OpenAI ↔ Gemini)

| Variável | OpenAI | Gemini |
|---|---|---|
| `PROVIDER` | `openai` | `gemini` |
| Embeddings | `text-embedding-3-small` | `models/embedding-001` |
| LLM | `gpt-5-nano` | `gemini-2.5-flash-lite` |

> Os *embeddings* da ingestão e da busca precisam ser do **mesmo
> provedor/modelo** (as dimensões dos vetores diferem). Ao trocar de provedor,
> rode a ingestão novamente.

## Reset do banco

```bash
docker compose down -v   # remove o volume (apaga os dados) e recria do zero
```
