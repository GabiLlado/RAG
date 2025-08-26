import config

from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda

from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from pinecone import Pinecone

# Modelo
llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=config.OPENAI_API_KEY)

# Pinecone
pc = Pinecone(api_key=config.PINECONE_API_KEY)
index = pc.Index(config.INDEX_PINECONE)

embedding = PineconeEmbeddings(
    model=config.embedding_model,
    api_key=config.PINECONE_API_KEY
)

vector_store = PineconeVectorStore(embedding=embedding, index=index, text_key="body")

# Recuperar documentos de Pinecone para usarlos como contexto
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# --- memoria en RAM por sesión ---
_sessions = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _sessions:
        _sessions[session_id] = ChatMessageHistory()
    return _sessions[session_id]

# --- envuelve tu RAG con memoria ---
# Tu cadena base: rag = (RunnableParallel(...) | prompt | llm)
# Defino un prompt con el contexto, poniendo al bot en situación y haciendo la pregunta. Esto es una plantilla.
prompt = ChatPromptTemplate.from_messages([
    ("system", "Responde solo con el contexto. Si falta info, dilo."),
    ("user", "Pregunta: {question}\n\nContexto:\n{context}")
])

# Formateo los documentos para el prompt. De momento no uso los metadatos de las noticias
def format_docs(docs):
    return "\n\n".join(f"• {d.page_content}" for d in docs)

rag = (
    RunnableParallel(
        context=retriever | RunnableLambda(format_docs),
        question=lambda x: x
    )
    | prompt
    | llm
)
chat_chain = RunnableWithMessageHistory(
    rag,
    get_session_history,
    input_messages_key="question",    # mensaje de entrada
    history_messages_key="chat_history"  # nombre del campo que pasaremos al prompt
)

# ⚠️ Ajusta tu prompt para aceptar el historial (no cambia la lógica)
from langchain_core.prompts import MessagesPlaceholder
prompt_with_history = ChatPromptTemplate.from_messages([
    ("system", "Responde solo con el contexto. Si falta info, dilo."),
    MessagesPlaceholder("chat_history"),
    ("user", "Pregunta: {question}\n\nContexto:\n{context}")
])

# vuelve a cablear el tramo final con el nuevo prompt
rag_with_history = (
    # reusamos tu RunnableParallel idéntico
    rag.steps[0]   # mantiene: RunnableParallel(context=..., question=lambda x: x)
    | prompt_with_history
    | llm
)

chat_chain = RunnableWithMessageHistory(
    rag_with_history,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# --- uso en consola (loop sencillo) ---
session_id = "usuario-1"

while True:
    query = input("Tú: ").strip()
    if not query:
        break
    resp = chat_chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": session_id}}
    )
    print("Asistente:", resp.content)
