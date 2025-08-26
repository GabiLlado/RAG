# guarda como app.py
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# --- importa tus objetos ya definidos en tu script principal ---
# asumiendo que tienes `chat_chain` y `get_session_history` tal como arriba
from history import chat_chain  # <-- cambia 'your_module' por el nombre de tu archivo .py

st.set_page_config(page_title="RAG Chat", page_icon="💬")
st.title("💬 RAG Chat (Pinecone + LangChain)")

# estado inicial
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "web-session-1"

# muestra historial
for m in st.session_state.messages:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)

# input chat
if user_text := st.chat_input("Escribe tu mensaje..."):
    st.session_state.messages.append(HumanMessage(content=user_text))
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            resp = chat_chain.invoke(
                {"question": user_text},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            st.markdown(resp.content)
            st.session_state.messages.append(AIMessage(content=resp.content))
