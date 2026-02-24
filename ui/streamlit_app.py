"""
Streamlit Chat Interface for RAG Chatbot.

A clean, modern chat UI with authentication support.
"""

import streamlit as st
import httpx
import time
from typing import Optional

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e3f2fd;
    }
    .chat-message.assistant {
        background-color: #f5f5f5;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .user-badge {
        background-color: #4caf50;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


def get_auth_header() -> dict:
    """Get authorization header if logged in."""
    if "access_token" in st.session_state and st.session_state.access_token:
        return {"Authorization": f"Bearer {st.session_state.access_token}"}
    return {}


def check_api_health() -> dict:
    """Check if API is healthy."""
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json()
    except Exception:
        return {"status": "unavailable", "services": {}}


def register_user(username: str, password: str, email: str = None) -> Optional[dict]:
    """Register a new user."""
    try:
        data = {"username": username, "password": password}
        if email:
            data["email"] = email
        response = httpx.post(
            f"{API_BASE_URL}/api/v1/auth/register",
            json=data,
            timeout=10,
        )
        if response.status_code == 201:
            return response.json()
        else:
            error = response.json()
            st.error(f"Registration failed: {error.get('detail', {}).get('error', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Registration failed: {e}")
        return None


def login_user(username: str, password: str) -> Optional[dict]:
    """Login and get access token."""
    try:
        response = httpx.post(
            f"{API_BASE_URL}/api/v1/auth/login",
            data={"username": username, "password": password},
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Invalid username or password")
            return None
    except Exception as e:
        st.error(f"Login failed: {e}")
        return None


def get_document_stats() -> Optional[dict]:
    """Get document statistics."""
    try:
        response = httpx.get(
            f"{API_BASE_URL}/api/v1/documents/stats",
            headers=get_auth_header(),
            timeout=5,
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def ingest_documents(documents: list[dict]) -> Optional[dict]:
    """Ingest documents into the knowledge base."""
    try:
        response = httpx.post(
            f"{API_BASE_URL}/api/v1/documents/ingest",
            json={"documents": documents},
            headers=get_auth_header(),
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        error = response.json()
        st.error(f"Ingestion failed: {error}")
        return None
    except Exception as e:
        st.error(f"Failed to ingest documents: {e}")
        return None


def upload_file(file) -> Optional[dict]:
    """Upload a file to the knowledge base."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = httpx.post(
            f"{API_BASE_URL}/api/v1/documents/upload",
            files=files,
            headers=get_auth_header(),
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()
        error = response.json()
        st.error(f"Upload failed: {error.get('detail', {}).get('error', 'Unknown error')}")
        return None
    except Exception as e:
        st.error(f"Failed to upload file: {e}")
        return None


def chat(message: str, use_rag: bool = True) -> Optional[dict]:
    """Send a chat message."""
    endpoint = "/api/v1/chat" if use_rag else "/api/v1/chat/direct"
    try:
        response = httpx.post(
            f"{API_BASE_URL}{endpoint}",
            json={"message": message, "include_sources": True},
            headers=get_auth_header(),
            timeout=120,
        )
        data = response.json()

        if response.status_code != 200:
            error_msg = data.get("detail", {}).get("error", str(data))
            st.error(f"API Error: {error_msg}")
            return None

        return data
    except httpx.ConnectError:
        st.error("Cannot connect to API. Make sure uvicorn is running.")
        return None
    except Exception as e:
        st.error(f"Chat failed: {e}")
        return None


def clear_documents() -> bool:
    """Clear all documents."""
    try:
        response = httpx.delete(
            f"{API_BASE_URL}/api/v1/documents",
            headers=get_auth_header(),
            timeout=10,
        )
        return response.status_code == 200
    except Exception:
        return False


if "messages" not in st.session_state:
    st.session_state.messages = []

if "use_rag" not in st.session_state:
    st.session_state.use_rag = True

if "access_token" not in st.session_state:
    st.session_state.access_token = None

if "username" not in st.session_state:
    st.session_state.username = None

if "user_id" not in st.session_state:
    st.session_state.user_id = None


with st.sidebar:
    st.title("🤖 RAG Chatbot")
    st.markdown("---")

    if st.session_state.access_token:
        st.success(f"👤 Logged in as: **{st.session_state.username}**")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.access_token = None
            st.session_state.username = None
            st.session_state.user_id = None
            st.session_state.messages = []
            st.rerun()
    else:
        auth_tab, register_tab = st.tabs(["🔐 Login", "📝 Register"])

        with auth_tab:
            login_username = st.text_input("Username", key="login_user")
            login_password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", use_container_width=True, key="login_btn"):
                if login_username and login_password:
                    result = login_user(login_username, login_password)
                    if result:
                        st.session_state.access_token = result["access_token"]
                        st.session_state.username = result["username"]
                        st.session_state.user_id = result["user_id"]
                        st.rerun()
                else:
                    st.warning("Please enter username and password")

        with register_tab:
            reg_username = st.text_input("Username", key="reg_user")
            reg_email = st.text_input("Email (optional)", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_pass")
            reg_password2 = st.text_input("Confirm Password", type="password", key="reg_pass2")

            if st.button("Register", use_container_width=True, key="reg_btn"):
                if not reg_username or not reg_password:
                    st.warning("Username and password are required")
                elif reg_password != reg_password2:
                    st.warning("Passwords do not match")
                elif len(reg_password) < 6:
                    st.warning("Password must be at least 6 characters")
                else:
                    result = register_user(reg_username, reg_password, reg_email or None)
                    if result:
                        st.toast("Registration successful! Please login.")

    st.markdown("---")

    health = check_api_health()

    st.subheader("System Status")
    api_status = "🟢" if health.get("status") in ["healthy", "degraded"] else "🔴"
    ollama_status = "🟢" if health.get("services", {}).get("ollama") == "healthy" else "🔴"

    st.markdown(f"**API:** {api_status} {health.get('status', 'unavailable')}")
    st.markdown(f"**Ollama:** {ollama_status} {health.get('services', {}).get('ollama', 'unavailable')}")

    if st.session_state.access_token:
        st.markdown("---")

        stats = get_document_stats()
        if stats:
            st.subheader("Your Knowledge Base")
            st.markdown(f"**Documents:** {stats.get('document_count', 0)}")
            st.markdown(f"**Collection:** {stats.get('name', 'N/A')}")

        st.markdown("---")

        st.subheader("Settings")
        st.session_state.use_rag = st.toggle("Use RAG (Knowledge Base)", value=True)

        if not st.session_state.use_rag:
            st.info("RAG disabled. Chatting directly with LLM.")

        st.markdown("---")

        st.subheader("Upload Documents")

        upload_tab, paste_tab = st.tabs(["📁 Upload File", "📝 Paste Text"])

        with upload_tab:
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["pdf", "txt", "md"],
                help="Supported: PDF, TXT, Markdown",
            )

            if uploaded_file is not None:
                st.info(f"📄 {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

                if st.button("⬆️ Upload & Process", use_container_width=True, key="upload_btn"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        result = upload_file(uploaded_file)
                        if result:
                            st.success(f"Added {result.get('chunk_count', 0)} chunks!")
                            st.rerun()

        with paste_tab:
            uploaded_text = st.text_area(
                "Paste document content:",
                height=150,
                placeholder="Paste your document text here...",
            )

            doc_source = st.text_input("Source name (optional):", placeholder="e.g., manual, faq")

            if st.button("📄 Add Document", use_container_width=True, key="paste_btn"):
                if uploaded_text.strip():
                    with st.spinner("Ingesting document..."):
                        result = ingest_documents([{
                            "content": uploaded_text,
                            "metadata": {"source": doc_source or "user-upload"}
                        }])
                        if result:
                            st.success(f"Added {result.get('chunk_count', 0)} chunks!")
                            st.rerun()
                else:
                    st.warning("Please enter some text.")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Docs", use_container_width=True):
                if clear_documents():
                    st.success("Documents cleared!")
                    st.rerun()
        with col2:
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()


st.title("💬 Chat")

if health.get("status") == "unavailable":
    st.error("""
    **API is not running!**

    Start the API with:
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    """)
    st.stop()

if not st.session_state.access_token:
    st.warning("👈 Please **login** or **register** in the sidebar to start chatting.")
    st.info("""
    **Why authentication?**
    
    Each user has their own private knowledge base. Your documents and chat history 
    are isolated from other users.
    
    **Quick start:**
    1. Register a new account in the sidebar
    2. Login with your credentials
    3. Upload documents to your knowledge base
    4. Start chatting!
    """)
    st.stop()

if health.get("services", {}).get("ollama") != "healthy":
    st.warning("""
    **Ollama is not available.** Make sure Ollama is running on Windows with:
    ```powershell
    $env:OLLAMA_HOST="0.0.0.0:11434"
    ollama serve
    ```
    """)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for i, source in enumerate(msg["sources"], 1):
                    score = source.get('score', 0)
                    content = source.get('content', '')[:200]
                    st.markdown(f"**[{i}]** (score: {score:.2f})")
                    st.markdown(f"> {content}...")

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start_time = time.time()
            response = chat(prompt, use_rag=st.session_state.use_rag)
            elapsed = time.time() - start_time

        if response and "message" in response:
            st.markdown(response["message"])

            sources = response.get("sources", [])
            if sources:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(sources, 1):
                        score = source.get('score', 0)
                        content = source.get('content', '')[:200]
                        st.markdown(f"**[{i}]** (score: {score:.2f})")
                        st.markdown(f"> {content}...")

            st.caption(f"⏱️ {elapsed:.1f}s | {'RAG' if st.session_state.use_rag else 'Direct'}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response["message"],
                "sources": sources,
            })
        elif response:
            st.error(f"Unexpected response: {response}")
        else:
            st.error("Failed to get response. Check if API and Ollama are running.")
