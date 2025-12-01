# app.py (updated)
# Disable Streamlit file watcher immediately
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_FILE_WATCHER"] = "none"

import streamlit as st

# MUST BE FIRST - Set page config before any other Streamlit commands
st.set_page_config(page_title="IT Support Bot", page_icon="💻")

from dotenv import load_dotenv
from pathlib import Path
import re
import time
import json
import uuid
from datetime import datetime
import extra_streamlit_components as stx

# --- LangChain Core ---
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- LangChain Models & Tools ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# --- Qdrant client for DB (users & tickets) ---
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# =============================
# STREAMLIT STATE INITIALIZATION
# =============================
DEFAULTS = {
    "chat_history": [],
    "clarification_mode": False,
    "clarification_index": 0,
    "clarification_questions": [],
    "clarification_answers": [],
    "awaiting_resolution_confirmation": False,
    "awaiting_technician_confirmation": False,
    "show_buttons": False,
    "greeted": False,
    "technician_assign": False,
    "authenticated": False,
    "show_register": False,
    "users": {},
    "current_user": None,
    "show_reset_countdown": False,
    # Added defaults for ticket tracking
    "current_ticket_id": None,
    "ticket_created": False,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =============================
# LOAD ENV + BASE DIR
# =============================
load_dotenv()
BASE_DIR = Path(__file__).parent.parent
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_KEY:
    st.error("GOOGLE_API_KEY missing in .env file. Add it before running.")
    st.stop()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
# Vector dim for small metadata collections (we use dummy 1D vector)
_METADATA_VECTOR_DIM = 1

# =============================
# QDRANT INITIALIZATION (users & tickets)
# =============================
try:
    qdrant = QdrantClient(url=QDRANT_URL)
except Exception as e:
    st.error(f"Unable to connect to Qdrant at {QDRANT_URL}: {e}")
    st.stop()

# Helper to create metadata collections using 1-d dummy vectors
def ensure_metadata_collection(name: str):
    try:
        qdrant.get_collection(name)
    except Exception:
        # create collection with 1-d vectors (we'll store payloads, vector set to [0.0])
        try:
            qdrant.recreate_collection(
                collection_name=name,
                vectors_config=rest.VectorParams(size=_METADATA_VECTOR_DIM, distance=rest.Distance.COSINE),
            )
        except Exception as e:
            st.error(f"Could not create Qdrant collection '{name}': {e}")
            st.stop()

ensure_metadata_collection("users")
ensure_metadata_collection("tickets")
ensure_metadata_collection("ticket_conversations")

# =============================
# USERS (persisted in Qdrant 'users' collection)
# =============================
def load_users_from_qdrant():
    """Load users from qdrant into session_state.users (map by email)."""
    try:
        users = {}
        # scroll returns (points, next_page)
        points, _ = qdrant.scroll(collection_name="users", limit=1000)
        for point in points:
            payload = point.payload or {}
            email = payload.get("email")
            if email:
                users[email] = payload
        st.session_state.users = users
    except Exception as e:
        st.session_state.users = {}
        st.warning(f"Could not load users from DB: {e}")


def save_user_to_qdrant(user_data: dict):
    """
    Save user to Qdrant:
      - Create a Qdrant-safe UUID as the point id
      - Keep the email inside the payload (so we can map by email on load)
    """
    try:
        # Ensure email present
        email = user_data.get("email")
        if not email:
            raise ValueError("user_data must include 'email'")

        # Use a UUID as the Qdrant point id (Qdrant requires int or UUID)
        point_id = str(uuid.uuid4())
        # Ensure payload contains the email (so load_users_from_qdrant can index by email)
        user_data_copy = dict(user_data)
        user_data_copy["qid"] = point_id
        # Upsert into Qdrant
        qdrant.upsert(
            collection_name="users",
            points=[
                rest.PointStruct(
                    id=point_id,
                    vector=[0.0],  # dummy vector for metadata collection
                    payload=user_data_copy,
                )
            ],
        )
    except Exception as e:
        # Show error but do not crash; caller should handle result
        st.error(f"Failed to save user to DB: {e}")
        raise

# initialize users from qdrant
load_users_from_qdrant()

# =============================
# COOKIE MANAGER
# =============================
cookie_manager = stx.CookieManager()
current_user_cookie = cookie_manager.get("current_user")
# If cookie exists and user exists in loaded users, mark authenticated
if current_user_cookie and current_user_cookie in st.session_state.users:
    st.session_state.authenticated = True
    st.session_state.current_user = current_user_cookie

# =============================
# AUTH SCREEN (LOGIN / REGISTER)
# =============================
def render_auth_ui():
    st.title("🔐 IT Support Portal")

    if not st.session_state.show_register:
        st.subheader("Sign In")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Sign In"):
            users = st.session_state.users
            if email and email in users and users[email].get("password") == password:
                st.session_state.authenticated = True
                st.session_state.current_user = email
                st.session_state.greeted = False
                cookie_manager.set("current_user", email, expires_at=None)
                st.rerun()
            else:
                st.error("Invalid email or password.")

        st.write("Don't have an account?")
        if st.button("Register"):
            st.session_state.show_register = True
            st.rerun()
        return

    # Register view
    st.subheader("Register New Account")
    name = st.text_input("Full Name", key="reg_name")
    email = st.text_input("Email", key="reg_email")
    password = st.text_input("Password", type="password", key="reg_password")
    tier = st.selectbox(
        "Select Tier",
        options=["Staff", "Manager", "Contractor"],
        index=0,
        key="reg_tier"
    )
    if st.button("Create Account"):
        users = st.session_state.users
        if not email or not password or not name or not tier:
            st.error("Please provide name, email, password and select a tier.")
        elif email in users:
            st.error("Account already exists.")
        else:
            user_data = {
                "email": email,
                "password": password,
                "name": name,
                "tier": tier,
                "created_at": datetime.utcnow().isoformat()
            }
            # Save to Qdrant (will raise and show error if fails)
            try:
                save_user_to_qdrant(user_data)
            except Exception:
                # save_user_to_qdrant already displayed the error
                st.stop()

            # Refresh in-memory users map
            load_users_from_qdrant()

            # Ensure the user is present (sanity)
            if email in st.session_state.users:
                st.session_state.authenticated = True
                st.session_state.current_user = email
                cookie_manager.set("current_user", email, expires_at=None)
                st.success(f"Account created! You are now logged in as {name}.")
                st.session_state.show_register = False
                st.rerun()
            else:
                st.error("Account saved but could not load user. Try again or check DB.")
                st.stop()

    if st.button("Back to Login"):
        st.session_state.show_register = False
        st.rerun()

if not st.session_state.authenticated:
    render_auth_ui()
    st.stop()

# =============================
# BUILD KNOWLEDGE BASE (RAG)
# =============================
@st.cache_resource
def setup_rag_pipeline():
    DATA_PATH = BASE_DIR / "src/it_docs"
    CHROMA_PATH = BASE_DIR / "src/chroma_db"

    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        st.warning(f"No documents found in: {DATA_PATH}")
        return None

    loader = DirectoryLoader(str(DATA_PATH), glob="**/*.txt")
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(raw_docs)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=str(CHROMA_PATH)
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    st.success(f"Knowledge Base Ready — {len(docs)} chunks loaded.")
    return retriever

retriever = setup_rag_pipeline()

# =============================
# Utility: Title + Description generator (LLM)
# =============================
def get_title_description(issue_context: str):
    """
    Uses Gemini to generate a title and description for the ticket.
    Ensures we always return both fields.
    """
    template = """
You are an IT Support AI evaluator. Based on the issue described below,
generate a clear ticket title and a detailed but concise description.

Issue:
{context}

Respond in STRICT JSON ONLY:
{
    "title": "",
    "description": ""
}
"""
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        api_key=GEMINI_KEY
    )

    chain = prompt | llm | StrOutputParser()
    try:
        raw = chain.invoke({"context": issue_context}).strip()
    except Exception as e:
        # fallback
        return {"title": "Untitled Issue", "description": issue_context}

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)

    if not json_match:
        return {"title": "Untitled Issue", "description": issue_context}

    try:
        data = json.loads(json_match.group(0))
        return {
            "title": data.get("title", "Untitled Issue"),
            "description": data.get("description", issue_context)
        }
    except Exception:
        return {"title": "Untitled Issue", "description": issue_context}

# =============================
# Conversation payload builder
# =============================
def build_conversation_payload(ticket_id, message, is_user=True):
    """
    Builds the JSON payload for a single conversation message.
    User details are retrieved from st.session_state.
    """
    current_user_email = st.session_state.current_user
    user_data = st.session_state.users.get(current_user_email, {})

    if is_user:
        sender_type = "user"
        sender_id = current_user_email
        sender_name = user_data.get("name", "Unknown User")
    else:
        sender_type = "agent"
        sender_id = "agent_ai_01"
        sender_name = "AI Support Agent"

    conversation_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sender_type": sender_type,
        "sender_id": sender_id,
        "sender_name": sender_name,
        "message": message
    }
    return conversation_payload

# =============================
# TICKETS (persisted in Qdrant 'tickets' collection)
# =============================
def create_ticket():
    """
    Create a placeholder ticket at greet time.
    Title/description will be filled on the first user message.
    """
    current_user_email = st.session_state.current_user
    user_obj = st.session_state.users.get(current_user_email, {})
    user_name = user_obj.get("name", "Unknown")
    ticket_id = str(uuid.uuid4())


    ticket_payload = {
        "ticket_id": ticket_id,
        "title": "",
        "user_id": current_user_email,
        "user_name": user_name,
        "description": "",
        "priority": 2,
        "status": "open",
        "urgency": "medium",
        "category": "general",
        "knowledge_base_id": "",
        "assigned_to": "agent_ai_01",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "resolved_at": "",
        "is_resolved": False
    }

    try:
        qdrant.upsert(
            collection_name="tickets",
            points=[rest.PointStruct(
                id= ticket_id ,
                vector=[0.0],
                payload=ticket_payload
            )]
        )
        # initialize the conversation doc in ticket_conversations collection
        initialize_ticket_conversation(ticket_id)
    except Exception as e:
        st.error(f"Failed to create ticket in DB: {e}")
        return None

    st.session_state.current_ticket_id = ticket_id
    st.session_state.ticket_created = True
    return ticket_payload

def get_all_tickets():
    tickets = []
    try:
        points, _ = qdrant.scroll(collection_name="tickets", limit=1000)
        for p in points:
            tickets.append(p.payload)
    except Exception as e:
        st.error(f"Failed to fetch tickets: {e}")
    return tickets

def update_ticket_metadata(ticket_id: str, updates: dict):
    """
    Update (partial) fields of the ticket with ticket_id.
    """
    try:
        points, _ = qdrant.scroll(collection_name="tickets", limit=1000)
        for p in points:
            payload = p.payload or {}
            if payload.get("ticket_id") == ticket_id:
                payload.update(updates)
                payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
                qdrant.upsert(
                    collection_name="tickets",
                    points=[rest.PointStruct(
                        id=p.id,
                        vector=[0.0],
                        payload=payload
                    )]
                )
                return payload
    except Exception as e:
        st.error(f"Failed to update ticket: {e}")
    return None

def update_ticket_status(ticket_id: str, new_status: str):
    """
    Update only status (and mark resolved fields when appropriate).
    """
    updates = {"status": new_status}
    if new_status.lower() in ("resolved", "closed", "closed_by_user"):
        updates["is_resolved"] = True
        updates["resolved_at"] = datetime.utcnow().isoformat() + "Z"
    return update_ticket_metadata(ticket_id, updates)

# =============================
# TICKET CONVERSATIONS (one-document per ticket)
# =============================
def initialize_ticket_conversation(ticket_id):
    payload = {
        "ticket_id": ticket_id,
        "conversation": [],
        "events": []
    }
    try:
        qdrant.upsert(
            collection_name="ticket_conversations",
            points=[rest.PointStruct(
                id=ticket_id,
                vector=[0.0],
                payload=payload
            )]
        )
    except Exception as e:
        st.error(f"Failed to initialize ticket conversation: {e}")

def get_ticket_conversation(ticket_id):
    try:
        points, _ = qdrant.scroll(collection_name="ticket_conversations", limit=1000)
        for p in points:
            payload = p.payload or {}
            if payload.get("ticket_id") == ticket_id:
                return payload
    except Exception as e:
        st.error(f"Failed to read ticket conversation: {e}")
    return None

def add_conversation_message(ticket_id, message_payload):
    """
    Append a message dict to the 'conversation' array of the ticket_conversations document.
    """
    try:
        points, _ = qdrant.scroll(collection_name="ticket_conversations", limit=1000)
        for p in points:
            payload = p.payload or {}
            if payload.get("ticket_id") == ticket_id:
                payload.setdefault("conversation", []).append(message_payload)
                qdrant.upsert(
                    collection_name="ticket_conversations",
                    points=[rest.PointStruct(
                        id=p.id,
                        vector=[0.0],
                        payload=payload
                    )]
                )
                return payload
        # if not found, initialize and insert
        initialize_ticket_conversation(ticket_id)
        # append again
        payload = {
            "ticket_id": ticket_id,
            "conversation": [message_payload],
            "events": []
        }
        qdrant.upsert(
            collection_name="ticket_conversations",
            points=[rest.PointStruct(
                id=ticket_id,
                vector=[0.0],
                payload=payload
            )]
        )
        return payload
    except Exception as e:
        st.error(f"Failed to append conversation message: {e}")
        return None

def add_ticket_event(ticket_id, event_type, actor_type, actor_id, message):
    event_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "actor_type": actor_type,
        "actor_id": actor_id,
        "message": message
    }
    try:
        points, _ = qdrant.scroll(collection_name="ticket_conversations", limit=1000)
        for p in points:
            payload = p.payload or {}
            if payload.get("ticket_id") == ticket_id:
                payload.setdefault("events", []).append(event_payload)
                qdrant.upsert(
                    collection_name="ticket_conversations",
                    points=[rest.PointStruct(
                        id=p.id,
                        vector=[0.0],
                        payload=payload
                    )]
                )
                return payload
        # If not found, initialize doc with this event
        payload = {
            "ticket_id": ticket_id,
            "conversation": [],
            "events": [event_payload]
        }
        qdrant.upsert(
            collection_name="ticket_conversations",
            points=[rest.PointStruct(
                id=ticket_id,
                vector=[0.0],
                payload=payload
            )]
        )
        return payload
    except Exception as e:
        st.error(f"Failed to append ticket event: {e}")
        return None

# =============================
# HELPER FUNCTIONS (existing)
# =============================
def combine_documents(docs):
    return "\n\n".join(d.page_content for p in docs for d in ([p] if hasattr(p, "page_content") else [])) if docs else ""

def reset_chat():
    """Reset chat to start a new conversation"""
    st.session_state.chat_history = []
    st.session_state.clarification_mode = False
    st.session_state.clarification_index = 0
    st.session_state.clarification_questions = []
    st.session_state.clarification_answers = []
    st.session_state.awaiting_resolution_confirmation = False
    st.session_state.awaiting_technician_confirmation = False
    st.session_state.show_buttons = False
    st.session_state.technician_assign = False
    st.session_state.greeted = False
    st.session_state.show_reset_countdown = False
    st.session_state.current_ticket_id = None
    st.session_state.ticket_created = False

def is_technical_issue(user_query):
    """Detect if user query is a technical issue that needs resolution"""
    casual_patterns = [
        r'\b(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b',
        r'\b(how are you|what\'s up|wassup)\b',
        r'\b(thank you|thanks|thank|cheers)\b',
        r'\b(bye|goodbye|see you|later)\b',
        r'\b(ok|okay|alright|cool|nice)\b'
    ]
    
    query_lower = user_query.lower().strip()
    
    # Check if it's just a casual message
    for pattern in casual_patterns:
        if re.search(pattern, query_lower) and len(query_lower.split()) <= 5:
            return False
    
    # Check for technical keywords
    technical_keywords = [
        'error', 'issue', 'problem', 'not working', 'can\'t', 'cannot', 'won\'t',
        'unable', 'failed', 'crash', 'bug', 'broken', 'slow', 'freeze', 'stuck',
        'help with', 'fix', 'solve', 'troubleshoot', 'install', 'configure',
        'reset', 'password', 'login', 'access', 'connection', 'network', 'syncing', 'migration',
        'vpn', 'authentication', 'mfa', 'cached credentials', 'vpn profile', 'credential manager',
        'outlook', 'mailbox', 'exchange', 'ost file', 'profile', 'autodiscover', 'cached exchange mode',
        'docker', 'ide', 'vs code', 'cpu', 'ram', 'memory', 'startup apps', 'graphics driver',
        'chipset driver', 'bsod', 'blue screen', 'minidump', 'hardware replacement', 'ssd',
        'usb', 'peripherals', 'dock', 'firmware', 'power brick', 'boot', 'disk fragmentation', 'sfc',
        'dism', 'post-update patches', 'boot logging', 'slow boot', 'performance', 'critical driver'
    ]
    
    return any(keyword in query_lower for keyword in technical_keywords)

def generate_clarification_questions(user_query):
    st.session_state.show_buttons = False
    st.session_state.awaiting_resolution_confirmation = False

    template = """
You are an IT Support Technician. The user has described an issue: "{user_question}".
Your knowledge base returned insufficient context.
Generate 5 specific, numbered diagnostic questions (Q1-Q5) to clarify the problem.
Do NOT provide a solution.
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, api_key=GEMINI_KEY)
    chain = ({"user_question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    questions_str = chain.invoke({"user_question": user_query})
    questions = re.findall(r'Q\d+[:.)]\s*(.*)', questions_str)
    if not questions:
        questions = [q.strip() for q in questions_str.split("\n") if q.strip()]
    return questions[:5]

def get_rag_answer(user_query, context_override=None, is_technical=True):
    # Use vectorstore similarity search directly if available
    context_docs = retriever.vectorstore.similarity_search(user_query, k=3) if retriever else []
    context_text = combine_documents(context_docs) if context_docs else ""
    if context_override:
        context_text += "\n\n" + context_override

    if is_technical:
        template = """
You are an IT Support Technician. Answer using CONTEXT first. 
If the context has the resolution steps then just display them and mention the source from the KB.
If context is insufficient, be professional and helpful.
Provide step-by-step troubleshooting when addressing technical issues.

CONTEXT:
--------- 
{context}
---------

Chat history: {chat_history}
User question: {user_question}
        """
    else:
        template = """
You are a friendly IT Support Assistant. Respond naturally to the user's message.
Be conversational and helpful for general queries.

Chat history: {chat_history}
User message: {user_question}
        """
    
    rag_prompt = ChatPromptTemplate.from_template(template)
    chat_hist_text = "\n".join(msg.content for msg in st.session_state.chat_history)

    rag_chain = (
        {
            "context": lambda _: context_text if is_technical else "",
            "user_question": RunnablePassthrough(),
            "chat_history": lambda _: chat_hist_text,
        }
        | rag_prompt
        | ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, api_key=GEMINI_KEY)
        | StrOutputParser()
    )
    return rag_chain.stream({"user_question": user_query})

def handle_user_query(user_query):
    ticket_id = st.session_state.get("current_ticket_id")
    ticket_payload = None

    if ticket_id:
        points, _ = qdrant.scroll(collection_name="tickets", limit=1000)
        for p in points:
            payload = p.payload or {}
            if payload.get("ticket_id") == ticket_id:
                ticket_payload = payload
                break

    # First significant user message: generate title/description if empty
    if ticket_payload and not ticket_payload.get("title"):
        td = get_title_description(user_query)
        update_ticket_metadata(ticket_id, {
            "title": td["title"],
            "description": td["description"],
            "created_at": ticket_payload.get("created_at", datetime.utcnow().isoformat() + "Z")
        })
        add_ticket_event(ticket_id, "created", "system", "system", "Ticket created from initial user message.")

    # Append user's message to conversation
    if ticket_id:
        user_msg_payload = build_conversation_payload(ticket_id, user_query, is_user=True)
        add_conversation_message(ticket_id, user_msg_payload)
        add_ticket_event(ticket_id, "message", "user", st.session_state.current_user, user_query)

    # Rest of your logic for RAG, clarifications, etc...

    # Check technical classification
    is_technical = is_technical_issue(user_query)
    
    if not is_technical:
        # For casual conversation, no buttons needed
        return get_rag_answer(user_query, is_technical=False), False
    
    # For technical issues, check context
    context_docs = retriever.vectorstore.similarity_search(user_query, k=3) if retriever else []
    context_text = combine_documents(context_docs) if context_docs else ""

    # If KB context insufficient, trigger clarifications
    if len(context_text) < 200:
        st.session_state.clarification_mode = True
        st.session_state.clarification_index = 0
        st.session_state.clarification_questions = generate_clarification_questions(user_query)
        st.session_state.clarification_answers = []
        # return first clarification question
        return iter([st.session_state.clarification_questions[0]]), False
    
    st.session_state.clarification_mode = False
    return get_rag_answer(user_query, is_technical=True), True

def evaluate_ai_capability(issue_context):
    """Evaluate if AI can solve the issue or if human intervention is needed"""
    template = """
You are an IT Support AI evaluator. Based on the conversation history and issue context, determine if this issue can be resolved by an AI agent or requires human technician intervention.

Context: {context}

Respond with ONLY ONE WORD:
- "AI_SOLVABLE" if the issue can be resolved through automated steps, scripts, or configuration changes
- "HUMAN_NEEDED" if the issue requires physical access, hardware replacement, complex judgment, or escalated permissions

Your response:""" 

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, api_key=GEMINI_KEY)
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"context": issue_context}).strip().upper()
    
    # Extract the first word only
    first_word = result.split()[0] if result else ""
    
    return first_word == "AI_SOLVABLE"

def get_next_clarification():
    idx = st.session_state.clarification_index + 1
    if idx < len(st.session_state.clarification_questions):
        st.session_state.clarification_index = idx
        return iter([st.session_state.clarification_questions[idx]])
    st.session_state.clarification_mode = False
    context_override = "User clarification answers:\n" + "\n".join(st.session_state.clarification_answers)
    return get_rag_answer("Final comprehensive diagnosis and solution", context_override=context_override)

# =============================
# CHATBOT UI
# =============================
st.title("IT Support Chat-Bot")
st.info("""
A ticket will be created for each conversation.
Once the ticket is resolved or cancelled, the chat will reset for new Conversation
""")
st.info("This conversation will be stored in our database.")

# greet
# greet
if not st.session_state.greeted:
    current = st.session_state.current_user
    user_obj = st.session_state.users.get(current)
    user_name = user_obj.get("name", "there") if user_obj else "there"
    
    # Create ticket only if it doesn't exist
    if not st.session_state.current_ticket_id:
        ticket_payload = create_ticket()
        # add creation event
        if ticket_payload:
            add_ticket_event(ticket_payload["ticket_id"], "created", "system", "system", "Ticket placeholder created at greet.")
    
    greeting = f"Hello {user_name}! 👋 I'm your IT Support Assistant. How can I help you today?"
    st.session_state.chat_history.append(AIMessage(greeting))
    st.session_state.greeted = True


# display chat history
for msg in st.session_state.chat_history:
    role = "Human" if isinstance(msg, HumanMessage) else "AI"
    with st.chat_message(role):
        st.markdown(msg.content)

# --- Clarification mode ---
if st.session_state.clarification_mode:
    next_q = st.session_state.clarification_questions[st.session_state.clarification_index]
    with st.chat_message("AI"):
        st.markdown(next_q)
    ans = st.chat_input("Answer the clarification question:")
    if ans:
        st.session_state.chat_history.append(HumanMessage(ans))
        st.session_state.clarification_answers.append(ans)
        with st.chat_message("AI"):
            ai_stream = get_next_clarification()
            final = st.write_stream(ai_stream)
            st.session_state.chat_history.append(AIMessage(final))
            st.session_state.show_buttons = True
        st.rerun()

# --- Resolution buttons ---
elif st.session_state.show_buttons:
    st.markdown("---")
    with st.chat_message("AI"):
        st.markdown("**Was your issue resolved?**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes — Resolved"):
            st.session_state.show_buttons = False
            # Update the existing ticket status to resolved
            ticket_id = st.session_state.get("current_ticket_id")
            if ticket_id:
                update_ticket_status(ticket_id, "resolved")
                add_ticket_event(ticket_id, "resolved", "agent", "agent_ai_01", "Ticket resolved by agent/AI.")
            st.session_state.chat_history.append(HumanMessage("Yes"))
            st.session_state.chat_history.append(AIMessage("🎉 Glad your issue is resolved! Feel free to ask if you have any other questions."))
            st.session_state.chat_history.append(AIMessage("Resetting the chat to start a new conversation!"))
            time.sleep(2)
            st.session_state.awaiting_technician_confirmation = False
            reset_chat()
            st.rerun()
    with col2:
        if st.button("❌ No — Not resolved"):
            st.session_state.show_buttons = False
            st.session_state.chat_history.append(HumanMessage("No"))
            
            # Evaluate if AI can solve the issue
            chat_context = "\n".join([msg.content for msg in st.session_state.chat_history[-10:]])
            can_ai_solve = evaluate_ai_capability(chat_context)
            
            if can_ai_solve:
                st.session_state.awaiting_resolution_confirmation = True
            else:
                st.session_state.awaiting_technician_confirmation = True
            
            st.rerun()

# --- AI resolution confirmation ---
elif st.session_state.awaiting_resolution_confirmation:
    st.markdown("---")
    with st.chat_message("AI"):
        st.markdown("**AI can attempt an automatic fix. Do you want to proceed?**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Proceed with AI fix"):
            st.session_state.chat_history.append(HumanMessage("Proceed with AI fix"))
            st.session_state.chat_history.append(AIMessage("Proceeding with AI fix"))
            with st.chat_message("AI"):
                st.session_state.chat_history.append(AIMessage("Please wait while the chat-bot is running System Automation Checks")) 
            st.session_state.show_buttons = True
            st.session_state.awaiting_resolution_confirmation = False
            st.rerun()
    with col2:
        if st.button("❌ Don't proceed"):
            st.session_state.chat_history.append(HumanMessage("No, don't proceed with AI fix"))
            st.session_state.chat_history.append(AIMessage("Okay, Would you like to escalate this issue to a human technician?"))
            st.session_state.awaiting_resolution_confirmation = False
            st.session_state.awaiting_technician_confirmation = True
            st.rerun()

# --- Technician escalation confirmation ---
elif st.session_state.awaiting_technician_confirmation:
    st.markdown("---")
    with st.chat_message("AI"):
        st.markdown("**Would you like to escalate this issue to a human technician?**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes — Escalate to Technician"):
            st.session_state.chat_history.append(HumanMessage("Yes, escalate to technician"))
            # Update ticket status to escalated and persist conversation snapshot
            ticket_id = st.session_state.get("current_ticket_id")
            if ticket_id:
                update_ticket_status(ticket_id, "escalated")
                add_ticket_event(ticket_id, "assigned", "system", "system", "Assigned to human technician")
            ticket_record = {"ticket_id": ticket_id}
            st.session_state.chat_history.append(AIMessage(f"✅ Your issue has been escalated to our technical team. A technician will contact you shortly. Ticket ID: #{ticket_id}"))
            st.session_state.technician_assign = True
            st.session_state.awaiting_technician_confirmation = False
            st.rerun()
    with col2:
        if st.button("❌ No — Start New Chat"):
            st.session_state.chat_history.append(HumanMessage("No, start a new chat"))
            st.session_state.chat_history.append(AIMessage("Understood! Let's start fresh. How can I help you today?"))
            st.session_state.awaiting_technician_confirmation = False
            reset_chat()
            st.rerun()

# --- After technician assignment ---
elif st.session_state.technician_assign:
    st.markdown("---")
    st.success("🎫 Your ticket has been created. A technician will reach out soon.")
    if st.button("🔄 Start New Chat"):
        reset_chat()
        st.rerun()

# --- Normal chat input ---
else:
    user_query = st.chat_input("Ask something about IT…")
    if user_query and user_query.strip():
        # Reset escalation states when new query comes in
        st.session_state.awaiting_resolution_confirmation = False
        st.session_state.awaiting_technician_confirmation = False
        st.session_state.technician_assign = False
        
        # Append to chat history (UI)
        st.session_state.chat_history.append(HumanMessage(user_query))
        with st.chat_message("Human"):
            st.markdown(user_query)

        # handle user query (this will also create/update ticket title/description and append conversation)
        with st.chat_message("AI"):
            ai_stream, should_show_buttons = handle_user_query(user_query)
            final = st.write_stream(ai_stream)
            # Save AI reply to conversation as well
            st.session_state.chat_history.append(AIMessage(final))
            # Append agent message to conversation doc
            ticket_id = st.session_state.get("current_ticket_id")
            if ticket_id:
                agent_msg_payload = build_conversation_payload(ticket_id, final, is_user=False)
                add_conversation_message(ticket_id, agent_msg_payload)
                add_ticket_event(ticket_id, "agent_reply", "agent", "agent_ai_01", final)
            st.session_state.show_buttons = should_show_buttons
        st.rerun()
