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

# =============================
# USERS PERSISTENCE
# =============================
USERS_FILE = BASE_DIR / "users.json"

def load_users():
    if USERS_FILE.exists():
        with open(USERS_FILE, "r") as f:
            st.session_state.users = json.load(f)
    else:
        st.session_state.users = {}

def save_users():
    with open(USERS_FILE, "w") as f:
        json.dump(st.session_state.users, f)

load_users()

# =============================
# COOKIE MANAGER
# =============================
cookie_manager = stx.CookieManager()
current_user_cookie = cookie_manager.get("current_user")
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
            if email and email in users and users[email]["password"] == password:
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
            users[email] = {"password": password, "name": name, "tier": tier}
            save_users()
            st.session_state.authenticated = True
            st.session_state.current_user = email
            cookie_manager.set("current_user", email, expires_at=None)
            st.success(f"Account created! You are now logged in as {name}.")
            st.session_state.show_register = False
            st.rerun()

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
    DATA_PATH = BASE_DIR / "it_docs"
    CHROMA_PATH = BASE_DIR / "chroma_db"

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
# HELPER FUNCTIONS
# =============================
def combine_documents(docs):
    return "\n\n".join(d.page_content for d in docs)

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
        'VPN', 'authentication', 'MFA', 'cached credentials', 'VPN profile', 'credential manager',
        'Outlook', 'mailbox', 'Exchange', 'OST file', 'profile', 'AutoDiscover', 'Cached Exchange Mode',
        'Docker', 'IDE', 'VS Code', 'CPU', 'RAM', 'memory', 'startup apps', 'graphics driver',
        'chipset driver', 'BSOD', 'blue screen', 'minidump', 'hardware replacement', 'SSD',
        'USB', 'peripherals', 'dock', 'firmware', 'power brick', 'boot', 'disk fragmentation', 'sfc',
        'DISM', 'post-update patches', 'boot logging', 'slow boot', 'performance', 'critical driver'
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
    context_docs = retriever.get_relevant_documents(user_query) if retriever else []
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
    # Check if this is a technical issue
    is_technical = is_technical_issue(user_query)
    
    if not is_technical:
        # For casual conversation, no buttons needed
        return get_rag_answer(user_query, is_technical=False), False
    
    # For technical issues, check context
    context_docs = retriever.get_relevant_documents(user_query) if retriever else []
    context_text = combine_documents(context_docs) if context_docs else ""

    if len(context_text) < 200:
        st.session_state.clarification_mode = True
        st.session_state.clarification_index = 0
        st.session_state.clarification_questions = generate_clarification_questions(user_query)
        st.session_state.clarification_answers = []
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
Once the ticket is resovled or cancelled, the chat will reset for new Conversation
""")
st.info("This conversation will be stored in our database.")

# greet
if not st.session_state.greeted:
    current = st.session_state.current_user
    user_name = st.session_state.users[current]["name"] if current else "there"
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
            st.session_state.chat_history.append(HumanMessage("Yes"))
            st.session_state.chat_history.append(AIMessage("🎉 Glad your issue is resolved! Feel free to ask if you have any other questions."))
            st.session_state.chat_history.append(AIMessage("Resetting the chat to start a new conversation!"))
            time.sleep(3)
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
            ticket_id = str(hash(str(st.session_state.chat_history)) % 10000).zfill(4)
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
        
        st.session_state.chat_history.append(HumanMessage(user_query))
        with st.chat_message("Human"):
            st.markdown(user_query)
        with st.chat_message("AI"):
            ai_stream, should_show_buttons = handle_user_query(user_query)
            final = st.write_stream(ai_stream)
            st.session_state.chat_history.append(AIMessage(final))
            st.session_state.show_buttons = should_show_buttons
        st.rerun()