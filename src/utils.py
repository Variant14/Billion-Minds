import streamlit as st
import os
import re
import json
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_pipeline import combine_documents
from src.qdrant import (
    retrieve_ticket_byid
)

# NOTE: GEMINI_KEY will be passed from app.py to handle_user_query

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
    # [Rest of the is_technical_issue function code remains the same]
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
        # General Failure & Troubleshooting
        'error', 'issue', 'problem', 'not working', 'can\'t', 'cannot', 'won\'t',
        'unable', 'failed', 'crash', 'bug', 'broken', 'slow', 'freeze', 'stuck',
        'disconnects', 'help with', 'fix', 'solve', 'troubleshoot', 'install',
        'configure', 'reset', 'performance', 'ticket', 'service desk',

        # Access, Security & Credentials
        'password', 'login', 'access', 'authentication', 'MFA', 'credential manager',
        'cached credentials', 'access denied', 'permission', 'local admin',
        'phishing', 'compromised account', 'encryption', 'token', 'key', 'two-factor', '2FA',
        'licensing', 'compliance',

        # Networking & Connectivity
        'connection', 'network', 'VPN', 'VPN profile', 'IP address', 'DNS', 'DHCP',
        'router', 'switch', 'latency', 'bandwidth', 'Proxy', 'LAN', 'WAN', 'WLAN',
        'Firewall', 'network security',

        # Email & Collaboration (Outlook/Exchange/Cloud)
        'Outlook', 'mailbox', 'Exchange', 'OST file', 'profile', 'AutoDiscover',
        'Cached Exchange Mode', 'syncing', 'migration', 'SharePoint', 'OneDrive',
        'Teams', 'Office 365', 'cloud', 'SaaS', 'IaaS',

        # Hardware & Peripherals
        'CPU', 'RAM', 'memory', 'startup apps', 'graphics driver', 'chipset driver',
        'BSOD', 'blue screen', 'minidump', 'hardware replacement', 'SSD', 'USB',
        'peripherals', 'dock', 'firmware', 'power brick', 'monitor', 'display',
        'resolution', 'keyboard', 'mouse', 'webcam', 'printer', 'scanner',
        'driver signature', 'laptop', 'desktop',

        # Operating System & System Utilities
        'boot', 'disk fragmentation', 'sfc', 'DISM', 'post-update patches',
        'boot logging', 'slow boot', 'critical driver', 'Windows', 'macOS', 'Linux',
        'registry', 'Group Policy', 'GPO', 'Active Directory', 'Azure AD',
        'application', 'software', 'program', 'update', 'patch', 'rollback',
        'virtual machine', 'VM', 'hypervisor', 'antivirus', 'malware',

        # Development & Specialized Tools
        'Docker', 'IDE', 'VS Code', 'GitHub', 'Kubernetes', 'scripting', 'CI/CD',

        # Mobile & Remote Computing
        'mobile device', 'phone', 'tablet', 'iOS', 'Android', 'MDM',
        'VDI', 'Citrix'
    ]
    
    return any(keyword in query_lower for keyword in technical_keywords)


def generate_clarification_questions(user_query, GEMINI_KEY): # <-- Key added
    """Generates 5 diagnostic questions for clarification."""
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

def get_rag_answer(user_query, retriever, GEMINI_KEY, context_override=None, is_technical=True):
    """Retrieves relevant context and generates an AI answer."""
    context_docs = retriever.get_relevant_documents(user_query) if retriever else []
    context_text = combine_documents(context_docs) if context_docs else ""
    if context_override:
        context_text += "\n\n" + context_override

    if is_technical:
        template = """
You are an IT Support Technician. Answer using CONTEXT first. 
If the context has the resolution steps then just display them and mention the source from the KB.
If context is insufficient, be professional and helpful.
Provide step-by-step troubleshooting when addressing technical issues and clearly state that this scenario isnt in the Knowledge base

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

def get_next_clarification(retriever, GEMINI_KEY):
    """Gets the next clarification question or final RAG answer."""
    idx = st.session_state.clarification_index + 1
    if idx < len(st.session_state.clarification_questions):
        st.session_state.clarification_index = idx
        return iter([st.session_state.clarification_questions[idx]]), False
    
    st.session_state.clarification_mode = False
    context_override = "User clarification answers:\n" + "\n".join(st.session_state.clarification_answers)
   
    return get_rag_answer(
        "Final comprehensive diagnosis and solution based on all clarification answers provided.", 
        retriever, 
        GEMINI_KEY,
        context_override=context_override, 
        is_technical=True
    ), True

def handle_user_query(user_query, retriever, GEMINI_KEY):
    """Main logic to process a user query, deciding between casual, RAG, or clarification mode."""
    ticket_id = st.session_state.get("current_ticket_id")
    if not ticket_id:
        st.error("Ticket not initialized. Call create_ticket() at greet.")
        return "Ticket not initialized.", False

    # Retrieve ticket directly using UUID
    try:
        ticket_payload = retrieve_ticket_byid(ticket_id)
    except Exception as e:
        st.error(f"Ticket load failed: {e}")
        return "Error loading ticket.", False

    # First technical message → generate title & description
    if ticket_payload and not ticket_payload.get("title"):
        td = get_title_description(user_query)

        update_ticket_metadata(ticket_id, {
            "title": td["title"],
            "description": td["description"],
            "created_at": ticket_payload["created_at"]
        })

        add_ticket_event(ticket_id, "created", "system", "system",
                         "Ticket created from initial user message.")

    # Save user message
    user_msg_payload = build_conversation_payload(ticket_id, user_query, is_user=True)
    add_conversation_message(ticket_id, user_msg_payload)
    add_ticket_event(ticket_id, "message", "user", st.session_state.current_user, user_query)

    # Check if this is a technical issue
    is_technical = is_technical_issue(user_query)
    
    if not is_technical:
        return get_rag_answer(user_query, retriever, GEMINI_KEY, is_technical=False), False
    
    context_docs = retriever.get_relevant_documents(user_query) if retriever else []
    
    # Determine if we need to start clarification mode if there isn't sufficient context in KB
    if len(context_docs) == 200:
        st.session_sttate.clarification_mode = True
        st.session_sate.clarification_index = 0
        st.session_state.clarification_questions = generate_clarification_questions(user_query, GEMINI_KEY)
        st.session_state.clarification_answers = []
        return iter([st.session_state.clarification_questions[0]]), False
    
    st.session_state.clarification_mode = False
    # Return RAG answer and indicate buttons should be shown after stream finishes
    return get_rag_answer(user_query, retriever, GEMINI_KEY, is_technical=True), True

def get_title_description(issue_context, GEMINI_KEY):
    """Generates a ticket title and description from the conversation context."""
    template = """
You are an IT Support AI evaluator. Based on the full conversation and issue context, generate a **clear, concise ticket title** and a **human-readable description**.

Context:
{context}

Respond in STRICT JSON ONLY:
{{
    "title": "",
    "description": ""
}}
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        api_key=GEMINI_KEY
    )
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"context": issue_context}).strip()
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return {"title": "Untitled Issue", "description": issue_context}

    json_text = json_match.group(0)
    try:
        data = json.loads(json_text)
        return data
    except:
        return {"title": "Untitled Issue", "description": issue_context}

def evaluate_ai_capability(issue_context, GEMINI_KEY):
    """Evaluates if AI can solve the issue or if human intervention is needed"""
    template = """
You are an IT Support AI evaluator. Based on the conversation history and issue context, determine if this issue can be resolved by an AI agent or requires human technician intervention.
These are the automation that the Agent is allowed to carry out, command types must be limited to
    - restarting services
    - checking status
    - clearing caches
    - network resets
    - restarting NetworkManager
    - checking disk usage
    - checking CPU usage
    - checking memory usage
    - Identify heavy applications running

    Never go outside these categories.
If doing these categories will resolve the issue then, result should be AI_SOLVABLE

Context: {context}

- "AI_SOLVABLE" if the issue can be resolved through automated steps, scripts, or configuration changes
- "HUMAN_NEEDED" if the issue requires physical access, hardware replacement, complex judgment, or escalated permissions

Respond with ONLY in strict JSON:
{{
  "decision": "AI_SOLVABLE" | "HUMAN_NEEDED"
}}
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, api_key=GEMINI_KEY)
    
    chain = prompt | llm | StrOutputParser()
    raw_result = chain.invoke({"context": issue_context}).strip()
    json_match = re.search(r"\{.*\}", raw_result, re.DOTALL)

    if not json_match:
        st.sidebar.warning(f"AI Eval failed: No JSON found in raw output: {raw_result[:50]}...")
        return False
        
    json_text = json_match.group(0)

    try:
        data = json.loads(json_text)
        decision = data.get("decision", "").strip().upper()
        return decision == "AI_SOLVABLE"
    
    except json.JSONDecodeError as e:
        st.sidebar.error(f"JSON Decode Error in AI Eval: {e}. Raw text: {json_text[:50]}...")
        return False
    except Exception as e:
        st.sidebar.error(f"Unexpected error in AI Eval: {e}")
        return False

def classify_category(issue_context, GEMINI_KEY):
    """Classifies the issue into a predefined category."""
    template = """
You are an IT Support AI evaluator. Based on the conversation history and issue context, determine which category this issue falls under.

Context: {context}

Respond with ONLY ONE WORD, choosing from:
- "Software"
- "Hardware"
- "Network"
- "Performance"

Your response:
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, api_key=GEMINI_KEY)
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"context": issue_context}).strip()

    return result.split()[0].capitalize()


def build_conversation_payload(ticketId, message, isUser):
    """
    Builds the JSON payload for a single conversation message.
    """
    current_user_email = st.session_state.current_user
    user_data = st.session_state.users.get(current_user_email, {})

    if isUser:
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

def get_title_description_with_ticket_match(issue_context: str):
    """
    Like get_title_description, but also:
    - Checks for a similar past ticket via vector search.
    - Asks the LLM to return a JSON including matching ticket info:
        {
          "title": "",
          "description": "",
          "matching_ticket": {
            "ticket_id": "",
            "context": ""
          }
        }
    """
    # 1) Look for similar tickets
    similar = search_similar_tickets(issue_context, top_k=1, score_threshold=0.8)
    if similar:
        best = similar[0]
        similar_ticket_id = best.payload.get("ticket_id", "")
        similar_context = best.payload.get("text", "")
    else:
        similar_ticket_id = ""
        similar_context = ""

    template = """
You are an IT Support AI evaluator.

The user has reported the following issue:
{context}

Below is the most similar previous ticket we could find.
This may be empty if there was no sufficiently similar ticket.

SIMILAR_TICKET_ID: {similar_ticket_id}
SIMILAR_TICKET_CONTEXT:
{similar_ticket_context}

Your tasks:

1. Generate a clear, human‑readable title for the CURRENT issue only.
2. Generate a concise 2–4 sentence description for the CURRENT issue only.
3. If the similar ticket genuinely describes the same underlying problem,
   include its id and context in the "matching_ticket" object.
   If there is no useful match, set both fields in "matching_ticket"
   to empty strings.

Respond in STRICT JSON ONLY using exactly this schema:

{{
  "title": "",
  "description": "",
  "matching_ticket": {{
    "ticket_id": "",
    "context": ""
  }}
}}
"""
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        api_key=GEMINI_KEY
    )

    chain = prompt | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "context": issue_context,
            "similar_ticket_id": similar_ticket_id or "",
            "similar_ticket_context": similar_context or "",
        }).strip()
    except Exception:
        # fallback: use basic title/description, no matching_ticket
        base = get_title_description(issue_context)
        base["matching_ticket"] = {"ticket_id": "", "context": ""}
        return base

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        base = get_title_description(issue_context)
        base["matching_ticket"] = {"ticket_id": "", "context": ""}
        return base

    try:
        data = json.loads(json_match.group(0))
        return {
            "title": data.get("title", "Untitled Issue"),
            "description": data.get("description", issue_context),
            "matching_ticket": {
                "ticket_id": data.get("matching_ticket", {}).get("ticket_id", "") or similar_ticket_id or "",
                "context": data.get("matching_ticket", {}).get("context", "") or similar_context or "",
            },
        }
    except Exception:
        base = get_title_description(issue_context)
        base["matching_ticket"] = {"ticket_id": "", "context": ""}
        return base

def email_to_uuid(email: str) -> str:
    """Convert email to deterministic UUID"""
    import hashlib
    hash_obj = hashlib.md5(email.encode())
    return str(uuid.UUID(hash_obj.hexdigest()))