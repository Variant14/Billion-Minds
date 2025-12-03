# Disable Streamlit file watcher immediately
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_FILE_WATCHER"] = "none"

import streamlit as st

# MUST BE FIRST - Set page config before any other Streamlit commands
st.set_page_config(page_title="IT Support Bot", page_icon="💻")

from dotenv import load_dotenv
from pathlib import Path

# --- LOCAL MODULE IMPORTS ---
from src.session_state import initialize_session_state
from src.auth import load_users, check_cookie_auth, render_auth_ui
from src.rag_pipeline import setup_rag_pipeline
from src.utils import (
    reset_chat,
    handle_user_query,
    get_next_clarification,
    evaluate_ai_capability,
    build_conversation_payload,
    create_ticket,
    assign_to_technician
)
from src.qdrant import (
    ensure_metadata_collection,
    load_users_from_qdrant,
    add_ticket_event, 
    update_ticket_status,
    add_conversation_message
)

# --- LangChain Core Imports ---
from langchain_core.messages import HumanMessage, AIMessage

# =============================
# INITIALIZATION
# =============================
load_dotenv()
BASE_DIR = Path(__file__).parent
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_KEY:
    st.error("GOOGLE_API_KEY missing in .env file. Add it before running.")
    st.stop()

# 1. Initialize State
initialize_session_state()

ensure_metadata_collection("users")
ensure_metadata_collection("tickets")
ensure_metadata_collection("ticket_conversations")

load_users_from_qdrant()

# 2. Load Users and Authenticate via Cookie
load_users()
check_cookie_auth()

# =============================
# AUTH SCREEN BLOCKER
# =============================
if not st.session_state.authenticated:
    render_auth_ui()
    st.stop()

# =============================
# BUILD KNOWLEDGE BASE (RAG)
# =============================
retriever = setup_rag_pipeline()
if not retriever:
    st.warning("RAG pipeline failed to initialize. Chatbot will operate on general knowledge only.")

# =============================
# CHATBOT UI
# =============================
st.title("IT Support Chat-Bot")
st.info("""
A ticket will be created for each technical conversation.
This conversation will be stored in our database.
""")

#START
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
    ticketId = st.session_state.current_ticket_id
    greeting = f"Hello {user_name}! 👋 I'm your IT Support Assistant. My current Ticket ID for this conversation is **{ticketId}**. How can I help you today?"
    add_conversation_message(ticket_id, build_conversation_payload(ticketId, greeting, False))
    st.session_state.chat_history.append(AIMessage(greeting))
    st.session_state.greeted = True
    #st.rerun() 

#Display Chats
for msg in st.session_state.chat_history:
    role = "Human" if isinstance(msg, HumanMessage) else "AI"
    with st.chat_message(role):
        st.markdown(msg.content)

#Clarify if the KB doesn't exisits
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

#Ask if the provided resolution steps worked
elif st.session_state.show_buttons:
    ticketId = st.session_state.current_ticket_id
    st.markdown("---")
    with st.chat_message("AI"):
        add_conversation_message(ticket_id, build_conversation_payload(ticketId,"Was your issue resolved?", False))
        st.markdown("**Was your issue resolved?**")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes — Resolved"):
            st.session_state.show_buttons = False
            ticket_id = st.session_state.get("current_ticket_id")
            if ticket_id:
                update_ticket_status(ticket_id, "resolved")
                add_ticket_event(ticket_id, "resolved", "agent", "agent_ai_01", "Ticket resolved by agent/AI.")
            user_msg = "Yes, issue resolved."
            add_conversation_message(ticket_id, build_conversation_payload(ticketId, user_msg, True))
            st.session_state.chat_history.append(HumanMessage(user_msg))
            st.session_state.awaiting_technician_confirmation = False
            ai_msg = "🎉 Glad your issue is resolved!"
            add_conversation_message(ticket_id, build_conversation_payload(ticketId, ai_msg, False))
            st.session_state.chat_history.append(AIMessage(ai_msg))

            #update_ticket({"status": "resolved", "resolved_at": datetime.utcnow().isoformat() + "Z"})
            st.session_state.show_reset_countdown = True
            st.rerun()
    with col2:
        if st.button("❌ No — Not resolved"):
            st.session_state.show_buttons = False

            user_msg = "No, not resolved."
            add_conversation_message(ticket_id, build_conversation_payload(ticketId, user_msg, True))
            st.session_state.chat_history.append(HumanMessage(user_msg))

            chat_context = "\n".join(msg.content for msg in st.session_state.chat_history)
            can_ai_solve = evaluate_ai_capability(chat_context,GEMINI_KEY)
            st.sidebar.write(can_ai_solve)
            if can_ai_solve:
                st.session_state.awaiting_resolution_confirmation = True
            else:
                st.session_state.awaiting_technician_confirmation = True
            
            st.rerun()

elif st.session_state.show_reset_countdown:
    # Issue is resolved, offer reset
    ticketId = st.session_state.current_ticket_id
    st.markdown("---")
    closing_message = f"✅ Issue Resolved! Ticket ID **{ticketId}** closed. Thank you for using IT Support."
    st.success(closing_message)
    add_conversation_message(ticket_id, build_conversation_payload(st.session_state.ticketId, closing_message, False))
    if st.button("🔄 Start New Conversation"):
        reset_chat()
        st.rerun()

elif st.session_state.awaiting_resolution_confirmation:
    # Prompt for AI fix attempt
    ticketId = st.session_state.current_ticket_id
    st.markdown("---")
    with st.chat_message("AI"):
        st.markdown("**AI can attempt an automatic fix. Do you want to proceed?**")
        add_conversation_message(ticket_id, build_conversation_payload(ticketId, "AI can attempt an automatic fix. Do you want to proceed?" , False))
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Proceed with AI fix"):
            user_msg = "Proceed with AI fix"
            add_conversation_message(ticket_id, build_conversation_payload(ticketId, user_msg, True))
            st.session_state.chat_history.append(HumanMessage(user_msg))
            
            ai_msg_auto = "Please wait while the chat-bot is running System Automation Checks"
            add_conversation_message(ticket_id, build_conversation_payload(ticketId, ai_msg_auto, False))
            st.session_state.show_button = False
            with st.chat_message("AI"):
                st.session_state.chat_history.append(AIMessage(ai_msg_auto)) 
            st.session_state.show_buttons = True
            st.session_state.awaiting_resolution_confirmation = False
            st.rerun()
    with col2:
        if st.button("❌ Don't proceed"):
            user_msg = "No, don't proceed with AI fix"
            add_conversation_message(ticket_id, build_conversation_payload(ticketId, user_msg, True))
            st.session_state.chat_history.append(HumanMessage(user_msg))

            ai_msg = "Okay, Would you like to escalate this issue to a human technician?"
            add_conversation_message(ticket_id, build_conversation_payload(ticketId, ai_msg, False))
            st.session_state.chat_history.append(AIMessage(ai_msg))
            
            st.session_state.awaiting_resolution_confirmation = False
            st.session_state.awaiting_technician_confirmation = True
            st.rerun()

elif st.session_state.awaiting_technician_confirmation:
    ticketId = st.session_state.current_ticket_id
    st.markdown("---")
    with st.chat_message("AI"):
        st.markdown("**Would you like to escalate this issue to a human technician?**")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes — Escalate to Technician"):
            user_msg = "Yes, escalate to technician"
            add_conversation_message(ticket_id,build_conversation_payload(ticketId, user_msg, True))
            st.session_state.chat_history.append(HumanMessage(user_msg))
            
            ticket_id = st.session_state.get("current_ticket_id")
            if ticket_id:
                update_ticket_status(ticket_id, "escalated")
                add_ticket_event(ticket_id, "assigned", "system", "system", "Assigned to human technician")
            ticket_record = {"ticket_id": ticket_id}

            ai_msg = f"✅ Your issue has been escalated to our technical team. A technician will contact you shortly. Updated Ticket ID: #{ticketId}"
            add_conversation_message(ticket_id,build_conversation_payload(ticketId, ai_msg, False))
            st.session_state.chat_history.append(AIMessage(ai_msg))

            st.session_state.technician_assign = True
            st.session_state.awaiting_technician_confirmation = False
            st.rerun()
    with col2:
        if st.button("❌ No — Start New Chat"):
            user_msg = "No, start a new chat"
            add_conversation_message(ticket_id,build_conversation_payload(ticketId, user_msg, True))
            st.session_state.chat_history.append(HumanMessage(user_msg))

            ai_msg = "Understood! Let's start fresh. How can I help you today?"
            add_conversation_message(ticket_id, build_conversation_payload(ticketId, ai_msg, False))
            st.session_state.chat_history.append(AIMessage(ai_msg))
            
            st.session_state.awaiting_technician_confirmation = False
            reset_chat()
            st.rerun()

elif st.session_state.technician_assign:
    ticketId = st.session_state.current_ticket_id
    st.markdown("---")
    ai_msg = f"🎫 Your ticket **#{ticketId}** has been created. A technician will reach out soon."
    st.success(ai_msg)
    if st.button("🔄 Start New Chat"):
        reset_chat()
        st.rerun()

else:
    user_query = st.chat_input("Ask something about IT…")
    if user_query and user_query.strip():
        st.session_state.awaiting_resolution_confirmation = False
        st.session_state.awaiting_technician_confirmation = False
        st.session_state.technician_assign = False

        userMessage =build_conversation_payload(st.session_state.ticketId, user_query, True)
        add_conversation_message(ticket_id, userMessage)
        st.session_state.chat_history.append(HumanMessage(user_query))
        
        with st.chat_message("Human"):
            st.markdown(user_query)
        with st.chat_message("AI"):
            ai_stream, should_show_buttons = handle_user_query(user_query, retriever,GEMINI_KEY)
            final = st.write_stream(ai_stream)
            add_conversation_message(ticket_id,build_conversation_payload(st.session_state.ticketId, final, False))           
            st.session_state.chat_history.append(AIMessage(final))
            ticket_id = st.session_state.get("current_ticket_id")
            if ticket_id:
                agent_msg_payload = build_conversation_payload(ticket_id, final, is_user=False)
                add_conversation_message(ticket_id, agent_msg_payload)
                add_ticket_event(ticket_id, "agent_reply", "agent", "agent_ai_01", final)
            st.session_state.show_buttons = should_show_buttons
        st.rerun()