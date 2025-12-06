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
from session_state import initialize_session_state
from auth import load_users, check_cookie_auth, render_auth_ui
from rag_pipeline import setup_rag_pipeline
from utils import (
    reset_chat,
    handle_user_query,
    get_next_clarification,
    evaluate_ai_capability,
    build_conversation_payload,
    create_ticket,
    assign_to_technician
)

# --- LangChain Core Imports ---
from langchain_core.messages import HumanMessage, AIMessage

# --- Troubleshooting Imports ---
from troublshoot import troubleshoot_node, diagnostics_node, log_collector_node

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
    user_name = st.session_state.users[current]["name"] if current else "there"
    
    # Create Ticket ID 
    st.session_state.ticketId = create_ticket() 
    
    greeting = f"Hello {user_name}! 👋 I'm your IT Support Assistant. My current Ticket ID for this conversation is **{st.session_state.ticketId}**. How can I help you today?"
    build_conversation_payload(st.session_state.ticketId, greeting, False)
    st.session_state.chat_history.append(AIMessage(greeting))
    st.session_state.greeted = True
    st.rerun() 

#Display Chats
for msg in st.session_state.chat_history:
    role = "Human" if isinstance(msg, HumanMessage) else "AI"
    with st.chat_message(role):
        st.markdown(msg.content)

#Clarify if the KB doesn't exisits
if st.session_state.clarification_mode:
    current_q = st.session_state.clarification_questions[st.session_state.clarification_index]
    
    ans = st.chat_input(f"Answer for: {current_q[:30]}...")
    if ans:
        build_conversation_payload(st.session_state.ticketId, ans, True)
        st.session_state.chat_history.append(HumanMessage(ans))
        st.session_state.clarification_answers.append(ans)
        
        with st.chat_message("AI"):
            ai_stream, is_final = get_next_clarification(retriever,GEMINI_KEY)
            final = st.write_stream(ai_stream)
            build_conversation_payload(st.session_state.ticketId, final, False)
            st.session_state.chat_history.append(AIMessage(final))
            
            if is_final:
                st.session_state.show_buttons = True
        st.rerun()

#Ask if the provided resolution steps worked
elif st.session_state.show_buttons:
    ticketId = st.session_state.ticketId
    st.markdown("---")
    with st.chat_message("AI"):
        build_conversation_payload(ticketId,"Was your issue resolved?", False)
        st.markdown("**Was your issue resolved?**")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes — Resolved"):
            st.session_state.show_buttons = False

            user_msg = "Yes, issue resolved."
            build_conversation_payload(ticketId, user_msg, True)
            st.session_state.chat_history.append(HumanMessage(user_msg))

            ai_msg = "🎉 Glad your issue is resolved!"
            build_conversation_payload(ticketId, ai_msg, False)
            st.session_state.chat_history.append(AIMessage(ai_msg))

            #update_ticket({"status": "resolved", "resolved_at": datetime.utcnow().isoformat() + "Z"})
            st.session_state.show_reset_countdown = True
            st.rerun()
    with col2:
        if st.button("❌ No — Not resolved"):
            st.session_state.show_buttons = False

            user_msg = "No, not resolved."
            build_conversation_payload(ticketId, user_msg, True)
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
    ticketId = st.session_state.ticketId
    st.markdown("---")
    st.success(f"✅ Issue Resolved! Ticket ID **{ticketId}** closed. Thank you for using IT Support.")
    if st.button("🔄 Start New Conversation"):
        reset_chat()
        st.rerun()

elif st.session_state.awaiting_resolution_confirmation:
    # Prompt for AI fix attempt
    ticketId = st.session_state.ticketId
    st.markdown("---")
    with st.chat_message("AI"):
        st.markdown("**AI can attempt an automatic fix. Do you want to proceed?**")
        build_conversation_payload(ticketId, "AI can attempt an automatic fix. Do you want to proceed?" , False)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Proceed with AI fix"):
            try:
                user_msg = "Proceed with AI fix"
                build_conversation_payload(ticketId, user_msg, True)
                st.session_state.chat_history.append(HumanMessage(user_msg))
                
                ai_msg_auto = "Please wait while the chat-bot is running System Automation Checks"
                build_conversation_payload(ticketId, ai_msg_auto, False)
                st.session_state.chat_history.append(AIMessage(ai_msg_auto))
                st.session_state.show_buttons = False
                
                with st.spinner("🔄 Processing your request..."):
                    with st.chat_message("AI"):
                        st.markdown("**AI analyzing logs**")
                    
                    # Collect logs
                    logs = log_collector_node("general")["logs"]
                    st.markdown("**Logs collected. Running diagnostics...**")
                    
                    # Run diagnostics
                    diagnostics_node_result = diagnostics_node(logs, st.session_state.chat_history)
                    
                    # Build AI message with issues and commands
                    if diagnostics_node_result and "detected_issues" in diagnostics_node_result:
                        issues = diagnostics_node_result["detected_issues"]
                        
                        if issues:
                            # Display issues in UI
                            st.markdown("**Issues detected:**")
                            
                            # Build formatted message for chat history
                            issues_message = "**Diagnostic Results:**\n\n**Issues Detected:**\n"
                            
                            for idx, issue in enumerate(issues, 1):
                                issue_text = issue.get('issue', 'Unknown issue')
                                st.markdown(f" {idx+1}. {issue_text}")
                                issues_message += f"{idx}. {issue_text}\n"
                                human_intervention = issue.get('human_intervention_needed', False)
                                if human_intervention:
                                    issues_message += "   - _Human intervention needed._\n"
                                    st.warning(f"**Human Intervention Needed:** {issue_text}")
                            # Add suggested commands if available
                            # if any('command' in issue or 'suggested_commands' in issue for issue in issues):
                            #     issues_message += "\n**Suggested Commands:**\n"
                            #     # st.markdown("\n**Suggested Commands:**")
                                
                            #     for idx, issue in enumerate(issues, 1):
                            #         command = issue.get('command') or issue.get('suggested_commands')
                            #         if command:
                            #             # st.code(command, language="bash")
                            #             issues_message += f"{idx}. `{command}`\n"
                            if any('human_intervention_needed' in issue for issue in issues):
                                issues_message += "\n_⚠️ Some issues require human intervention. Please consider escalating to a technician._\n"
                                st.warning("⚠️ Some issues require human intervention. Please consider escalating to a technician.")
                            else:
                                issues_message += "\n✅ All detected issues have suggested safe commands for resolution.\n"
                                st.markdown("Troubleshooting starts now...")
                                # Execute troubleshooting node
                            
                            # Add to chat history
                            build_conversation_payload(ticketId, issues_message, False)
                            st.session_state.chat_history.append(AIMessage(issues_message))
                        else:
                            no_issues_msg = "✅ No issues detected. System appears to be functioning normally."
                            st.info(no_issues_msg)
                            build_conversation_payload(ticketId, no_issues_msg, False)
                            st.session_state.chat_history.append(AIMessage(no_issues_msg))
                    else:
                        warning_msg = "⚠️ Diagnostics completed but no results were returned."
                        st.warning(warning_msg)
                        build_conversation_payload(ticketId, warning_msg, False)
                        st.session_state.chat_history.append(AIMessage(warning_msg))
                
                st.session_state.show_buttons = True
                st.session_state.awaiting_resolution_confirmation = False
                
            except Exception as e:
                error_msg = f"⚠️ An error occurred during diagnostics: {str(e)}"
                st.error(error_msg)
                build_conversation_payload(ticketId, error_msg, False)
                st.session_state.chat_history.append(AIMessage(error_msg))
                st.session_state.show_buttons = True
                st.session_state.awaiting_resolution_confirmation = False
            finally:
                st.session_state.processing = False
                st.rerun()
        with col2:
            if st.button("❌ Don't proceed"):
                user_msg = "No, don't proceed with AI fix"
                build_conversation_payload(ticketId, user_msg, True)
                st.session_state.chat_history.append(HumanMessage(user_msg))

                ai_msg = "Okay, Would you like to escalate this issue to a human technician?"
                build_conversation_payload(ticketId, ai_msg, False)
                st.session_state.chat_history.append(AIMessage(ai_msg))
                
                st.session_state.awaiting_resolution_confirmation = False
                st.session_state.awaiting_technician_confirmation = True
                st.rerun()

elif st.session_state.awaiting_technician_confirmation:
    ticketId = st.session_state.ticketId
    st.markdown("---")
    with st.chat_message("AI"):
        st.markdown("**Would you like to escalate this issue to a human technician?**")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes — Escalate to Technician"):
            user_msg = "Yes, escalate to technician"
            build_conversation_payload(ticketId, user_msg, True)
            st.session_state.chat_history.append(HumanMessage(user_msg))
            
            assign_to_technician(ticketId)

            ai_msg = f"✅ Your issue has been escalated to our technical team. A technician will contact you shortly. Updated Ticket ID: #{ticketId}"
            build_conversation_payload(ticketId, ai_msg, False)
            st.session_state.chat_history.append(AIMessage(ai_msg))

            st.session_state.technician_assign = True
            st.session_state.awaiting_technician_confirmation = False
            st.rerun()
    with col2:
        if st.button("❌ No — Start New Chat"):
            user_msg = "No, start a new chat"
            build_conversation_payload(ticketId, user_msg, True)
            st.session_state.chat_history.append(HumanMessage(user_msg))

            ai_msg = "Understood! Let's start fresh. How can I help you today?"
            build_conversation_payload(ticketId, ai_msg, False)
            st.session_state.chat_history.append(AIMessage(ai_msg))
            
            st.session_state.awaiting_technician_confirmation = False
            reset_chat()
            st.rerun()

elif st.session_state.technician_assign:
    ticketId = st.session_state.ticketId
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

        build_conversation_payload(st.session_state.ticketId, user_query, True)
        st.session_state.chat_history.append(HumanMessage(user_query))
        
        with st.chat_message("Human"):
            st.markdown(user_query)
        with st.chat_message("AI"):
            ai_stream, should_show_buttons = handle_user_query(user_query, retriever,GEMINI_KEY)
            
            final = st.write_stream(ai_stream)
            build_conversation_payload(st.session_state.ticketId, final, False)            
            st.session_state.chat_history.append(AIMessage(final))
            st.session_state.show_buttons = should_show_buttons
        st.rerun()