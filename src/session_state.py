import streamlit as st

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
    "ticket_created":False,
    "current_ticket_id": None,
    "ticket_created": False,
}

def initialize_session_state():
    """Initializes session state with default values if not already present."""
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v