import streamlit as st
import json
import extra_streamlit_components as stx
from pathlib import Path

from src.qdrant import (
    load_users_from_qdrant,
    save_user_to_qdrant
)

BASE_DIR = Path(__file__).parent.parent
USERS_FILE = BASE_DIR / "users.json"

# Cookie Manager initialized once
try:
    cookie_manager = stx.CookieManager()
except Exception as e:
    st.error(f"Error initializing CookieManager: {e}")
    cookie_manager = None

def load_users():
    """Loads user data from the JSON file into session state."""
    if USERS_FILE.exists():
        with open(USERS_FILE, "r") as f:
            st.session_state.users = json.load(f)
    else:
        st.session_state.users = {}

def save_users():
    """Saves user data from session state to the JSON file."""
    with open(USERS_FILE, "w") as f:
        json.dump(st.session_state.users, f)

def check_cookie_auth():
    """Checks for existing user cookie and authenticates if valid."""
    if cookie_manager:
        current_user_cookie = cookie_manager.get("current_user")
        if current_user_cookie and current_user_cookie in st.session_state.users:
            st.session_state.authenticated = True
            st.session_state.current_user = current_user_cookie
            return True
    return False

def render_auth_ui():
    """Renders the login/register interface."""
    st.title("🔐 IT Support Portal")

    if not st.session_state.show_register:
        # --- Sign In View ---
        st.subheader("Sign In")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Sign In"):
            users = st.session_state.users
            if email and email in users and users[email]["password"] == password:
                st.session_state.authenticated = True
                st.session_state.current_user = email
                st.session_state.greeted = False
                if cookie_manager:
                    cookie_manager.set("current_user", email, expires_at=None)
                st.rerun()
            else:
                st.error("Invalid email or password.")

        st.write("Don't have an account?")
        if st.button("Register"):
            st.session_state.show_register = True
            st.rerun()
        return

    # --- Register View ---
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