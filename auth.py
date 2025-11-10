import streamlit as st

def init_auth_state():
    if 'users' not in st.session_state:
        st.session_state.users = {'demo': '1234'}
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'auth_page' not in st.session_state:
        st.session_state.auth_page = 'login'


def login_page():
    st.title("🔐 Login to InjuryGuard AI")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login", key="login_btn"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.logged_in = True
            st.session_state.current_user = username
            st.session_state.auth_page = 'welcome'
            st.rerun()  # ✅ updated
        else:
            st.error("Invalid username or password")

    st.write("---")
    if st.button("Create new account"):
        st.session_state.auth_page = 'signup'
        st.rerun()  # ✅ updated


def signup_page():
    st.title("📝 Create an Account")
    username = st.text_input("Choose Username", key="signup_username")
    password = st.text_input("Choose Password", type="password", key="signup_password")
    confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")

    if st.button("Sign Up", key="signup_btn"):
        if not username or not password:
            st.warning("Please fill all fields")
        elif username in st.session_state.users:
            st.warning("Username already exists.")
        elif password != confirm:
            st.error("Passwords do not match")
        else:
            st.session_state.users[username] = password
            st.success("Account created! Please log in.")
            st.session_state.auth_page = 'login'
            st.rerun()  # ✅ updated

    st.write("---")
    if st.button("Back to login"):
        st.session_state.auth_page = 'login'
        st.rerun()  # ✅ updated


def welcome_page():
    st.title("🏆 Welcome to InjuryGuard AI")
    st.subheader(f"Hello, {st.session_state.current_user}!")
    st.markdown("Use AI-driven analysis to predict and prevent sports injuries.")

    if st.button("Start Screening"):
        st.session_state.auth_page = 'app'
        st.rerun()  # ✅ updated

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.auth_page = 'login'
        st.session_state.current_user = None
        st.rerun()  # ✅ updated
