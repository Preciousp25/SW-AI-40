import streamlit as st

# -----------------------------
# Initialize Session Variables
# -----------------------------
def init_auth_state():
    if 'users' not in st.session_state:
        st.session_state.users = {'demo': '1234'}  # default demo user
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'auth_page' not in st.session_state:
        st.session_state.auth_page = 'login'  # can be: login, signup, welcome, or app


# -----------------------------
# LOGIN PAGE
# -----------------------------
def login_page():
    st.title(" Login to InjuryGuard AI")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.logged_in = True
            st.session_state.current_user = username
            st.session_state.auth_page = 'welcome'
            st.success(f"Welcome back, {username}!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

    st.write("Don't have an account?")
    if st.button("Create New Account"):
        st.session_state.auth_page = 'signup'
        st.experimental_rerun()


# -----------------------------
# SIGNUP PAGE
# -----------------------------
def signup_page():
    st.title("Create an Account")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if username == "" or password == "":
            st.warning("Please fill all fields.")
        elif username in st.session_state.users:
            st.warning("Username already exists. Please log in instead.")
        elif password != confirm:
            st.error("Passwords do not match.")
        else:
            st.session_state.users[username] = password
            st.session_state.auth_page = 'login'
            st.success("Account created successfully! Please log in.")
            st.experimental_rerun()

    st.write("Already have an account?")
    if st.button("Go to Login"):
        st.session_state.auth_page = 'login'
        st.experimental_rerun()


# -----------------------------
# WELCOME PAGE
# -----------------------------
def welcome_page():
    st.title("👋 Welcome to InjuryGuard AI")
    st.subheader(f"Hello, {st.session_state.current_user}!")
    st.markdown("""
        This platform uses AI-driven biosensor analytics to predict and prevent sports injuries.  
        Click below to begin real-time risk assessment.
    """)

    if st.button(" Start Screening"):
        st.session_state.auth_page = 'app'
        st.experimental_rerun()

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.auth_page = 'login'
        st.session_state.current_user = None
        st.experimental_rerun()
