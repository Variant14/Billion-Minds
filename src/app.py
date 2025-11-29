# Disable Streamlit file watcher immediately
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_FILE_WATCHER"] = "none"

import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import re

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
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "clarification_mode" not in st.session_state:
    st.session_state.clarification_mode = False
if "pending_questions" not in st.session_state:
    st.session_state.pending_questions = []
if "clarification_answers" not in st.session_state:
    st.session_state.clarification_answers = []
if "show_buttons" not in st.session_state:
    st.session_state.show_buttons = False
# GREETING ON START
if not st.session_state.chat_history:
    greeting = "Hello! 👋 I’m your IT Support Assistant. How can I help you today?"
    st.session_state.chat_history.append(AIMessage(greeting))

# =============================
# LOAD ENV + BASE DIR
# =============================
load_dotenv()
BASE_DIR = Path(__file__).parent.parent
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_KEY:
    st.error("GOOGLE_API_KEY missing in .env file. Add it before running.")
    st.stop()

st.set_page_config(page_title="IT Support Bot", page_icon="💻")
st.title("💻 IT Support Chat-Bot (Gemini + RAG)")
st.text("Bot will ask clarification questions if KB context is insufficient.")

# =============================
# BUILD KNOWLEDGE BASE
# =============================
@st.cache_resource
def setup_rag_pipeline():
    DATA_PATH = BASE_DIR / "it_docs"
    CHROMA_PATH = BASE_DIR / "chroma_db"
    
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        st.warning(f"No documents found in: {DATA_PATH}")
        return None, {}
    
    loader = DirectoryLoader(str(DATA_PATH), glob="**/*.txt")
    raw_docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(raw_docs)
    
    faq_dict = {d.page_content.strip().lower(): d.page_content for d in docs}
    
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
    return retriever, faq_dict

retriever, faq_dict = setup_rag_pipeline()

# =============================
# HELPER FUNCTIONS
# =============================
def combine_documents(docs):
    return "\n\n".join(d.page_content for d in docs)

def generate_clarification_questions(user_query):
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
        questions = [q.strip() for q in questions_str.split('\n') if q.strip()]
    
    return questions

def get_rag_answer(user_query, context_override=None):
    context_docs = retriever.get_relevant_documents(user_query) if retriever else []
    context_text = combine_documents(context_docs) if context_docs else ""
    
    if context_override:
        context_text += "\n\n" + context_override
    
    template = """
You are an IT Support Technician. Answer using CONTEXT first.
If context is insufficient, say so politely and ask for followup questions to better understand the situation.
Be professional and helpful.

CONTEXT:
---------
{context}
---------

Chat history: {chat_history}
User question: {user_question}
"""
    rag_prompt = ChatPromptTemplate.from_template(template)
    chat_hist_copy = st.session_state.chat_history
    
    rag_chain = (
        {
            "context": lambda _: context_text,
            "user_question": RunnablePassthrough(),
            "chat_history": lambda _: chat_hist_copy,
        }
        | rag_prompt
        | ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, api_key=GEMINI_KEY)
        | StrOutputParser()
    )
    
    return rag_chain.stream({"user_question": user_query})

def handle_user_query(user_query):
    context_docs = retriever.get_relevant_documents(user_query) if retriever else []
    context_text = combine_documents(context_docs) if context_docs else ""
    
    if len(context_text) < 150:
        st.session_state.clarification_mode = True
        questions = generate_clarification_questions(user_query)
        st.session_state.pending_questions = questions[1:]
        return iter([questions[0]])
    else:
        st.session_state.clarification_mode = False
        st.session_state.pending_questions = []
        st.session_state.clarification_answers = []
        return get_rag_answer(user_query)

def get_next_clarification():
    if st.session_state.pending_questions:
        next_q = st.session_state.pending_questions.pop(0)
        return iter([next_q])
    else:
        st.session_state.clarification_mode = False
        context_override = "User clarification answers:\n" + "\n".join(st.session_state.clarification_answers)
        # Show buttons after final solution
        st.session_state.show_buttons = True
        final_answer_stream = get_rag_answer("Final comprehensive diagnosis and solution", context_override=context_override)
        return final_answer_stream

# =============================
# STREAMLIT UI
# =============================
# Display chat history
for msg in st.session_state.chat_history:
    role = "Human" if isinstance(msg, HumanMessage) else "AI"
    with st.chat_message(role):
        st.markdown(msg.content)

# Clarification phase
if st.session_state.clarification_mode:
    user_answer = st.chat_input("Please answer the clarification question:")
    if user_answer:
        st.session_state.chat_history.append(HumanMessage(user_answer))
        st.session_state.clarification_answers.append(user_answer)

        with st.chat_message("Human"):
            st.markdown(user_answer)

        with st.chat_message("AI"):
            ai_reply_stream = get_next_clarification()
            full_response = st.write_stream(ai_reply_stream)
            st.session_state.chat_history.append(AIMessage(full_response))

        st.rerun()

# Normal chat flow
else:
    user_query = st.chat_input("Ask something about IT…")
    if user_query and user_query.strip():
        st.session_state.chat_history.append(HumanMessage(user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            ai_reply_stream = handle_user_query(user_query)
            full_response = st.write_stream(ai_reply_stream)
            st.session_state.chat_history.append(AIMessage(full_response))

            # Only show buttons if final answer looks like a resolution
            if not st.session_state.clarification_mode and len(full_response) > 200:
                st.session_state.show_buttons = True

        st.rerun()

# Resolution buttons
if st.session_state.show_buttons and not st.session_state.clarification_mode:
    st.markdown("---")
    st.markdown("**Was your issue resolved?**")
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("✅ Yes", key="yes_btn"):
            st.session_state.chat_history.append(HumanMessage("Yes"))
            st.session_state.chat_history.append(AIMessage("🎉 Glad your issue is resolved!"))
            st.session_state.show_buttons = False
            st.session_state.clarification_answers = []
            st.rerun()

    with col2:
        if st.button("❌ No", key="no_btn"):
            st.session_state.chat_history.append(HumanMessage("No"))
            st.session_state.chat_history.append(AIMessage("⚠️ Sorry to hear that. Please provide more details."))
            st.session_state.show_buttons = False
            st.session_state.clarification_answers = []
            st.rerun()
