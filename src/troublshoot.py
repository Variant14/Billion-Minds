from utils import ssh_run 
from prompts import SAFE_COMMAND_GENERATION_PROMPT, FIX_COMPARE_PROMPT, ISSUE_DETECTION_AND_SAFE_COMMAND_GENERATION_PROMPT

LOG_COMMANDS = {
    "network": [
        "ping -c 3 8.8.8.8",
        "dig google.com",
        "ip a",
    ],
    "performance": [
        "top -b -n 1 | head -n 5",
        "free -m",
        "df -h",
    ],
    "software": [
        "systemctl --failed",
        "journalctl -p 3 -n 20",
    ],
    "general": [
        "uptime",
        "dmesg | tail -n 50",
        "journalctl -n 30"
    ]
}


def log_collector_node(category):
    """Collects logs from the target system based on the specified category."""
    selected_commands = LOG_COMMANDS.get(category, LOG_COMMANDS["general"])
    print(selected_commands)

    logs = {}
    
    for cmd in selected_commands:
        try:
            logs[cmd] = ssh_run(cmd)
        except Exception as e:
            print(f"Error executing command {cmd}: {e}")
            logs[cmd] = f"ERROR executing command {cmd}: {e}"
    return {"logs": logs}


from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import json

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4
    )
    

# ------------- Diagnostics Node -------------
def diagnostics_node(logs, context):

    prompt = SAFE_COMMAND_GENERATION_PROMPT.format(
        logs=json.dumps(logs, indent=2),
        context=context
    )

    try:
        response = llm.invoke(prompt).content
        if response is None:
            return {"detected_issues": []}
        output = json.loads(response)
        # return structured diagnostics
        return {"detected_issues": output["issues"]}
    except Exception as e:
        print("Error occurred:", e)
        return {"detected_issues": []}

# ------------- Scanning Node -------------
def scanning_node(logs, context):

    prompt = ISSUE_DETECTION_AND_SAFE_COMMAND_GENERATION_PROMPT.format(
        logs=json.dumps(logs, indent=2),
        context=context
    )

    try:
        response = llm.invoke(prompt).content
        if response is None:
            return {"detected_issues": []}
        output = json.loads(response)
        # return structured diagnostics
        return {"detected_issues": output["issues"]}
    except Exception as e:
        print("Error occurred:", e)
        return {"detected_issues": []}

# ------------- Troubleshooting Node -------------
from utils import is_blacklisted, ssh_run
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
import json
import streamlit as st
from utils import build_conversation_payload
import logging

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4
    )

def troubleshoot_node(state):
    issues = state.get("detected_issues", [])
    before_logs = state["logs"]
    ticketId = st.session_state.get("ticketId", None)
    safe_actions = []
    executed_results = []
    
    troubleshoot_ai_msg = "\n**Troubleshooting starts now...**\n"
    markdown_msg = "**Troubleshooting starts now...**"
    st.markdown(markdown_msg)

    if ticketId is None:
        error_msg = "Something went wrong, unable to continue without a Ticket Id"
        markdown_msg = "⚠️ " + error_msg
        st.error(error_msg)
        st.session_state.chat_history.append(AIMessage(markdown_msg))
        return {}
    
    try:
        # 1. Extract suggested commands
        logging.basicConfig(filename=f"./troubleshooting.log", level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s")

        for issue in issues:
            suggested_commands = issue.get("suggested_commands", None)
            if suggested_commands is None:
                continue
            for cmd in issue["suggested_commands"]:

                # Check blacklist regex first
                if is_blacklisted(cmd):
                    continue

                safe_actions.append(cmd) 


        # 2. Execute safe commands
        troubleshoot_ai_msg += f"\nExecuting troubleshooting actions...\n"
        st.info(f"Executing troubleshooting actions...")
        for cmd in safe_actions:
            result = ssh_run(cmd)
            logging.info(f"{ticketId}: Executed command: {cmd}")
            executed_results.append({
                "command": cmd,
                "output": result
            })

        # 3. Re-collect logs
        troubleshoot_ai_msg += f"\nRe-collecting logs...\n"
        st.markdown(f"Re-collecting logs...")
        after_logs_state = log_collector_node("general")
        after_logs = after_logs_state["logs"]

        # 4. Compare logs
        # You can re-run diagnostics node here or use a comparison LLM
        prompt = FIX_COMPARE_PROMPT.format(
            beforeLog=json.dumps(before_logs, indent=2),
            afterLog=json.dumps(after_logs, indent=2)
        )
        troubleshoot_ai_msg += f"\nComparing logs before and after troubleshooting...\n"
        st.markdown(f"Comparing logs before and after troubleshooting...")
        response = llm.invoke(prompt).content
        if response is None:
            troubleshoot_ai_msg += f"ERROR: Failed to compare logs before and after troubleshooting"
            st.error("ERROR: Failed to compare logs before and after troubleshooting")
            st.session_state.chat_history.append(AIMessage(troubleshoot_ai_msg))
            build_conversation_payload(ticketId, troubleshoot_ai_msg, False)
            return state
        output = json.loads(response)
        # return structured diagnostics
        return {"summary": output}
    except Exception as e:
        print("Error occurred:", e)
        troubleshoot_ai_msg += f"ERROR: {e}"
        st.error(f"ERROR: {e}")
        st.session_state.chat_history.append(AIMessage(troubleshoot_ai_msg))
        build_conversation_payload(ticketId, troubleshoot_ai_msg, False)
        return state