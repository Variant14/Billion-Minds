from utils import ssh_run, is_allowed, ALLOWED_COMMANDS
from prompts import SAFE_COMMAND_GENERATION_PROMPT, FIX_COMPARE_PROMPT, ISSUE_DETECTION_AND_SAFE_COMMAND_GENERATION_PROMPT, LOG_COMMAND_GENERATION_PROMPT
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import streamlit as st

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4
    )

# LOG_COMMANDS = {
#     "network": [
#         "ping -c 3 8.8.8.8",
#         "dig google.com",
#         "ip a",
#     ],
#     "performance": [
#         "top -b -n 1 | head -n 5",
#         "free -m",
#         "df -h",
#     ],
#     "software": [
#         "systemctl --failed",
#         "journalctl -p 3 -n 20",
#     ],
#     "general": [
#         "uptime",
#         "dmesg | tail -n 50",
#         "journalctl -n 30"
#     ]
# }

def log_commands_generator_node(category):
    prompt = LOG_COMMAND_GENERATION_PROMPT.format(
        context=st.session_state.get("chat_history", []),
        issue_category=category,
        allowed_command_patterns = ALLOWED_COMMANDS
        )

    try:
        response = llm.invoke(prompt).content
        if response is None:
            return []
        output = json.loads(response)
        print("Log commands generated:")
        print(output)
        return output
    except Exception as e:
        print("Error occurred during log commands generation:", e)
        return []
    

def log_collector_node(category):
    """Collects logs from the target system based on the specified category."""
    selected_commands =  st.session_state.get("log_commands")

    if not selected_commands or len(selected_commands) == 0:
        selected_commands = log_commands_generator_node(category)
        st.session_state.log_commands = selected_commands

    logging.basicConfig(filename=f"./log_analysis.log", level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s")
    logs = {}
    print("selected commands:\n\n")
    print(selected_commands)
    
    for cmd in selected_commands:
        print(f"Executing command: {cmd.get('command')}")
        try:
            if not is_allowed(cmd.get('command')):
                print(f"Command {cmd.get('command')} is not allowed")
                continue
            st.markdown(f"{cmd.get("message")}")
            st.session_state.chat_history.append(AIMessage(cmd.get("message")))
            build_conversation_payload(st.session_state.ticketId, cmd.get("message"), False)
            logs[cmd.get('command')] = ssh_run(cmd.get('command'))
            logging.info(f"Executed command: {cmd.get('command')}")
        except Exception as e:
            print(f"Error executing command {cmd.get('command')}: {e}")
            logging.error(f"Error executing command {cmd.get('command')}: {e}")
            logs[cmd.get('command')] = f"ERROR executing command {cmd.get('command')}: {e}"
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
        context=context,
        allowed_command_patterns = ALLOWED_COMMANDS
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
from utils import is_allowed, ssh_run
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
import json
import streamlit as st
from utils import build_conversation_payload


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
        for issue in issues:
            suggested_commands = issue.get("suggested_commands", None)
            if suggested_commands is None:
                continue
            print(suggested_commands)
            for cmd in issue["suggested_commands"]:

                # Check whitelisted regex first
                if not is_allowed(cmd.get("command")):
                    continue
                safe_actions.append(cmd.get("command")) 


        # 2. Execute safe commands
        troubleshoot_ai_msg += f"\nExecuting troubleshooting actions...\n"
        st.info(f"Executing troubleshooting actions...")
        for cmd in safe_actions:
            try:
                result = ssh_run(cmd)
                logging.info(f"{ticketId}: Executed command: {cmd}")
                executed_results.append({
                    "command": cmd,
                    "output": result
                })
            except Exception as e:
                logging.error(f"{ticketId}: Error executing command {cmd}: {e}")

        # 3. Re-collect logs
        troubleshoot_ai_msg += f"\nRe-collecting logs...\n"
        st.markdown(f"Re-collecting logs...")
        
        after_logs_state = log_collector_node(st.session_state.ticket.get("category", "General"))
        after_logs = after_logs_state["logs"]

        # 4. Compare logs
        # You can re-run diagnostics node here or use a comparison LLM
        prompt = FIX_COMPARE_PROMPT.format(
            beforeLog=json.dumps(before_logs, indent=2),
            afterLog=json.dumps(after_logs, indent=2)
        )
        troubleshoot_ai_msg += f"\nComparing logs before and after troubleshooting...\n\n"
        st.markdown(f"Comparing logs before and after troubleshooting...")
        response = llm.invoke(prompt).content
        if response is None:
            troubleshoot_ai_msg += f"ERROR: Failed to compare logs before and after troubleshooting\n\n"
            st.error("ERROR: Failed to compare logs before and after troubleshooting")
            st.session_state.chat_history.append(AIMessage(troubleshoot_ai_msg))
            build_conversation_payload(ticketId, troubleshoot_ai_msg, False)
            return state
        output = json.loads(response)
        # return structured diagnostics
        troubleshoot_ai_msg += f"\nComparison completed\n\n"
        st.session_state.chat_history.append(AIMessage(troubleshoot_ai_msg))
        build_conversation_payload(ticketId, troubleshoot_ai_msg, False)
        return {"summary": output}
    except Exception as e:
        print("Error occurred:", e)
        troubleshoot_ai_msg += f"ERROR: {e}"
        st.error(f"ERROR: {e}")
        st.session_state.chat_history.append(AIMessage(troubleshoot_ai_msg))
        build_conversation_payload(ticketId, troubleshoot_ai_msg, False)
        return state