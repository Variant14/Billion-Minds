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
    

import logging
logger = logging.getLogger("log_calls")

def log_collector_node(category):
    """Collects logs from the target system based on the specified category."""
    selected_commands =  st.session_state.get("log_commands")
    ticketId = st.session_state.get("current_ticket_id")

    if not selected_commands or len(selected_commands) == 0:
        selected_commands = log_commands_generator_node(category)
        st.session_state.log_commands = selected_commands

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
            build_conversation_payload(st.session_state.current_ticket_id, cmd.get("message"), False)
            logs[cmd.get('command')] = ssh_run(cmd.get('command'))
            logger.info(f"{ticketId} Executed command: {cmd.get('command')}")
        except Exception as e:
            print(f"Error executing command {cmd.get('command')}: {e}")
            logger.error(f"{ticketId} Error executing command {cmd.get('command')}: {e}")
            logs[cmd.get('command')] = f"ERROR executing command {cmd.get('command')}: {e}"
    return {"logs": logs}
    

import logging
import asyncio
from utils import send_command_and_wait

logger = logging.getLogger("log_calls")


async def log_collector_node(category, agent_id):
    """Collect logs via WebSocket agent instead of SSH"""

    selected_commands = st.session_state.get("log_commands")
    ticketId = st.session_state.get("current_ticket_id")

    if not selected_commands:
        selected_commands = log_commands_generator_node(category)
        st.session_state.log_commands = selected_commands

    logs = {}

    print("selected commands:\n", selected_commands)

    for cmd in selected_commands:
        command = cmd.get("command")
        message = cmd.get("message")

        print(f"Requesting command: {command}")

        try:
            # ✅ Keep local allow-list (defense in depth)
            if not is_allowed(command):
                print(f"Command not allowed: {command}")
                continue

            # ✅ UI + conversation unchanged
            st.markdown(message)
            st.session_state.chat_history.append(AIMessage(message))
            build_conversation_payload(ticketId, message, False)

            # ✅ WebSocket execution (REPLACEMENT)
            result = asyncio.run(send_command_and_wait(agent_id, command))

            logs[command] = result
            logger.info(f"{ticketId}: Executed command: {command}")

        except Exception as e:
            logger.error(f"{ticketId}: Error executing {command}: {e}")
            logs[command] = f"ERROR executing command {command}: {e}"

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
import logging
logger = logging.getLogger("troubleshoot_actions")

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4
    )

def troubleshoot_node(state):
    issues = state.get("detected_issues", [])
    before_logs =state.get("logs", "")
    ticketId = state.get("ticketId", None)
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
                logger.info(f"{ticketId}: Executed command: {cmd}")
                executed_results.append({
                    "command": cmd,
                    "output": result
                })
            except Exception as e:
                logger.error(f"{ticketId}: Error executing command {cmd}: {e}")

        # 3. Re-collect logs
        troubleshoot_ai_msg += f"\nRe-collecting logs...\n"
        st.markdown(f"Re-collecting logs...")
        
        after_logs_state = log_collector_node(state.get("category", "General"))
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
  
async def notify_user(message, state="info"):
    await st.session_state.ui_message_queue.put({
        "message": message,
        "state": state
    })


async def start_auto_fix(ticketId, agentId, context, category):
    print("Starting auto fix...")
    
    logs = log_collector_node(category)["logs"]
    
    diagnostics_node_result = diagnostics_node(logs, context)
    
    # Build AI message with issues and commands
    if diagnostics_node_result and "detected_issues" in diagnostics_node_result:
        issues = diagnostics_node_result["detected_issues"]
        
        if issues:
            # Display issues in UI
            await notify_user("**Diagnostics completed. Issues detected:**", "info")
            
            # Build formatted message for chat history
            # issues_message = "**Diagnostic Results:**\n\n**Issues Detected:**\n"
            for idx, issue in enumerate(issues, 1):
                issue_text = issue.get('issue', 'Unknown issue')
                human_intervention = issue.get('human_intervention_needed', False)
                if human_intervention:
                    await notify_user(f" {idx}. {issue_text} - Human intervention needed.", "info")
                else:
                    await notify_user(f" {idx}. {issue_text}", "info")                                          
                
            if all(issue.get("suggested_commands") is None or len(issue.get("suggested_commands")) == 0 for issue in issues):
                await notify_user("Sorry, we could not find any solution for this issue at the moment. Please consider escalating to a technician.", "human_intervention_needed")
                return
                
            elif any(issue.get("human_intervention_needed", False) and idx < len(issues) - idx for idx, issue in enumerate(issues)):
                await notify_user("⚠️ Some critical issues require human intervention. Please consider escalating to a technician.", "human_intervention_needed")
                return
            else:
                # Execute troubleshooting node
                troubleshoot_result = troubleshoot_node({
                    "logs": logs,
                    "detected_issues": issues,
                    "category": category,
                    "ticketId": ticketId
                })
                if troubleshoot_result and "summary" in troubleshoot_result:
                    await notify_user("**Troubleshooting Summary:**", "info")
                    build_conversation_payload(ticketId, ai_msg_auto, False)
                    st.session_state.chat_history.append(AIMessage(ai_msg_auto))
                    st.session_state.chat_history.append(AIMessage(troubleshoot_result["summary"]))
                    build_conversation_payload(ticketId, troubleshoot_result["summary"], False)
                    st.session_state.awaiting_resolution_confirmation = False
                    st.session_state.show_buttons = True
                else:
                    ai_msg_auto += "\nSorry, we could not find any solution for this issue at the moment. Please consider escalating to a technician.\n"
                    st.warning("Sorry, we could not find any solution for this issue at the moment. Please consider escalating to a technician.")
                    build_conversation_payload(ticketId, ai_msg_auto, False)
                    st.session_state.chat_history.append(AIMessage(ai_msg_auto))
                    # Call for human intervention
                    st.session_state.awaiting_resolution_confirmation = False
                    st.session_state.awaiting_technician_confirmation = True
        else:
            no_issues_msg = "✅ No issues detected. System appears to be functioning normally."
            st.info(no_issues_msg)
            build_conversation_payload(ticketId, no_issues_msg, False)
            st.session_state.chat_history.append(AIMessage(no_issues_msg))
            st.session_state.awaiting_resolution_confirmation = False
            st.session_state.awaiting_technician_confirmation = True
    else:
        warning_msg = "⚠️ Diagnostics completed but no results were returned."
        st.warning(warning_msg)
        build_conversation_payload(ticketId, warning_msg, False)
        st.session_state.chat_history.append(AIMessage(warning_msg))
        st.session_state.awaiting_resolution_confirmation = False
        st.session_state.awaiting_technician_confirmation = True