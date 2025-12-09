from utils import ssh_run, is_allowed, ALLOWED_COMMANDS, send_command_request
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

def log_commands_generator_node(category, context):
    prompt = LOG_COMMAND_GENERATION_PROMPT.format(
        context=context,
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
        return None
    

import logging
import uuid
logger = logging.getLogger("log_calls")

async def log_collector_node(ticketId, agentId, selected_commands):
    """Collects logs from the target system based on the specified category."""
    print(f"Collecting logs for ticket {ticketId}...")

    logs = {}
    print("selected commands:\n\n")
    print(selected_commands)
    
    for cmd in selected_commands:
        print(f"Executing command: {cmd.get('command')}")
        try:
            if not is_allowed(cmd.get('command')):
                print(f"Command {cmd.get('command')} is not allowed")
                continue
            await notify_user(f"{cmd.get("message")}")
            try:
                request_id = uuid.uuid4() 
                result = await send_command_request({
                    "request_id": request_id,
                    "command": cmd.get('command'),
                    "agent_Id": agentId
                })
                while True:
                    result = st.session_state.ui_message_queue.get_nowait()
                    if result.get("request_id") == request_id:
                        if result is None:
                            raise Exception("Unable to collect logs for this command")
                        data = json.loads(result)
                        logs[cmd.get('command')] = data.get("data")
                        print("logs collected for command", cmd.get('command'))
                        print(data)
                        logger.info(f"{ticketId} Command request satisfied: {cmd.get('command')}")
                        break
            except Exception as e:
                print(f"Error executing command {cmd.get('command')}: {e}")
                logger.error(f"{ticketId} Error executing command {cmd.get('command')}: {e}")
        except Exception as e:
            print(f"Error executing command {cmd.get('command')}: {e}")
            logger.error(f"{ticketId} Error executing command {cmd.get('command')}: {e}")
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
import logging
logger = logging.getLogger("troubleshoot_actions")

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4
    )

async def troubleshoot_node(state):
    issues = state.get("detected_issues", [])
    before_logs =state.get("logs", "")
    ticketId = state.get("ticketId", None)
    agentId = state.get("agentId", None)
    log_commands = state.get("selected_commands", [])
    
    safe_actions = []
    executed_results = []
    
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
        for cmd in safe_actions:
            try:
                request_id = uuid.uuid4() 
                result = await send_command_request({
                    "request_id": request_id,
                    "command": cmd.get('command'),
                    "agent_Id": agentId
                })
                while True:
                    result = st.session_state.ui_message_queue.get_nowait()
                    if result.get("request_id") == request_id:
                        if result is None:
                            raise Exception("Unable to collect logs for this command")
                        data = json.loads(result)
                        print("logs collected for command", cmd.get('command'))
                        print(data)
                        logger.info(f"{ticketId} Command request satisfied: {cmd.get('command')}")
                        break
            except Exception as e:
                print(f"Error executing command {cmd.get('command')}: {e}")
                logger.error(f"{ticketId} Error executing command {cmd.get('command')}: {e}")
                result = ssh_run(cmd)
                logger.info(f"{ticketId}: Executed command: {cmd}")
                executed_results.append({
                    "command": cmd,
                    "output": result
                })
            except Exception as e:
                logger.error(f"{ticketId}: Error executing command {cmd}: {e}")

        # 3. Re-collect logs      
        result = await log_collector_node(ticketId, agentId, log_commands)
        if result is None:
            return {
                "summary": None,
                "error": "Failed to collect logs after troubleshooting"
            }
        data = json.loads(result)
        after_logs = data.get("data", None)
        if after_logs is None:
            return {
                "summary": None,
                "error": "Failed to collect logs after troubleshooting"
            }

        # 4. Compare logs
        # You can re-run diagnostics node here or use a comparison LLM
        prompt = FIX_COMPARE_PROMPT.format(
            beforeLog=json.dumps(before_logs, indent=2),
            afterLog=json.dumps(after_logs, indent=2)
        )
        response = llm.invoke(prompt).content
        if response is None:
            return {
                "summary": None,
                "error": "Failed to compare logs before and after troubleshooting"
            }
        output = json.loads(response)
        # return structured diagnostics
        return {"summary": output, "error": None}
    except Exception as e:
        print("Error occurred:", e)
        return {
            "summary": None,
            "error": str(e)
        }
  
async def notify_user(message, state="info"):
    await st.session_state.ui_message_queue.put({
        "message": message,
        "state": state
    })


async def start_auto_fix(payload):
    print("Starting auto fix...")
    ticketId = payload.ticketId
    agentId = payload.agentId
    category = payload.category
    context = payload.context
    
    selected_commands = payload.get("selected_commands", [])
    # 1. Collect logs
    if not selected_commands or len(selected_commands) == 0:
        selected_commands = log_commands_generator_node(category)
        if selected_commands is None:
            await notify_user("Failed to generate commands", "error")
            return
        data = {
            "key": "selected_commands",
            "value": selected_commands
        }
        await notify_user(json.dumps(data), "set")
    result = await log_collector_node(ticketId, agentId, category, selected_commands)
    
    if result is None:
        await notify_user("Failed to collect logs", "error")
        return
    
    data = json.loads(result)
    logs = data.get("data", None)
    if logs is None or logs == {}:
        await notify_user("Failed to collect logs", "error")
        return
    
    diagnostics_node_result = diagnostics_node(logs, context)
    
    # Build AI message with issues and commands
    if diagnostics_node_result and "detected_issues" in diagnostics_node_result:
        issues = diagnostics_node_result["detected_issues"]
        
        if issues:
            ai_msg_auto = "\n**Diagnostics completed. Issues detected:**\n"
            # Build formatted message for chat history
            # issues_message = "**Diagnostic Results:**\n\n**Issues Detected:**\n"
            for idx, issue in enumerate(issues, 1):
                issue_text = issue.get('issue', 'Unknown issue')
                human_intervention = issue.get('human_intervention_needed', False)
                if human_intervention:
                    ai_msg_auto += f" {idx}. {issue_text} - Human intervention needed."
                else:
                    ai_msg_auto += f" {idx}. {issue_text}"                                         
                
            if all(issue.get("suggested_commands") is None or len(issue.get("suggested_commands")) == 0 for issue in issues):
                ai_msg_auto += "\n\n**Sorry, we could not find any solution for this issue at the moment. Please consider escalating to a technician.**"
                await notify_user(ai_msg_auto, "human_intervention_needed")
                return
                
            elif any(issue.get("human_intervention_needed", False) and idx < len(issues) - idx for idx, issue in enumerate(issues)):
                ai_msg_auto += "\n\n**⚠️ Some critical issues require human intervention. Please consider escalating to a technician.**"
                await notify_user(ai_msg_auto, "human_intervention_needed")
                return
            else:
                await notify_user(ai_msg_auto, "info")
                # Execute troubleshooting node
                troubleshoot_result = await troubleshoot_node({
                    "logs": logs,
                    "detected_issues": issues,
                    "category": category,
                    "ticketId": ticketId,
                    "agentId": agentId,
                    "selected_commands": selected_commands
                })
                if troubleshoot_result is None or troubleshoot_result.get("error"):
                    await notify_user("Failed to troubleshoot", troubleshoot_result.get("error"))
                    return
                ai_msg_auto = "\n**Troubleshooting Summary:**\n\n"
                if troubleshoot_result and "summary" in troubleshoot_result:
                    summary = troubleshoot_result.get("summary")
                    if summary.get("issues_fixed") and len(summary.get("issues_fixed")) > 0:
                        ai_msg_auto += "**Issues Fixed:**\n\n"
                        for issue in summary.get("issues_fixed"):
                            issue_text = issue.get('issue', 'Unknown issue')
                            issue_details = issue.get('issue_details', 'Unknown details')
                            ai_msg_auto += f"- **{issue_text}**: {issue_details}\n"

                    if summary.get("issues_remaining") and len(summary.get("issues_remaining")) > 0:
                        ai_msg_auto += "**Issues Remaining:**\n\n"
                        for issue in summary.get("issues_remaining"):
                            issue_text = issue.get('issue', 'Unknown issue')
                            issue_details = issue.get('issue_details', 'Unknown details')
                            ai_msg_auto += f"- **{issue_text}**: {issue_details}\n"
                    
                    if summary.get("notes"):
                        ai_msg_auto += "**Notes:**\n"
                        notes = summary.get("notes")
                        ai_msg_auto += notes
                    
                    if summary.get("needs_human_intervention"):
                        ai_msg_auto += "\n_⚠️For further troubleshooting, system recommend to consider escalating to a technician._\n"       
                        await notify_user(ai_msg_auto, "human_intervention_needed")
                        return
                    
                    else:
                        ai_msg_auto += "\n✅ Troubleshooting completed. System appears to be functioning normally.\n"
                        await notify_user(ai_msg_auto, "success")
                        return
                else:
                    ai_msg_auto += "\nSorry, we could not find any solution for this issue at the moment. Please consider escalating to a technician.\n"
                    await notify_user(ai_msg_auto, "human_intervention_needed")
                    return
        else:
            no_issues_msg = "No issues detected. System appears to be functioning normally."
            await notify_user(no_issues_msg, "human_intervention_needed")
            return
    else:
        warning_msg = "⚠️ Diagnostics completed but no results were returned."
        await notify_user(warning_msg)
        return