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
        except:
            logs[cmd] = "ERROR executing command"

    print(logs)
    return {"logs": logs}


from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import json

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
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
import json

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    )

def troubleshoot_node(state):
    issues = state.get("detected_issues", [])
    before_logs = state["logs"]
    safe_actions = []
    executed_results = []

    # 1. Extract suggested commands
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
    for cmd in safe_actions:
        result = ssh_run(cmd)
        executed_results.append({
            "command": cmd,
            "output": result
        })

    # 3. Re-collect logs
    after_logs_state = log_collector_node(state)
    after_logs = after_logs_state["logs"]

    # You can re-run diagnostics node here or use a comparison LLM
    prompt = FIX_COMPARE_PROMPT.format(
        beforeLog=json.dumps(before_logs, indent=2),
        afterLog=json.dumps(after_logs, indent=2)
    )

    try:
        response = llm.invoke(prompt).content
        if response is None:
            return state
        output = json.loads(response)
        # return structured diagnostics
        return {"summary": output}
    except Exception as e:
        print("Error occurred:", e)
        return state