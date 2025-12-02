from utils import ssh_run 

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

    # for key, cmd in LOG_COMMANDS.items():
    #     collected[key] = ssh_run(cmd, host, username, password)
    for cmd in selected_commands:
        try:
            logs[cmd] = ssh_run(cmd)
        except:
            logs[cmd] = "ERROR executing command"
    return {"logs": logs}


from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import json

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    )
    
SAFE_COMMAND_GENERATION_PROMPT = """
    You are a safe Linux troubleshooting assistant.

    You receive:
    - System logs: {logs}
    - Context: {context}

    Your job:
    1. Identify potential issues in the logs
    2. Suggest possible SAFE commands to fix these issues
    3. ALL commands must:
    - Be non-destructive
    - CAn restart services
    - Flush cache
    - Collect statuses
    - NEVER delete files, kill random processes, or modify system files

    Allowed command types must be limited to
    - restarting services
    - checking status
    - clearing caches
    - network resets
    - restarting NetworkManager
    - checking disk usage
    - checking CPU usage
    - checking memory usage
    - Identify heavy applications running

    Never go outside these categories.

    CRITICAL: Return ONLY raw JSON with no markdown formatting, no code blocks, no backticks.
    Just the JSON object starting with {{ and ending with }}. If no issue is found, return nothing!

    Format:
    {{"issues": [{{"issue": "description with possible solutions", "suggested_commands": [...]}}]}}
    """

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

# ------------- Troubleshooting Node -------------
from utils import is_blacklisted, ssh_run
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import json

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    )

FIX_COMPARE_PROMPT = """
You are a Linux system troublshoot verification assistant.

Compare BEFORE logs and AFTER logs.

Identify:
- Which issues appear fixed
- Which issues still remain
- Any new warnings or errors

Before log: {beforeLog}
After log: {afterLog}

CRITICAL: Return ONLY raw JSON with no markdown formatting, no code blocks, no backticks.
Just the JSON object starting with {{ and ending with }}.

Format:
{{
  "fix_status": "solved" | "partial" | "failed",
  "issues_fixed": [...],
  "issues_remaining": [...],
  "notes": "..."
}}
"""


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