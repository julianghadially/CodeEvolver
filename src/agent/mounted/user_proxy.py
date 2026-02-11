from sandbox.mounted.utils import timer_printer

def create_user_proxy(prefix: str = "AGENT", handle_questions: bool = True):
    """Create a user proxy callback for autonomous agent execution.

    Args:
        prefix: Log prefix (e.g., "AGENT" or "REFLECT")
        handle_questions: If True, auto-answer AskUserQuestion with first option

    Returns:
        Async function suitable for can_use_tool callback
    """
    async def user_proxy(tool_name: str, input_data: dict, context):
        # Import here to avoid issues when module is loaded outside agent context
        from claude_agent_sdk.types import PermissionResultAllow

        if tool_name == "ExitPlanMode":
            timer_printer(f"User proxy: ExitPlanMode")
            print(f"[{prefix}] User proxy: Auto-approving ExitPlanMode")
            return PermissionResultAllow(updated_input=input_data)

        if tool_name == "AskUserQuestion" and handle_questions:
            timer_printer(f"User proxy: AskUserQuestion")
            questions = input_data.get("questions", [])
            answers = {}
            for q in questions:
                options = q.get("options", [])
                if options:
                    answers[q["question"]] = options[0]["label"]
                    print(f"[{prefix}] User proxy: Auto-answering '{q['question'][:50]}...' with '{options[0]['label']}'")
                else:
                    answers[q["question"]] = "Proceed with default"
            return PermissionResultAllow(updated_input={
                "questions": questions,
                "answers": answers
            })

        return PermissionResultAllow(updated_input=input_data)

    return user_proxy

