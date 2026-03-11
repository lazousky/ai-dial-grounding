import asyncio
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `flow_diagram.png` to see the flow of app

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY: 
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }

llm_client = AzureChatOpenAI(
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    azure_deployment="gpt-4o-mini",
    api_version="",
)

token_tracker = TokenTracker()


def join_context(context: list[dict[str, Any]]) -> str:
    lines = []
    for user in context:
        lines.append("User:")
        for key, value in user.items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    response = await llm_client.ainvoke(messages)
    total_tokens = response.response_metadata["token_usage"]["total_tokens"]
    token_tracker.add_tokens(total_tokens)
    print(response.content)
    print(f"Tokens used: {total_tokens}")
    return response.content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")

        users = UserClient().get_all_users()
        user_batches = [users[i:i + 100] for i in range(0, len(users), 100)]

        tasks = [
            generate_response(
                BATCH_SYSTEM_PROMPT,
                USER_PROMPT.format(context=join_context(batch), query=user_question),
            )
            for batch in user_batches
        ]
        batch_results = await asyncio.gather(*tasks)

        matches = [r for r in batch_results if "NO_MATCHES_FOUND" not in r]

        if matches:
            combined = "\n\n".join(matches)
            await generate_response(
                FINAL_SYSTEM_PROMPT,
                USER_PROMPT.format(context=combined, query=user_question),
            )
        else:
            print("No users found matching the query.")

        print(f"\nToken usage summary: {token_tracker.get_summary()}")


if __name__ == "__main__":
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation