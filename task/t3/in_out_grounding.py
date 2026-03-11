import asyncio
import json
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# Chroma DB, async add documents: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
# Chroma DB, get all records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
# Chroma DB, delete records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)


SYSTEM_PROMPT = """You are a hobby extraction assistant. Given user profiles, extract hobbies and map them to user IDs.

## Instructions:
1. Analyze the user question to understand what hobbies/activities are being searched for
2. Review each user profile in the RAG CONTEXT
3. Extract users whose about_me section matches the searched hobbies
4. Group users by their specific hobby names
5. Return ONLY user IDs - do not reproduce any personal information

## Response Format:
{format_instructions}
"""

USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUESTION:
{query}"""


class HobbyGroup(BaseModel):
    hobby: str = Field(description="The specific hobby name")
    user_ids: list[int] = Field(description="List of user IDs who have this hobby", default_factory=list)


class HobbyExtractionResult(BaseModel):
    hobby_groups: list[HobbyGroup] = Field(
        description="Groups of users organised by hobby",
        default_factory=list,
    )


def format_user_document(user: dict[str, Any]) -> str:
    return f"User ID: {user['id']}\nAbout me: {user.get('about_me', '')}"


class HobbiesWizard:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.embeddings = embeddings
        self.llm_client = llm_client
        self.user_client = UserClient()
        self.vectorstore: Optional[Chroma] = None

    async def __aenter__(self):
        print("🔎 Loading all users...")
        users = self.user_client.get_all_users()
        documents = [
            Document(id=str(user["id"]), page_content=format_user_document(user))
            for user in users
        ]
        self.vectorstore = Chroma(embedding_function=self.embeddings)
        await self.vectorstore.aadd_documents(documents)
        print("✅ Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _sync_vectorstore(self, current_users: list[dict[str, Any]]):
        current_ids = {str(user["id"]) for user in current_users}
        stored = self.vectorstore.get()
        stored_ids = set(stored["ids"])

        deleted_ids = stored_ids - current_ids
        if deleted_ids:
            self.vectorstore.delete(ids=list(deleted_ids))
            print(f"Removed {len(deleted_ids)} deleted users from vectorstore")

        new_ids = current_ids - stored_ids
        if new_ids:
            new_users = [u for u in current_users if str(u["id"]) in new_ids]
            new_docs = [
                Document(id=str(u["id"]), page_content=format_user_document(u))
                for u in new_users
            ]
            await self.vectorstore.aadd_documents(new_docs)
            print(f"Added {len(new_ids)} new users to vectorstore")

    async def retrieve_context(self, query: str, k: int = 20, score: float = 0.1) -> str:
        current_users = self.user_client.get_all_users()
        await self._sync_vectorstore(current_users)

        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k, score_threshold=score)
        context_parts = []
        for doc, relevance_score in results:
            context_parts.append(doc.page_content)
            print(f"Score: {relevance_score}\n{doc.page_content}")
        return "\n\n".join(context_parts)

    def extract_hobbies(self, context: str, query: str) -> HobbyExtractionResult:
        parser = PydanticOutputParser(pydantic_object=HobbyExtractionResult)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT.format(context=context, query=query)),
        ]).partial(format_instructions=parser.get_format_instructions())
        return (prompt | self.llm_client | parser).invoke({})

    async def output_grounding(self, hobby_result: HobbyExtractionResult) -> dict[str, list[dict[str, Any]]]:
        result = {}
        for hobby_group in hobby_result.hobby_groups:
            users = []
            for user_id in hobby_group.user_ids:
                try:
                    user = await self.user_client.get_user(user_id)
                    users.append(user)
                except Exception:
                    print(f"User {user_id} not found, skipping")
            if users:
                result[hobby_group.hobby] = users
        return result

    async def search(self, query: str) -> dict[str, list[dict[str, Any]]]:
        context = await self.retrieve_context(query)
        hobby_result = self.extract_hobbies(context, query)
        return await self.output_grounding(hobby_result)


async def main():
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        azure_deployment="text-embedding-3-small-1",
        dimensions=384,
    )
    llm_client = AzureChatOpenAI(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        azure_deployment="gpt-4o-mini",
        api_version="",
    )

    async with HobbiesWizard(embeddings, llm_client) as wizard:
        print("Query samples:")
        print(" - I need people who love to go to mountains")
        print(" - Find users interested in painting")
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break
            result = await wizard.search(user_question)
            print(json.dumps(result, indent=2))


asyncio.run(main())
