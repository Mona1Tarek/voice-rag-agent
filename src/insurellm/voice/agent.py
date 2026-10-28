
from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import inference
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    RunContext,
    TurnHandlingOptions,
    function_tool,
)
from livekit.plugins import silero

from insurellm.rag.pipeline import generate_answer
from insurellm.rag.vector_store import get_vector_store
from insurellm.llm.client import LLMClient


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("insurellm.voice")


def required_env(name: str) -> str:
    """Read a required model configuration from .env."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing {name} in .env. Configure your LiveKit Inference model."
        )
    return value


class InsurellmVoiceAgent(Agent):
    def __init__(self, vector_store, llm_client) -> None:
        # Pass `vector_store` and `llm_client` as arguments to the constructor, and store them in the instance
        # So that each call to search_knowledge_base() can use the same objects.
        self.vector_store = vector_store
        self.llm_client = llm_client

        super().__init__(
            instructions="""
                You are the voice assistant for Insurellm.

                Your role is to answer questions about Insurellm using its knowledge base.

                Rules:
                - Use the search_knowledge_base tool for questions about Insurellm,
                its company information, employees, products, and contracts.
                - Do not invent facts that are not supported by the retrieved information.
                - If the knowledge base does not contain the answer, say so clearly.
                - Speak naturally and concisely.
                - Reply in the language the user speaks, including Arabic or English.
                - Do not read citation IDs, document IDs, or technical metadata aloud.
                - If the user asks a general question unrelated to Insurellm, answer briefly
                only when it does not require information from the private knowledge base.
                """
        )

    @function_tool
    async def search_knowledge_base(
        self,
        context: RunContext,
        question: str,
    ) -> str:
        """Search Insurellm's knowledge base and answer a question using RAG.

        Use this tool whenever the user asks about Insurellm company information,
        employees, products, or contracts.
        """
        logger.info("RAG question: %s", question)

        try:
            # The existing RAG pipeline is synchronous, so run it outside
            # the event loop to avoid blocking LiveKit's audio processing.
            result = await asyncio.to_thread(
                generate_answer,
                question,
                stream=False,
                vector_store=self.vector_store,
                llm_client=self.llm_client,
            )

            answer = result.get("answer", "")
            citations = result.get("generated_citations", [])
            retrieved = result.get("retrieved_documents", [])

            # Keep RAG evidence visible for debugging and evaluation.
            logger.info("RAG answer: %s", answer)
            logger.info("Generated citations: %s", citations)
            logger.info("Retrieved documents: %d", len(retrieved))

            for item in retrieved:
                if isinstance(item, tuple):
                    document, score = item
                else:
                    document, score = item, None

                metadata = getattr(document, "metadata", {})
                logger.info(
                    "Retrieved source=%s score=%s",
                    metadata.get("source", "unknown"),
                    score,
                )

            if not answer:
                return (
                    "I couldn't generate a reliable answer from the "
                    "knowledge base. Please try rephrasing your question."
                )

            # Return only the answer to the conversational LLM.
            # Citations remain in logs and in the RAG result.
            return answer

        except Exception:
            logger.exception("RAG query failed")
            return (
                "I'm having trouble searching the knowledge base right now. "
                "Please try again."
            )


server = AgentServer()

# `entrypoint()` is executed for every session
# If we have many concurrent sessions, it’s worth considering whether it would be better to share dependencies at the worker level
@server.rtc_session(agent_name="insurellm-rag")
async def entrypoint(ctx: agents.JobContext) -> None:
    await ctx.connect()

    llm_client = await asyncio.to_thread(LLMClient)
    vector_store = await asyncio.to_thread(get_vector_store)
    
    session = AgentSession(
        stt=required_env("LIVEKIT_STT_MODEL"),
        llm=required_env("LIVEKIT_LLM_MODEL"),
        tts=required_env("LIVEKIT_TTS_MODEL"),
        vad=silero.VAD.load(),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
        ),
    )

    await session.start(
        room=ctx.room,
        agent=InsurellmVoiceAgent(
            vector_store=vector_store,
            llm_client=llm_client,
            ),
    )

    await session.generate_reply(
        instructions=(
            "Greet the user briefly. Introduce yourself as the Insurellm "
            "voice assistant and ask how you can help."
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
