#### Root Cause Analysis Agent ####
""" Azure AI Foundry Agent that analyzes system logs to identify root causes of issues. """

import os

from azure.ai.agents import AgentsClient
from azure.ai.agents.models import Agent, MessageRole, ListSortOrder
from azure.identity import DefaultAzureCredential

class RootCauseAnalysisAgent:

    def __init__(self):

        # Create the agents client
        self.client = AgentsClient(
            endpoint=os.environ['PROJECT_ENDPOINT'],
            credential=DefaultAzureCredential(
                exclude_environment_credential=True,
                exclude_managed_identity_credential=True
            )
        )

        self.agent: Agent | None = None

    async def create_agent(self) -> Agent:
        if self.agent:
            return self.agent

        # Create the title agent
        self.agent = self.client.create_agent(
            model=os.environ['MODEL_DEPLOYMENT_NAME'],
            name='root-cause-analysis-agent',
            instructions="""
            You are a system log analysis assistant.
            Given system logs, identify potential root causes of issues. Share the root cause analysis in a structured format.
            Always provide clear and concise explanations.
            """,
        )
        return self.agent

    async def run_conversation(self, user_message: str) -> list[str]:
        if not self.agent:
            await self.create_agent()

        # Create a thread for the chat session
        thread = self.client.threads.create()

        # Send user message
        self.client.messages.create(thread_id=thread.id, role=MessageRole.USER, content=user_message)

        # Create and run the agent
        run = self.client.runs.create_and_process(thread_id=thread.id, agent_id=self.agent.id)

        if run.status == 'failed':
            print(f'Root Cause Analysis Agent: Run failed - {run.last_error}')
            return [f'Error: {run.last_error}']

        # Get response messages
        messages = self.client.messages.list(thread_id=thread.id, order=ListSortOrder.DESCENDING)
        responses = []
        for msg in messages:
            # Only get the latest assistant response
            if msg.role == 'assistant' and msg.text_messages:
                for text_msg in msg.text_messages:
                    responses.append(text_msg.text.value)
                break 

        return responses if responses else ['No response received']

async def create_foundry_RCA_agent() -> RootCauseAnalysisAgent:
    agent = RootCauseAnalysisAgent()
    await agent.create_agent()
    return agent



#===============================================
# Execute the Root Cause Analysis Agent
#===============================================



""" Azure AI Foundry Agent that generates an outline """

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentCard, Part, TaskState
from a2a.utils.message import new_agent_text_message
#from outline_agent.agent import OutlineAgent, create_foundry_outline_agent

# An AgentExecutor that runs Azure AI Foundry-based agents. Adapted from the ADK agent executor pattern.
class RootCauseAnalysisAgentExecutor(AgentExecutor):

    def __init__(self, card: AgentCard):
        self._card = card
        self._foundry_agent: RootCauseAnalysisAgent | None = None

    async def _get_or_create_agent(self) -> RootCauseAnalysisAgent:
        if not self._foundry_agent:
            self._foundry_agent = await create_foundry_RCA_agent()
        return self._foundry_agent

    async def _process_request(self, message_parts: list[Part], context_id: str, task_updater: TaskUpdater) -> None:
        # Process a user request through the Foundry agent

        try:
            # Retrieve message from A2A parts
            user_message = message_parts[0].root.text

            # Get the RCA agent
            agent = await self._get_or_create_agent()

            # Update the task status
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message('RCA Agent is processing your request...', context_id=context_id)
            )

            # Run the conversation
            responses = await agent.run_conversation(user_message)

            # Update the task with responses
            for response in responses:
                await task_updater.update_status(
                    TaskState.working,
                    message=new_agent_text_message( response, context_id=context_id)
                )

            # Mark the task as complete
            final_message = responses[-1] if responses else 'Task completed.'
            await task_updater.complete(
                message=new_agent_text_message(final_message, context_id=context_id)
            )

        except Exception as e:
            await task_updater.failed(
                message=new_agent_text_message('RCA Agent failed to process the request.', 
                context_id=context_id)
            )

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        
        # Create task updater
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.submit()

        # Start working
        await updater.start_work()

        # Process the request
        await self._process_request(context.message.parts, context.context_id, updater)

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        print(f'RCA Agent: Cancelling execution for context {context.context_id}')

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.failed(
            message=new_agent_text_message('Task cancelled by user', context_id=context.context_id)
        )

def create_foundry_agent_executor(card: AgentCard) -> RootCauseAnalysisAgentExecutor:
    return RootCauseAnalysisAgentExecutor(card)

##############################################
#Server Code to host the RCA Agent
###############################################
import os
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv
#from outline_agent.agent_executor import create_foundry_agent_executor
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
import asyncio

load_dotenv()

#host = os.environ["SERVER_URL"]
#port = os.environ["TITLE_AGENT_PORT"]
host = "127.0.0.1"
port = "8001"

# Define agent skills
skills = [
    AgentSkill(
        id='Generate RCA',
        name='Generate RCA',
        description='Generates a root cause analysis based on a topic',
        tags=['rca'],
        examples=[
            'Can you analyze these system logs and provide a root cause analysis?',
        ],
    ),
]

# Create agent card
agent_card = AgentCard(
    name='AI Foundry RCA Agent',
    description='An intelligent root cause analysis agent powered by Azure AI Foundry. '
    'I can help you analyze system logs and provide root cause analysis.',
    url=f'http://{host}:{port}/',
    version='1.0.0',
    default_input_modes=['text'],
    default_output_modes=['text'],
    capabilities=AgentCapabilities(streaming=True),
    skills=skills,
)

# Create agent executor
agent_executor = RootCauseAnalysisAgentExecutor(agent_card)

# Create request handler
request_handler = DefaultRequestHandler(
    agent_executor=agent_executor, task_store=InMemoryTaskStore()
)

# Create A2A application
a2a_app = A2AStarletteApplication(
    agent_card=agent_card, http_handler=request_handler
)

# Get routes
routes = a2a_app.routes()

# Add health check endpoint
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse('AI Foundry RCA Agent is running!')

routes.append(Route(path='/health', methods=['GET'], endpoint=health_check))

# Create Starlette app
app = Starlette(routes=routes)


def main():
    # Run the server
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == '__main__':
    main()
