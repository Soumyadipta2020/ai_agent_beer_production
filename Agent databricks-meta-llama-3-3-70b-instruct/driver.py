# Databricks notebook source
# MAGIC %md
# MAGIC #Tool-calling Agent
# MAGIC
# MAGIC This is an auto-generated notebook created by an AI Playground export.
# MAGIC
# MAGIC This notebook uses [Mosaic AI Agent Framework](https://docs.databricks.com/generative-ai/agent-framework/build-genai-apps.html) to recreate your agent from the AI Playground. It  demonstrates how to develop, manually test, evaluate, log, and deploy a tool-calling agent in LangGraph.
# MAGIC
# MAGIC The agent code implements [MLflow's ChatAgent](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatAgent) interface, a Databricks-recommended open-source standard that simplifies authoring multi-turn conversational agents, and is fully compatible with Mosaic AI agent framework functionality.
# MAGIC
# MAGIC  **_NOTE:_**  This notebook uses LangChain, but AI Agent Framework is compatible with any agent authoring framework, including LlamaIndex or pure Python agents written with the OpenAI SDK.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - Address all `TODO`s in this notebook.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow-skinny[databricks] langgraph==0.3.4 databricks-langchain databricks-agents uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Define the agent in code
# MAGIC Below we define our agent code in a single cell, enabling us to easily write it to a local Python file for subsequent logging and deployment using the `%%writefile` magic command.
# MAGIC
# MAGIC For more examples of tools to add to your agent, see [docs](https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html).

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC from typing import Any, Generator, Optional, Sequence, Union
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import (
# MAGIC     ChatDatabricks,
# MAGIC     VectorSearchRetrieverTool,
# MAGIC     DatabricksFunctionClient,
# MAGIC     UCFunctionToolkit,
# MAGIC     set_uc_function_client,
# MAGIC )
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langchain_core.tools import BaseTool
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.graph import CompiledGraph
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC client = DatabricksFunctionClient()
# MAGIC set_uc_function_client(client)
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC system_prompt = """You are an agent who have information about monthly beer production in Austria. You will answer the questions that only relevant to beer production in Austria only."""
# MAGIC
# MAGIC ###############################################################################
# MAGIC ## Define tools for your agent, enabling it to retrieve data or take actions
# MAGIC ## beyond text generation
# MAGIC ## To create and see usage examples of more tools, see
# MAGIC ## https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html
# MAGIC ###############################################################################
# MAGIC tools = []
# MAGIC
# MAGIC # You can use UDFs in Unity Catalog as agent tools
# MAGIC uc_tool_names = ["workspace.my_data.total_beer_production", "workspace.my_data.avg_beer_production"]
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
# MAGIC tools.extend(uc_toolkit.tools)
# MAGIC
# MAGIC #####################
# MAGIC ## Define agent logic
# MAGIC #####################
# MAGIC def create_tool_calling_agent(
# MAGIC     model: LanguageModelLike,
# MAGIC     tools: Union[Sequence[BaseTool], ToolNode],
# MAGIC     system_prompt: Optional[str] = None,
# MAGIC ) -> CompiledGraph:
# MAGIC     model = model.bind_tools(tools)
# MAGIC
# MAGIC     # Define the function that determines which node to go to
# MAGIC     def should_continue(state: ChatAgentState):
# MAGIC         messages = state["messages"]
# MAGIC         last_message = messages[-1]
# MAGIC         # If there are function calls, continue. else, end
# MAGIC         if last_message.get("tool_calls"):
# MAGIC             return "continue"
# MAGIC         else:
# MAGIC             return "end"
# MAGIC
# MAGIC     if system_prompt:
# MAGIC         preprocessor = RunnableLambda(
# MAGIC             lambda state: [{"role": "system", "content": system_prompt}]
# MAGIC             + state["messages"]
# MAGIC         )
# MAGIC     else:
# MAGIC         preprocessor = RunnableLambda(lambda state: state["messages"])
# MAGIC     model_runnable = preprocessor | model
# MAGIC
# MAGIC     def call_model(
# MAGIC         state: ChatAgentState,
# MAGIC         config: RunnableConfig,
# MAGIC     ):
# MAGIC         response = model_runnable.invoke(state, config)
# MAGIC
# MAGIC         return {"messages": [response]}
# MAGIC
# MAGIC     workflow = StateGraph(ChatAgentState)
# MAGIC
# MAGIC     workflow.add_node("agent", RunnableLambda(call_model))
# MAGIC     workflow.add_node("tools", ChatAgentToolNode(tools))
# MAGIC
# MAGIC     workflow.set_entry_point("agent")
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "agent",
# MAGIC         should_continue,
# MAGIC         {
# MAGIC             "continue": "tools",
# MAGIC             "end": END,
# MAGIC         },
# MAGIC     )
# MAGIC     workflow.add_edge("tools", "agent")
# MAGIC
# MAGIC     return workflow.compile()
# MAGIC
# MAGIC
# MAGIC class LangGraphChatAgent(ChatAgent):
# MAGIC     def __init__(self, agent: CompiledStateGraph):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC
# MAGIC         messages = []
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 messages.extend(
# MAGIC                     ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
# MAGIC                 )
# MAGIC         return ChatAgentResponse(messages=messages)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> Generator[ChatAgentChunk, None, None]:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 yield from (
# MAGIC                     ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
# MAGIC                 )
# MAGIC
# MAGIC
# MAGIC # Create the agent object, and specify it as the agent object to use when
# MAGIC # loading the agent back for inference via mlflow.models.set_model()
# MAGIC agent = create_tool_calling_agent(llm, tools, system_prompt)
# MAGIC AGENT = LangGraphChatAgent(agent)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. Since this notebook called `mlflow.langchain.autolog()` you can view the trace for each step the agent takes.
# MAGIC
# MAGIC Replace this placeholder input with an appropriate domain-specific example for your agent.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from agent import AGENT

AGENT.predict({"messages": [{"role": "user", "content": "Average beer production in Austria till now?"}]})

# COMMAND ----------

for event in AGENT.predict_stream(
    {"messages": [{"role": "user", "content": "total beer production in 1980s?"}]}
):
    print(event, "-----------\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Determine Databricks resources to specify for automatic auth passthrough at deployment time
# MAGIC - **TODO**: If your Unity Catalog Function queries a [vector search index](https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html) or leverages [external functions](https://docs.databricks.com/generative-ai/agent-framework/external-connection-tools.html), you need to include the dependent vector search index and UC connection objects, respectively, as resources. See [docs](https://docs.databricks.com/generative-ai/agent-framework/log-agent.html#specify-resources-for-automatic-authentication-passthrough) for more details.
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

# Determine Databricks resources to specify for automatic auth passthrough at deployment time
import mlflow
from agent import LLM_ENDPOINT_NAME, tools
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from pkg_resources import get_distribution
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]
for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        # TODO: If the UC function includes dependencies like external connection or vector search, please include them manually.
        # See the TODO in the markdown above for more information.
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "What's the total beer production till now"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        input_example=input_example,
        resources=resources,
        pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"mlflow=={get_distribution('mlflow').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
            f"langgraph=={get_distribution('langgraph').version}",
        ],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with [Agent Evaluation](https://docs.databricks.com/mlflow3/genai/eval-monitor)
# MAGIC
# MAGIC You can edit the requests or expected responses in your evaluation dataset and run evaluation as you iterate your agent, leveraging mlflow to track the computed quality metrics.
# MAGIC
# MAGIC Evaluate your agent with one of our [predefined LLM scorers](https://docs.databricks.com/mlflow3/genai/eval-monitor/predefined-judge-scorers), or try adding [custom metrics](https://docs.databricks.com/mlflow3/genai/eval-monitor/custom-scorers).

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety, RetrievalRelevance, RetrievalGroundedness

eval_dataset = [
    {
        "inputs": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an agent who have information about monthly beer production in Austria. You will answer the questions that only relevant to beer production in Austria only."
                },
                {
                    "role": "user",
                    "content": "What's the total beer production till now"
                }
            ]
        },
        "expected_response": None
    },
    {
        "inputs": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an agent who have information about monthly beer production in Austria. You will answer the questions that only relevant to beer production in Austria only."
                },
                {
                    "role": "user",
                    "content": "What's the total beer production till now"
                },
                {
                    "role": "assistant",
                    "content": "To find the total beer production till now, we need to consider all the months for which data is available. Since the data ranges from 1956 to 1995, we can set the min_month to 'January' and the max_month to 'December' to get the total production for the entire period.\n\n<uc_function_call>{\"id\":\"FIHzHOblI3an4RodZtCH\",\"name\":\"workspace.my_data.total_beer_production\",\"arguments\":\"{\\\"min_month\\\":\\\"January\\\",\\\"max_month\\\":\\\"December\\\"}\"}</uc_function_call>\n\n"
                },
                {
                    "role": "user",
                    "content": "What's the total beer production till now?"
                }
            ]
        },
        "expected_response": None
    },
    {
        "inputs": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an agent who have information about monthly beer production in Austria. You will answer the questions that only relevant to beer production in Austria only."
                },
                {
                    "role": "user",
                    "content": "What's the total beer production till now"
                },
                {
                    "role": "assistant",
                    "content": "To find the total beer production till now, we need to consider all the months for which data is available. Since the data ranges from 1956 to 1995, we can set the min_month to 'January' and the max_month to 'December' to get the total production for the entire period.\n\n<uc_function_call>{\"id\":\"FIHzHOblI3an4RodZtCH\",\"name\":\"workspace.my_data.total_beer_production\",\"arguments\":\"{\\\"min_month\\\":\\\"January\\\",\\\"max_month\\\":\\\"December\\\"}\"}</uc_function_call>\n\n"
                },
                {
                    "role": "user",
                    "content": "What's the total beer production till now?"
                },
                {
                    "role": "assistant",
                    "content": "To find the total beer production till now, we need to consider the entire available data range, from the earliest month to the latest month. The earliest month in the data is 1956-01-01 and the latest month is 1995-12-01. Therefore, we use the function workspace.my_data.total_beer_production with min_month as 1956-01-01 and max_month as 1995-12-01.\n\n<uc_function_call>{\"id\":\"ZplrWvl2IvLYqzR8MMay\",\"name\":\"workspace.my_data.total_beer_production\",\"arguments\":\"{\\\"min_month\\\":\\\"1956-01-01\\\",\\\"max_month\\\":\\\"1995-12-01\\\"}\"}</uc_function_call>\n\n"
                },
                {
                    "role": "user",
                    "content": "Give me the number of total beer production only"
                }
            ]
        },
        "expected_response": None
    },
    {
        "inputs": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an agent who have information about monthly beer production in Austria. You will answer the questions that only relevant to beer production in Austria only."
                },
                {
                    "role": "user",
                    "content": "What's the total beer production till now"
                },
                {
                    "role": "assistant",
                    "content": "To find the total beer production till now, we need to consider all the months for which data is available. Since the data ranges from 1956 to 1995, we can set the min_month to 'January' and the max_month to 'December' to get the total production for the entire period.\n\n<uc_function_call>{\"id\":\"FIHzHOblI3an4RodZtCH\",\"name\":\"workspace.my_data.total_beer_production\",\"arguments\":\"{\\\"min_month\\\":\\\"January\\\",\\\"max_month\\\":\\\"December\\\"}\"}</uc_function_call>\n\n"
                },
                {
                    "role": "user",
                    "content": "What's the total beer production till now?"
                },
                {
                    "role": "assistant",
                    "content": "To find the total beer production till now, we need to consider the entire available data range, from the earliest month to the latest month. The earliest month in the data is 1956-01-01 and the latest month is 1995-12-01. Therefore, we use the function workspace.my_data.total_beer_production with min_month as 1956-01-01 and max_month as 1995-12-01.\n\n<uc_function_call>{\"id\":\"ZplrWvl2IvLYqzR8MMay\",\"name\":\"workspace.my_data.total_beer_production\",\"arguments\":\"{\\\"min_month\\\":\\\"1956-01-01\\\",\\\"max_month\\\":\\\"1995-12-01\\\"}\"}</uc_function_call>\n\n"
                },
                {
                    "role": "user",
                    "content": "Give me the number of total beer production only"
                },
                {
                    "role": "assistant",
                    "content": "To find the total beer production, we need to consider the entire available data range. The function workspace.my_data.total_beer_production is used with min_month as 1956-01-01 and max_month as 1995-12-01.\n\n<uc_function_call>{\"id\":\"p4lUoMEz1P9S7KYEISlU\",\"name\":\"workspace.my_data.total_beer_production\",\"arguments\":\"{\\\"min_month\\\":\\\"1956-01-01\\\",\\\"max_month\\\":\\\"1995-12-01\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"p4lUoMEz1P9S7KYEISlU\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[64924.199999999975]]}\"}</uc_function_result>\n\nThe total beer production is 64924.2."
                },
                {
                    "role": "user",
                    "content": "And how much is average beer production?"
                }
            ]
        },
        "expected_response": None
    },
    {
        "inputs": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an agent who have information about monthly beer production in Austria. You will answer the questions that only relevant to beer production in Austria only."
                },
                {
                    "role": "user",
                    "content": "What's the total beer production till now"
                },
                {
                    "role": "assistant",
                    "content": "To find the total beer production till now, we need to consider all the months for which data is available. Since the data ranges from 1956 to 1995, we can set the min_month to 'January' and the max_month to 'December' to get the total production for the entire period.\n\n<uc_function_call>{\"id\":\"FIHzHOblI3an4RodZtCH\",\"name\":\"workspace.my_data.total_beer_production\",\"arguments\":\"{\\\"min_month\\\":\\\"January\\\",\\\"max_month\\\":\\\"December\\\"}\"}</uc_function_call>\n\n"
                },
                {
                    "role": "user",
                    "content": "What's the total beer production till now?"
                },
                {
                    "role": "assistant",
                    "content": "To find the total beer production till now, we need to consider the entire available data range, from the earliest month to the latest month. The earliest month in the data is 1956-01-01 and the latest month is 1995-12-01. Therefore, we use the function workspace.my_data.total_beer_production with min_month as 1956-01-01 and max_month as 1995-12-01.\n\n<uc_function_call>{\"id\":\"ZplrWvl2IvLYqzR8MMay\",\"name\":\"workspace.my_data.total_beer_production\",\"arguments\":\"{\\\"min_month\\\":\\\"1956-01-01\\\",\\\"max_month\\\":\\\"1995-12-01\\\"}\"}</uc_function_call>\n\n"
                },
                {
                    "role": "user",
                    "content": "Give me the number of total beer production only"
                },
                {
                    "role": "assistant",
                    "content": "To find the total beer production, we need to consider the entire available data range. The function workspace.my_data.total_beer_production is used with min_month as 1956-01-01 and max_month as 1995-12-01.\n\n<uc_function_call>{\"id\":\"p4lUoMEz1P9S7KYEISlU\",\"name\":\"workspace.my_data.total_beer_production\",\"arguments\":\"{\\\"min_month\\\":\\\"1956-01-01\\\",\\\"max_month\\\":\\\"1995-12-01\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"p4lUoMEz1P9S7KYEISlU\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[64924.199999999975]]}\"}</uc_function_result>\n\nThe total beer production is 64924.2."
                },
                {
                    "role": "user",
                    "content": "And how much is average beer production?"
                },
                {
                    "role": "assistant",
                    "content": "To find the average beer production, we need to calculate the average production over the entire available data range, from the earliest month to the latest month.\n\n<uc_function_call>{\"id\":\"dqEy7IaEuQJfg78uaY1w\",\"name\":\"workspace.my_data.avg_beer_production\",\"arguments\":\"{\\\"min_month\\\":\\\"1956-01-01\\\",\\\"max_month\\\":\\\"1995-12-01\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"dqEy7IaEuQJfg78uaY1w\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[136.39537815126045]]}\"}</uc_function_result>\n\nThe average beer production is 136.40."
                },
                {
                    "role": "user",
                    "content": "What's the average beer production month wise Jan, Feb etc.?"
                }
            ]
        },
        "expected_response": None
    }
]

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda messages: AGENT.predict({"messages": messages}),
    scorers=[RelevanceToQuery(), Safety()], # add more scorers here if they're applicable
)

# Review the evaluation results in the MLfLow UI (see console output)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform pre-deployment validation of the agent
# MAGIC Before registering and deploying the agent, we perform pre-deployment checks via the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See [documentation](https://docs.databricks.com/machine-learning/model-serving/model-serving-debug.html#validate-inputs) for details

# COMMAND ----------

import mlflow

# Load the model
model = mlflow.pyfunc.load_model(model_uri=f"runs:/{logged_agent_info.run_id}/agent")

# Prepare input data
input_data = {"messages": [{"role": "user", "content": "What's the average beer production till now?"}]}

# Perform prediction
predictions = model.predict(input_data)

# Display the predictions
display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "workspace"
schema = "my_data"
model_name = "llama_3_beer_prod"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents

agents.deploy(
    UC_MODEL_NAME,
    uc_registered_model_info.version,
    tags={"endpointSource": "playground"},
    scale_to_zero=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See [docs](https://docs.databricks.com/generative-ai/deploy-agent.html) for details
