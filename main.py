from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from agents.chains import mascot_chain

load_dotenv()


search_tool = TavilySearch(max_results=2)
tools = [search_tool]

llm = ChatGroq(model="llama-3.1-8b-instant")

llm_with_tools = llm.bind_tools(tools=tools)


class MascotState(TypedDict):
    messages: Annotated[list, add_messages]


def tools_router(state: MascotState):
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return END


def mascot(state: MascotState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


tool_node = ToolNode(tools=tools)

graph = StateGraph(MascotState)

graph.add_node("mascot", mascot)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("mascot")

graph.add_conditional_edges("mascot", tools_router)
graph.add_edge("tool_node", "mascot")

app = graph.compile()

while True:
    user_input = input("User: ")
    if user_input in ["exit", "end"]:
        break
    else:
        result = app.invoke({"messages": [HumanMessage(content=user_input)]})

        print(result)
