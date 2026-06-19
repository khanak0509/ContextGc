"""Multi-turn tool-using LangGraph agent (Ollama) with ContextGC memory."""

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent
from langchain import hub
from langchain.agents import AgentExecutor

from contextgc.integrations.langchain import ContextGCMemory
from contextgc.integrations.middleware import ContextGCMiddleware

AGENT_GOAL = (
    "Research quantum computing applications in healthcare and synthesize "
    "a detailed report using weather, market, news, and calculation tools"
)

SYSTEM_PROMPT = """
You are a research assistant building a detailed report on quantum computing in healthcare.
You have 4 tools:
- get_weather   : check weather of any city
- get_stock_price : check stock price of any company
- search_news   : search latest news on any topic
- calculator    : do any math calculation

Use tools when helpful. Give thorough, detailed answers that accumulate research findings.
Reference prior turns when the user asks follow-up questions.
"""


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    weather_data = {
        "mumbai": "Mumbai: 32°C, humid, partly cloudy",
        "delhi": "Delhi: 38°C, hot, clear sky",
        "boston": "Boston: 18°C, cool, overcast — good lab conditions",
        "zurich": "Zurich: 14°C, mild, rainy — CERN region",
    }
    return weather_data.get(city.lower(), f"{city}: 22°C, mild (default)")


@tool
def get_stock_price(company: str) -> str:
    """Get the current stock price of a company."""
    stocks = {
        "ibm": "IBM: $189 (+1.1%) — quantum division active",
        "google": "GOOGL: $178 (+0.4%) — Willow chip news",
        "ionq": "IONQ: $42 (+2.3%) — quantum hardware",
        "microsoft": "MSFT: $425 (+0.6%) — Azure Quantum",
    }
    return stocks.get(company.lower(), f"{company}: $100 (dummy price)")


@tool
def search_news(topic: str) -> str:
    """Search for latest news about a topic."""
    return (
        f"Top 3 news about '{topic}':\n"
        f"1. {topic} clinical trial pipeline expands — Nature Medicine\n"
        f"2. Hospital consortium pilots {topic} diagnostics — Reuters\n"
        f"3. FDA guidance draft on {topic} algorithms — STAT News"
    )


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        allowed = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
        }
        result = eval(expression, allowed)
        return f"Result of '{expression}' = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"


def build_agent(contextgc_memory: ContextGCMemory | None = None):
    """Build a ReAct LangChain agent, optionally with ContextGC memory."""
    llm = ChatOllama(model="llama3.2:3b", temperature=0)
    tools = [get_weather, get_stock_price, search_news, calculator]

    try:
        prompt = hub.pull("hwchase17/react")
    except Exception:
        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate.from_template(
            "Answer the following question as best you can. "
            "You have access to these tools: {tools}\n\n"
            "Use this format:\n"
            "Thought: you should always think about what to do\n"
            "Action: the action to take, should be one of [{tool_names}]\n"
            "Action Input: the input to the action\n"
            "Observation: the result of the action\n"
            "... (repeat as needed)\n"
            "Final Answer: the final answer\n\n"
            "Question: {input}\n"
            "{agent_scratchpad}"
        )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    memory_kwargs = {}
    if contextgc_memory is not None:
        memory_kwargs["memory"] = contextgc_memory

    return AgentExecutor(agent=agent, tools=tools, verbose=True, **memory_kwargs)
