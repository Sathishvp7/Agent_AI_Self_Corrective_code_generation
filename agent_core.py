# Agent Core Module - Refactored for Streamlit Integration
"""
Refactored version of the main.py agent code for better Streamlit integration.
This module contains the core agent functionality without the execution code.
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# Initialize GPT-4o
def initialize_llm(temperature=0.0):
    """Initialize the language model with configurable temperature"""
    return ChatOpenAI(model_name="gpt-4o", temperature=temperature)

# Prompt
CODE_GEN_SYS_PROMPT = [
    (
        "system",
        """You are a coding assistant.
            Ensure any code you provide can be executed with all required imports and variables defined.
            Make sure point 3 below has some code to run and execute any code or functions which you define

            Structure your answer as follows:
              1) a prefix describing the code solution
              2) the imports (if no imports needed keep it empty string)
              3) the functioning code blocks

            Here is the user question:""",
    )
]

# Data model
class Code(BaseModel):
    """Schema for code solutions to questions about coding."""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Just the import statements of the code")
    code: str = Field(description="Code blocks not including import statements")

# Langgraph - Create Agent State Schema
class CodeGenState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error_flag : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        code_solution : Code solution
        attempts : Number of tries
    """
    error_flag: str
    messages: Annotated[List[AnyMessage], add_messages]
    code_solution: Code
    attempts: int

## Node 1: Generate Code
def generate_code(state: CodeGenState) -> CodeGenState:
    """Generate code solution from GPT-4o, structured as prefix/imports/code."""
    print("--- GENERATING CODE SOLUTION ---")
    msgs = state["messages"]
    attempts_so_far = state["attempts"]

    # Get the LLM from the state or use default
    llm = state.get("llm", initialize_llm())
    code_generator = llm.with_structured_output(Code)

    # Call code_generation_chain
    code_soln = code_generator.invoke(CODE_GEN_SYS_PROMPT + msgs)

    # We'll record the chain's answer as a new assistant message in conversation.
    new_msg_content = (f"Here is my solution attempt:\n\nDescription: {code_soln.prefix}\n\n"
                       f"Imports: {code_soln.imports}\n\n"
                       f"Code:\n{code_soln.code}")

    msgs.append(("assistant", new_msg_content))
    attempts_so_far += 1

    return {
        "messages": msgs,
        "code_solution": code_soln,
        "attempts": attempts_so_far
    }

## Node 2: Check Code
"""We try to exec the imports, then exec the code. If errors occur, we pass them back to the conversation. Otherwise, success."""

def check_code_execution(state: CodeGenState) -> CodeGenState:
    print("--- CHECKING CODE EXECUTION ---")
    msgs = state["messages"]
    code_soln = state["code_solution"]
    imports_str = code_soln.imports
    code_str = code_soln.code
    attempts = state["attempts"]

    # Attempt to import:
    try:
        exec(imports_str)
    except Exception as e:
        # Import failed
        print("---CODE IMPORT CHECK: FAILED---")
        error_msg = f"""Import test failed!
                        Here is the exception trace details:
                        {e}.

                        Please fix the import section."""

        msgs.append(("user", error_msg))
        return {
            "code_solution": code_soln,
            "attempts": attempts,
            "messages": msgs,
            "error_flag": "yes"
        }

    # Attempt to run code:
    try:
        scope = {}
        exec(f"{imports_str}\n{code_str}", scope)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        error_msg = f"""Your code solution failed the code execution test!
                            Here is the exception trace details:
                            {e}

                            Reflect on this error and your prior attempt to solve the problem.

                            (1) State what you think went wrong with the prior solution
                            (2) try to solve this problem again.

                            Return the FULL SOLUTION.

                            Use the code tool to structure the output with a prefix, imports, and code block."""

        msgs.append(("user", error_msg))
        return {
            "code_solution": code_soln,
            "attempts": attempts,
            "messages": msgs,
            "error_flag": "yes"
        }

    # If no errors:
    print("--- NO ERRORS FOUND ---")
    return {
            "code_solution": code_soln,
            "attempts": attempts,
            "messages": msgs,
            "error_flag": "no"
    }

# Conditional Routing to Decide Next Step
""" If Pass give response to use else run in the loop until Max execution reach"""
MAX_ATTEMPTS = 3

def decide_next(state: CodeGenState) -> str:
    """If error or attempts < MAX_ATTEMPTS => go generate. Else end."""
    err = state["error_flag"]
    attempts = state["attempts"]
    max_attempts = state.get("max_attempts", MAX_ATTEMPTS)
    
    if err == "no" or attempts >= max_attempts:
        print("--- DECISION: FINISH ---")
        return "__end__"
    else:
        print("--- DECISION: RETRY ---")
        return "generate_code"

def create_agent(temperature=0.0, max_attempts=3):
    """Create and compile the agent with configurable parameters"""
    
    # Initialize LLM
    llm = initialize_llm(temperature)
    
    # Build the Reflection Agentic Graph
    graph = StateGraph(CodeGenState)

    # Add nodes:
    graph.add_node("generate_code", generate_code)
    graph.add_node("check_code", check_code_execution)

    # Edges:
    graph.set_entry_point("generate_code")
    graph.add_edge("generate_code", "check_code")
    graph.add_conditional_edges(
        "check_code",
        decide_next,
        [END, "generate_code"]
    )

    # Compile the graph
    agent = graph.compile()
    
    return agent

def call_reflection_coding_agent(agent, prompt, max_attempts=3, temperature=0.0, verbose=False):
    """Call the reflection coding agent with configurable parameters"""
    
    # Initialize state with LLM and max_attempts
    initial_state = {
        "messages": [HumanMessage(content=prompt)], 
        "attempts": 0,
        "error_flag": "unknown",
        "code_solution": None,
        "llm": initialize_llm(temperature),
        "max_attempts": max_attempts
    }
    
    events = agent.stream(initial_state, stream_mode="values")

    print('Running Agent. Please wait...')
    final_event = None
    
    for event in events:
        final_event = event
        if verbose:
            print(f"Attempt {event.get('attempts', 0)}: {event.get('error_flag', 'unknown')}")
            if event.get("code_solution"):
                print(f"Description: {event['code_solution'].prefix}")
                print(f"Code: {event['code_solution'].code[:100]}...")

    if final_event and final_event.get("code_solution"):
        print('\n\nFinal Solution:')
        print("\nDescription:\n" + final_event["code_solution"].prefix +
              "\nCode:\n"+final_event["code_solution"].imports + '\n\n' + final_event["code_solution"].code)
    
    return final_event

# Create default agent instance
coder_agent = create_agent()
