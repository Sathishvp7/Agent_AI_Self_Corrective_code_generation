import streamlit as st
import os
import sys
import time
import json
from typing import Dict, List, Any
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import traceback

# Add the current directory to Python path to import main module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the agent from agent_core.py
from agent_core import coder_agent, call_reflection_coding_agent, create_agent, CodeGenState
from langchain_core.messages import HumanMessage

# Page configuration
st.set_page_config(
    page_title="Reflective Self-Correcting Code Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .code-block {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 0.375rem;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
    }
    .workflow-step {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 0.375rem 0.375rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'execution_logs' not in st.session_state:
        st.session_state.execution_logs = []
    if 'current_attempt' not in st.session_state:
        st.session_state.current_attempt = 0
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'final_solution' not in st.session_state:
        st.session_state.final_solution = None
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0

def display_workflow_diagram():
    """Display the agent workflow diagram"""
    st.markdown("### 🔄 Agent Workflow")
    
    workflow_data = {
        'Step': ['1. Generate Code', '2. Execute & Test', '3. Error Analysis', '4. Reflection & Correction', '5. Success/Retry'],
        'Status': ['Ready', 'Ready', 'Ready', 'Ready', 'Ready'],
        'Description': [
            'AI generates code solution with imports and implementation',
            'Code is executed and tested for errors',
            'Any errors are captured and analyzed',
            'AI reflects on errors and generates improved solution',
            'Process continues until success or max attempts reached'
        ]
    }
    
    df_workflow = pd.DataFrame(workflow_data)
    
    # Create a visual workflow
    fig = go.Figure()
    
    # Add workflow steps as boxes
    for i, step in enumerate(df_workflow['Step']):
        fig.add_trace(go.Scatter(
            x=[i],
            y=[0],
            mode='markers+text',
            marker=dict(size=50, color='lightblue', symbol='square'),
            text=[step],
            textposition='middle center',
            name=step,
            hovertemplate=f"<b>{step}</b><br>{df_workflow['Description'].iloc[i]}<extra></extra>"
        ))
    
    # Add arrows between steps
    for i in range(len(df_workflow) - 1):
        fig.add_annotation(
            x=i + 0.5,
            y=0,
            ax=i + 0.3,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="gray"
        )
    
    fig.update_layout(
        title="Self-Correcting Code Generation Workflow",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_metrics():
    """Display execution metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Attempts",
            value=st.session_state.current_attempt,
            delta=None
        )
    
    with col2:
        st.metric(
            label="Errors Encountered",
            value=st.session_state.error_count,
            delta=None
        )
    
    with col3:
        success_rate = 0
        if st.session_state.current_attempt > 0:
            success_rate = ((st.session_state.current_attempt - st.session_state.error_count) / st.session_state.current_attempt) * 100
        st.metric(
            label="Success Rate",
            value=f"{success_rate:.1f}%",
            delta=None
        )
    
    with col4:
        status = "🟢 Ready" if not st.session_state.is_running else "🟡 Running"
        st.metric(
            label="Status",
            value=status,
            delta=None
        )

def run_agent_with_progress(prompt: str, max_attempts: int = 3, temperature: float = 0.0):
    """Run the agent with progress tracking"""
    st.session_state.is_running = True
    st.session_state.current_attempt = 0
    st.session_state.error_count = 0
    st.session_state.execution_logs = []
    st.session_state.final_solution = None
    
    # Create progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.container()
    
    try:
        # Create agent with configurable parameters
        agent = create_agent(temperature=temperature, max_attempts=max_attempts)
        
        # Initialize the agent state
        initial_state = {
            "messages": [HumanMessage(content=prompt)], 
            "attempts": 0,
            "error_flag": "unknown",
            "code_solution": None,
            "llm": None,  # Will be set by the agent
            "max_attempts": max_attempts
        }
        
        # Stream the agent execution
        events = agent.stream(initial_state, stream_mode="values")
        
        step_count = 0
        total_steps = max_attempts * 2  # Generate + Check for each attempt
        
        for event in events:
            step_count += 1
            progress = min(step_count / total_steps, 1.0)
            progress_bar.progress(progress)
            
            # Update status
            if "generate_code" in str(event.get("messages", [])):
                status_text.text("🤖 Generating code solution...")
            elif "check_code" in str(event.get("messages", [])):
                status_text.text("🔍 Testing code execution...")
            
            # Log the event
            log_entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "step": step_count,
                "event": event,
                "attempts": event.get("attempts", 0),
                "error_flag": event.get("error_flag", "unknown")
            }
            st.session_state.execution_logs.append(log_entry)
            
            # Update session state
            st.session_state.current_attempt = event.get("attempts", 0)
            
            if event.get("error_flag") == "yes":
                st.session_state.error_count += 1
            
            # Display real-time logs
            with log_container:
                st.markdown(f"**Step {step_count}** - {log_entry['timestamp']}")
                if event.get("code_solution"):
                    code_soln = event["code_solution"]
                    st.markdown(f"**Description:** {code_soln.prefix}")
                    st.code(code_soln.imports + "\n" + code_soln.code, language="python")
                
                if event.get("error_flag") == "yes":
                    st.error("❌ Error detected - Agent will retry...")
                elif event.get("error_flag") == "no":
                    st.success("✅ Code executed successfully!")
                
                st.markdown("---")
            
            # Check if we should stop
            if event.get("error_flag") == "no" or event.get("attempts", 0) >= max_attempts:
                break
        
        # Final status
        if event.get("error_flag") == "no":
            status_text.text("✅ Code generation completed successfully!")
            st.session_state.final_solution = event.get("code_solution")
        else:
            status_text.text("⚠️ Maximum attempts reached")
        
        progress_bar.progress(1.0)
        
    except Exception as e:
        st.error(f"❌ Error running agent: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
    finally:
        st.session_state.is_running = False

def display_final_solution():
    """Display the final solution"""
    if st.session_state.final_solution:
        st.markdown("### 🎯 Final Solution")
        
        code_soln = st.session_state.final_solution
        
        # Description
        st.markdown("**Description:**")
        st.info(code_soln.prefix)
        
        # Imports
        if code_soln.imports:
            st.markdown("**Imports:**")
            st.code(code_soln.imports, language="python")
        
        # Code
        st.markdown("**Code:**")
        st.code(code_soln.code, language="python")
        
        # Download button
        full_code = f"# {code_soln.prefix}\n\n{code_soln.imports}\n\n{code_soln.code}"
        st.download_button(
            label="📥 Download Code",
            data=full_code,
            file_name=f"generated_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
            mime="text/python"
        )

def display_execution_logs():
    """Display detailed execution logs"""
    if st.session_state.execution_logs:
        st.markdown("### 📋 Execution Logs")
        
        # Create a DataFrame for better display
        logs_data = []
        for log in st.session_state.execution_logs:
            logs_data.append({
                "Timestamp": log["timestamp"],
                "Step": log["step"],
                "Attempts": log["attempts"],
                "Status": "Error" if log["error_flag"] == "yes" else "Success" if log["error_flag"] == "no" else "Processing"
            })
        
        if logs_data:
            df_logs = pd.DataFrame(logs_data)
            st.dataframe(df_logs, use_container_width=True)
            
            # Create a timeline chart
            fig = px.timeline(
                df_logs, 
                x_start="Timestamp", 
                x_end="Timestamp",
                y="Step",
                color="Status",
                title="Execution Timeline",
                color_discrete_map={"Error": "red", "Success": "green", "Processing": "blue"}
            )
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Reflective Self-Correcting Code Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered code generation with iterative error correction and reflection</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use the code generator"
        )
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("✅ API Key set")
        else:
            st.warning("⚠️ Please enter your OpenAI API key")
        
        # Max attempts
        max_attempts = st.slider(
            "Maximum Attempts",
            min_value=1,
            max_value=10,
            value=3,
            help="Maximum number of attempts before stopping"
        )
        
        # Model settings
        st.markdown("### 🧠 Model Settings")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Controls randomness in code generation"
        )
        
        # Clear session button
        if st.button("🗑️ Clear Session", type="secondary"):
            st.session_state.clear()
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input section
        st.markdown("### 💬 Enter Your Code Request")
        
        # Example prompts
        example_prompts = [
            "Write a function to calculate the factorial of a number",
            "Create a data visualization using matplotlib",
            "Implement a binary search algorithm",
            "Write code to read and process a CSV file",
            "Create a simple web scraper using requests and BeautifulSoup"
        ]
        
        selected_example = st.selectbox(
            "Choose an example prompt:",
            ["Custom prompt..."] + example_prompts
        )
        
        if selected_example != "Custom prompt...":
            prompt = selected_example
        else:
            prompt = ""
        
        user_prompt = st.text_area(
            "Describe what code you want to generate:",
            value=prompt,
            height=100,
            placeholder="e.g., Write a function to sort a list of numbers using quicksort algorithm..."
        )
        
        # Generate button
        col_btn1, col_btn2 = st.columns([1, 1])
        
        with col_btn1:
            generate_btn = st.button(
                "🚀 Generate Code",
                type="primary",
                disabled=not user_prompt.strip() or not api_key or st.session_state.is_running
            )
        
        with col_btn2:
            if st.button("📊 View Workflow", disabled=st.session_state.is_running):
                st.session_state.show_workflow = True
    
    with col2:
        # Metrics
        display_metrics()
        
        # Workflow diagram
        if st.session_state.get("show_workflow", False):
            display_workflow_diagram()
            if st.button("❌ Close Workflow"):
                st.session_state.show_workflow = False
                st.rerun()
    
    # Generate code when button is clicked
    if generate_btn and user_prompt.strip() and api_key:
        with st.spinner("🤖 AI Agent is working..."):
            run_agent_with_progress(user_prompt, max_attempts, temperature)
    
    # Display results
    if st.session_state.final_solution:
        st.markdown("---")
        display_final_solution()
    
    # Display execution logs
    if st.session_state.execution_logs:
        st.markdown("---")
        display_execution_logs()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; margin-top: 2rem;'>
            <p>🤖 Reflective Self-Correcting Code Generator | Built with Streamlit & LangGraph</p>
            <p>This AI agent iteratively generates, tests, and refines code until it works correctly.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
