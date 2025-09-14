# 🤖 Reflective Self-Correcting Code Generator

A sophisticated AI-powered code generation system that uses iterative error correction and reflection to produce working code solutions. Built with Streamlit, LangGraph, and OpenAI GPT-4o.

![Uploading image.png…]()


## 🌟 Features

- **Intelligent Code Generation**: Uses GPT-4o to generate code based on natural language descriptions
- **Self-Correcting Mechanism**: Automatically detects and fixes errors through iterative refinement
- **Real-time Progress Tracking**: Visual progress indicators and execution logs
- **Interactive Web Interface**: Beautiful Streamlit-based UI with comprehensive controls
- **Configurable Parameters**: Adjustable temperature, max attempts, and model settings
- **Code Execution Testing**: Validates generated code by actually running it
- **Download Capability**: Export generated code as Python files
- **Workflow Visualization**: Visual representation of the agent's decision-making process

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API Key

### Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   Or enter it directly in the Streamlit app interface.

### Running the Application

1. **Streamlit Web App** (Recommended):
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Command Line Interface**:
   ```bash
   python main.py
   ```

## 🎯 How It Works

The system follows a sophisticated workflow:

1. **Code Generation**: AI generates code solution with imports and implementation
2. **Execution Testing**: Code is executed and tested for errors
3. **Error Analysis**: Any errors are captured and analyzed
4. **Reflection & Correction**: AI reflects on errors and generates improved solution
5. **Iteration**: Process continues until success or max attempts reached

## 🖥️ Streamlit Interface

The web interface provides:

- **Input Section**: Natural language code requests with example prompts
- **Configuration Panel**: API key, max attempts, temperature settings
- **Real-time Monitoring**: Progress bars, status updates, and execution logs
- **Results Display**: Formatted code output with syntax highlighting
- **Metrics Dashboard**: Success rates, attempt counts, and performance stats
- **Workflow Visualization**: Interactive diagram of the agent's process

## 📁 Project Structure

```
project_1_self_code_correction/
├── main.py                 # Original implementation with test code
├── agent_core.py          # Refactored agent core for Streamlit integration
├── streamlit_app.py       # Main Streamlit web application
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Configuration

### Model Settings
- **Temperature**: Controls randomness in code generation (0.0 = deterministic, 1.0 = creative)
- **Max Attempts**: Maximum number of retry attempts (1-10)
- **Model**: Currently uses GPT-4o (configurable in code)

### API Configuration
- Set `OPENAI_API_KEY` environment variable
- Or enter API key directly in the Streamlit interface

## 📊 Example Usage

1. **Start the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Enter your API key** in the sidebar

3. **Choose a prompt** or enter a custom request:
   - "Write a function to calculate the factorial of a number"
   - "Create a data visualization using matplotlib"
   - "Implement a binary search algorithm"

4. **Configure settings** (optional):
   - Adjust max attempts
   - Set temperature
   - Choose model parameters

5. **Click "Generate Code"** and watch the AI work!

6. **Review results**:
   - View the generated code
   - Check execution logs
   - Download the code file

## 🛠️ Technical Details

### Architecture
- **LangGraph**: State management and workflow orchestration
- **LangChain**: LLM integration and structured output
- **Streamlit**: Web interface and real-time updates
- **Pydantic**: Data validation and schema definition
- **Plotly**: Interactive visualizations

### Error Handling
- Import validation
- Code execution testing
- Graceful error recovery
- User-friendly error messages

### Performance
- Configurable retry limits
- Progress tracking
- Memory-efficient streaming
- Real-time updates

## 🔍 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your OpenAI API key is valid and has sufficient credits
2. **Import Errors**: The system will automatically retry with corrected imports
3. **Execution Errors**: The AI will reflect on errors and generate improved solutions
4. **Memory Issues**: Reduce max attempts or restart the application

### Debug Mode
Enable verbose logging by setting `verbose=True` in the agent calls.

## 🤝 Contributing

Feel free to contribute by:
- Adding new features
- Improving error handling
- Enhancing the UI
- Adding more example prompts
- Optimizing performance

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- OpenAI for GPT-4o
- LangChain team for the excellent framework
- Streamlit for the amazing web app framework
- The open-source community for inspiration and tools

---

**Happy Coding! 🚀**
