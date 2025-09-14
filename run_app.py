#!/usr/bin/env python3
"""
Launcher script for the Reflective Self-Correcting Code Generator Streamlit app.
This script provides an easy way to start the application with proper configuration.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import langchain
        import langchain_openai
        import langgraph
        import pandas
        import plotly
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_api_key():
    """Check if OpenAI API key is set"""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✅ OpenAI API key found in environment")
        return True
    else:
        print("⚠️  OpenAI API key not found in environment")
        print("You can set it with: export OPENAI_API_KEY='your-key-here'")
        print("Or enter it directly in the Streamlit app interface")
        return False

def run_streamlit_app(port=8501, host="localhost"):
    """Run the Streamlit app"""
    app_path = Path(__file__).parent / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"❌ Streamlit app not found at {app_path}")
        return False
    
    print(f"🚀 Starting Streamlit app on http://{host}:{port}")
    print("Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", str(port),
            "--server.address", host,
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit app: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Launch the Reflective Self-Correcting Code Generator"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501, 
        help="Port to run the Streamlit app on (default: 8501)"
    )
    parser.add_argument(
        "--host", 
        default="localhost", 
        help="Host to run the Streamlit app on (default: localhost)"
    )
    parser.add_argument(
        "--skip-checks", 
        action="store_true", 
        help="Skip dependency and API key checks"
    )
    
    args = parser.parse_args()
    
    print("🤖 Reflective Self-Correcting Code Generator")
    print("=" * 50)
    
    if not args.skip_checks:
        print("\n🔍 Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        
        print("\n🔑 Checking API key...")
        check_api_key()
    
    print("\n🚀 Launching application...")
    success = run_streamlit_app(port=args.port, host=args.host)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
