import os
import subprocess
import sys


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import langchain
        import openai
        import tavily
        import requests
        from bs4 import BeautifulSoup
        from dotenv import load_dotenv
        import markdown
        import fpdf

        print("✅ All required packages are installed.")

        # Check Python version
        if sys.version_info >= (3, 12):
            print(
                "⚠️ Python 3.12+ detected - Playwright/AsyncChromiumLoader may not work correctly."
            )
            print(
                "   The application will fall back to BeautifulSoup for web scraping."
            )

        # Check for web scraping dependencies - all are optional with fallbacks
        try:
            from langchain_community.document_loaders import AsyncChromiumLoader
            from langchain_community.document_transformers import (
                BeautifulSoupTransformer,
            )
            import playwright

            if sys.version_info < (3, 12):
                print(
                    "✅ LangChain web scraping dependencies are installed (primary scraper)."
                )
            else:
                print(
                    "⚠️ LangChain web scraping dependencies are installed but may not work with Python 3.12+."
                )
        except ImportError:
            print(
                "⚠️ LangChain web scraping dependencies are not installed. Will fall back to BeautifulSoup."
            )

        try:
            from bs4 import BeautifulSoup

            print("✅ BeautifulSoup is installed (secondary scraper).")
        except ImportError:
            print("⚠️ BeautifulSoup is not installed. Web scraping will be limited.")

        # Check for local embedding dependencies
        try:
            from langchain_openai import OpenAIEmbeddings

            print("✅ OpenAI embeddings are available for semantic search.")
            print(
                "   Using OpenAI embeddings for efficient document similarity and ranking."
            )
        except ImportError:
            print(
                "⚠️ OpenAI embeddings are not installed. Semantic search will be limited."
            )

        # Check if firecrawl is installed (optional)
        try:
            import firecrawl

            print(
                "✅ Firecrawl is installed (optional scraper, currently disabled by default)."
            )
        except ImportError:
            print(
                "⚠️ Firecrawl is not installed. This is an optional dependency for future use."
            )

        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False


def check_api_keys():
    """Check if API keys are set in .env file."""
    from dotenv import load_dotenv

    load_dotenv()

    # Required API keys
    missing_keys = []
    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    if not os.getenv("TAVILY_API_KEY"):
        missing_keys.append("TAVILY_API_KEY")

    if missing_keys:
        print(
            f"⚠️ The following API keys are missing in the .env file: {', '.join(missing_keys)}"
        )
        print("   The app will still run but some functionality may be limited.")
        return False

    # Optional API keys
    if not os.getenv("FIRECRAWL_API_KEY"):
        print(
            "ℹ️ FIRECRAWL_API_KEY is not set. This is optional as Firecrawl is disabled by default."
        )
    else:
        print(
            "✅ Firecrawl API key is set (Firecrawl is disabled by default but can be enabled in search_utils.py)."
        )

    print("✅ All required API keys are set.")
    print(
        "ℹ️ Using local embeddings - no additional API keys required for semantic search."
    )
    return True


def launch_app():
    """Launch the Streamlit app."""
    print("🏥 Starting Insights Agent...")
    # Disable file watcher to avoid PyTorch compatibility issues
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.fileWatcherType",
            "none",
        ]
    )


if __name__ == "__main__":
    print("=" * 50)
    print("Insights Agent Launcher")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        print("\n⚠️ Installing missing dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )

        # Install playwright drivers after playwright is installed
        try:
            import playwright

            print("\n⚠️ Installing Playwright drivers...")
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"])
        except ImportError:
            print(
                "\n⚠️ Playwright not installed correctly. Some web scraping features may not work."
            )

    # Check API keys
    check_api_keys()

    # Launch the app
    launch_app()
