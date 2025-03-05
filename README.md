# Insights Agent

A specialized research agent for physiotherapy and rehabilitation topics. This tool helps researchers, students, and healthcare professionals find, analyze, and synthesize the latest information in the field.

## Features

- **Smart Query Generation**: Automatically generates multiple search queries to find comprehensive information
- **Advanced Web Search**: Uses Tavily Search API for accurate and relevant results
- **Robust Content Scraping**: Uses BeautifulSoup for most websites, with Firecrawl as a fallback for complex sites
- **PDF Processing**: Automatically detects and processes PDF documents for academic research
- **OpenAI-powered Semantic Search**: Uses OpenAI embeddings with in-memory vector storage for efficient content ranking
- **Intelligent Analysis**: Processes information to extract key insights, methodologies, and evidence
- **Comprehensive Reports**: Generates structured reports with citations and download capability
- **User-Friendly Interface**: Simple Streamlit UI with start/stop controls for better research experience

## Installation

1. Clone this repository
   ```
   git clone https://github.com/rudralad/insights-agent-v1.git
   cd insights-agent-v1
   ```
2. Create a virtual environment:
   ```
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
5. Install Playwright browser (for web scraping):
   ```
   playwright install chromium
   ```
6. Set up your configuration files:
   - Copy `.env.example` to `.env` and add your API keys and Airtable configuration:
     ```
     cp .env.example .env
     ```
   - Edit the `.env` file to add your OpenAI API key, Tavily API key, and Airtable configuration.

## API Keys

This application requires the following API keys:

- **LLM Provider API Key**: Based on your chosen provider:
  - **OpenAI API Key**: If using OpenAI models (default)
  - **Groq API Key**: If using Groq models
  - **Gemini API Key**: If using Google Gemini models
- **OpenAI API Key**: Always required for embeddings functionality
- **Tavily API Key**: For web search capabilities
- **Firecrawl API Key** (optional): For enhanced scraping of complex websites (disabled by default)
- **Airtable API Key** (optional): For storing research data and analytics

## Configuration

The application supports multiple LLM providers, which can be configured in the `.env` file:

```
# LLM Provider Configuration
# Options: openai, groq, gemini
LLM_PROVIDER=openai

# LLM Model Selection
# For OpenAI: o1-mini, gpt-3.5-turbo, gpt-4, etc.
# For Groq: llama3-70b-8192, mixtral-8x7b-32768, etc.
# For Gemini: gemini-1.5-pro, gemini-1.5-flash, etc.
LLM_MODEL=o1-mini
```

## Usage

1. Ensure your virtual environment is activated
2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
   Or use the launcher:
   ```
   python launch.py
   ```
3. Enter your research query in the text input field
4. Click the "Start Research" button to begin processing
5. The agent will process your query and generate a comprehensive report
6. You can stop the research process at any time using the "Stop" button
7. You can download the report as a PDF file

## Project Structure

- `app.py`: Main Streamlit application
- `search_utils.py`: Functions for search and web scraping (LangChain, BeautifulSoup & Firecrawl)
- `llm_utils.py`: Functions for LLM interactions, embeddings, and report generation
- `airtable_utils.py`: Functions for Airtable integration
- `requirements.txt`: List of required Python packages
- `.env`: Configuration file for API keys and Airtable settings (not included in repository)

## Technical Features

- **OpenAI Embeddings**: Uses OpenAI's text-embedding-3-small model for high-quality semantic representations
- **Vector Storage**: Implements in-memory vector storage for similarity search and content ranking
- **Tiered Scraping**: Implements a fallback system from LangChain to BeautifulSoup to ensure robust content extraction
- **PDF Processing**: Automatically detects and extracts content from PDF documents using PyMuPDF and Unstructured
- **Smart Content Ranking**: Prioritizes content based on semantic relevance to the research query

## Troubleshooting

If you encounter any API-related errors:

- Make sure you have valid API keys with sufficient credits
- Ensure your API keys are correctly set in the .env file
- If embedding generation fails, check your OpenAI API usage/quota

### Python 3.12 Compatibility Note

The application will attempt to use Playwright-based web scraping (AsyncChromiumLoader) regardless of Python version. While there are known compatibility issues between Python 3.12 and Playwright's asyncio implementation, the code will try to use it and automatically fall back to BeautifulSoup if AsyncChromiumLoader fails.

If you encounter issues with web scraping:

- The application will automatically fall back to using BeautifulSoup for web scraping, which works reliably across all Python versions
- For most reliable results with AsyncChromiumLoader, consider using Python 3.11 instead of 3.12
- Future updates to Playwright may resolve the compatibility issues with Python 3.12

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- Powered by LangChain, OpenAI, Tavily Search, and BeautifulSoup
- Built with Streamlit and OpenAI embeddings
- Developed by Ability Labs
