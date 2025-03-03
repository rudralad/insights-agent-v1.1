from typing import List, Dict, Any
import os
import requests
import asyncio
from bs4 import BeautifulSoup
from tavily import Client as TavilyClient
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import (
    BeautifulSoupTransformer,
    Html2TextTransformer,
)
import tempfile
import urllib.parse
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import sys

# Load environment variables
load_dotenv()

# Global settings
# Set to False to disable Firecrawl as a fallback scraper
USE_FIRECRAWL = False  # Default: disabled

# Initialize API clients - Tavily
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(tavily_api_key)

# Initialize Firecrawl client if API key is available and enabled
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
firecrawl_client = (
    FirecrawlApp(api_key=firecrawl_api_key)
    if firecrawl_api_key and USE_FIRECRAWL
    else None
)

# Initialize OpenAI embeddings
openai_embeddings = None


def get_embedding_model():
    """
    Lazy-load the OpenAI embedding model when needed
    """
    global openai_embeddings
    if openai_embeddings is None:
        print("Initializing OpenAI embeddings...")
        openai_embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        print("OpenAI embeddings initialized")
    return openai_embeddings


# Function to check if a URL points to a PDF file
def is_pdf_url(url: str) -> bool:
    """
    Check if the URL points to a PDF file.

    Args:
        url: The URL to check

    Returns:
        True if the URL points to a PDF, False otherwise
    """
    # Check file extension
    if url.lower().endswith(".pdf"):
        return True

    # Check URL path
    parsed_url = urllib.parse.urlparse(url)
    path = parsed_url.path.lower()
    if path.endswith(".pdf"):
        return True

    # Check content type (send a HEAD request to avoid downloading the entire file)
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.head(url, headers=headers, timeout=5)
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type:
            return True
    except Exception:
        # If the HEAD request fails, don't assume it's a PDF
        pass

    return False


# Function to download a PDF file to a temporary location
def download_pdf(url: str) -> str:
    """
    Download a PDF file from a URL to a temporary file.

    Args:
        url: The URL of the PDF file

    Returns:
        Path to the downloaded temporary file
    """
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_path = temp_file.name
        temp_file.close()

        # Download the PDF
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        # Write to the temporary file
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return temp_path
    except Exception as e:
        print(f"Error downloading PDF from {url}: {e}")
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return None


# Function to extract text from a PDF file using LangChain's document loaders
def extract_text_from_pdf(file_path: str) -> Dict[str, Any]:
    """
    Extract text from a PDF file using LangChain document loaders.

    Args:
        file_path: Path to the PDF file

    Returns:
        Dictionary containing the extracted text and metadata
    """
    try:
        # Try PyMuPDFLoader first (faster and more reliable for text-based PDFs)
        print(f"Extracting text from PDF using PyMuPDFLoader: {file_path}")
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()

        # If PyMuPDFLoader didn't extract enough text, try UnstructuredPDFLoader
        if not documents or sum(len(doc.page_content) for doc in documents) < 500:
            print(
                f"PyMuPDFLoader extracted insufficient text, trying UnstructuredPDFLoader"
            )
            try:
                loader = UnstructuredPDFLoader(
                    file_path,
                    strategy="hi_res",  # Use high resolution strategy for better extraction
                    mode="elements",  # Extract by elements to preserve structure
                )
                documents = loader.load()
            except Exception as unstruct_error:
                print(f"Error with UnstructuredPDFLoader: {unstruct_error}")

        # Combine all document pages into one text
        if documents:
            title = ""
            combined_text = ""

            # Extract title from the first page or metadata
            for doc in documents:
                if hasattr(doc, "metadata") and doc.metadata.get("title"):
                    title = doc.metadata.get("title")
                    break

            # If no title in metadata, try to extract from first page content
            if not title and documents[0].page_content:
                # Try to find a title in the first few lines
                lines = documents[0].page_content.split("\n")
                for line in lines[:10]:
                    if len(line.strip()) > 15 and len(line.strip()) < 100:
                        title = line.strip()
                        break

            # If still no title, use filename
            if not title:
                title = os.path.basename(file_path)

            # Combine all page content
            combined_text = "\n\n".join([doc.page_content for doc in documents])

            return {
                "text": combined_text,
                "title": title,
                "num_pages": len(documents),
                "metadata": (
                    documents[0].metadata
                    if documents and hasattr(documents[0], "metadata")
                    else {}
                ),
            }
        else:
            print(f"No text extracted from PDF: {file_path}")
            return None

    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
        return None
    finally:
        # Clean up the temporary file
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception:
            pass


# Function to create embeddings for document content
def create_embeddings(text: str) -> List[float]:
    """
    Create embeddings for the given text using OpenAI embeddings.

    Args:
        text: The text to embed

    Returns:
        List of embedding values
    """
    if not text:
        return None

    try:
        # Get the embedding model (lazy-loaded)
        model = get_embedding_model()

        # Generate embeddings with OpenAI
        embedding = model.embed_query(text)
        return embedding
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return None


def generate_search_queries(original_query: str, num_queries: int = 3) -> List[str]:
    """
    Generate multiple search queries from the original query to get diverse results.

    Args:
        original_query: The user's original query
        num_queries: Number of different search queries to generate

    Returns:
        List of search queries
    """
    # This will be implemented with LLM in main.py
    pass


def search_tavily(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search for information using Tavily API.

    Args:
        query: The search query
        max_results: Maximum number of results to return

    Returns:
        List of search results with URLs and snippets
    """
    try:
        response = tavily_client.search(
            query, search_depth="basic", max_results=max_results
        )
        results = []
        for result in response.get("results", []):
            results.append(
                {
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                }
            )
        return results
    except Exception as e:
        print(f"Error searching with Tavily: {e}")
        return []


def scrape_content(urls: List[str]) -> List[Dict[str, Any]]:
    """
    Scrape content from URLs using a tiered approach:
    1. First check if URL is a PDF and handle it separately using document loaders
    2. Try LangChain AsyncChromiumLoader + BeautifulSoupTransformer (best for JS-rendered pages)
    3. Fall back to direct BeautifulSoup (faster, works for static pages)
    4. Last resort: Firecrawl (if available and enabled, best for complex websites but costly)

    Args:
        urls: List of URLs to scrape

    Returns:
        List of dictionaries containing scraped content and metadata
    """
    results = []

    # Separate PDF URLs from regular web page URLs
    pdf_urls = []
    web_urls = []

    for url in urls:
        if is_pdf_url(url):
            pdf_urls.append(url)
        else:
            web_urls.append(url)

    # Process PDF URLs
    if pdf_urls:
        print(f"Processing {len(pdf_urls)} PDF URLs")
        for url in pdf_urls:
            try:
                print(f"Downloading PDF from {url}")
                pdf_path = download_pdf(url)
                if pdf_path:
                    pdf_content = extract_text_from_pdf(pdf_path)
                    if pdf_content and pdf_content["text"]:
                        # Create embeddings for the PDF content
                        embedding = create_embeddings(pdf_content["text"])

                        # Add to results
                        results.append(
                            {
                                "url": url,
                                "title": pdf_content["title"],
                                "content": pdf_content["text"][
                                    :20000
                                ],  # Limit content length
                                "authors": pdf_content["metadata"].get("author", []),
                                "date_published": pdf_content["metadata"].get(
                                    "creation_date"
                                ),
                                "embedding": embedding,
                            }
                        )
                        print(f"Successfully processed PDF: {url}")
                    else:
                        print(f"Failed to extract content from PDF: {url}")
            except Exception as e:
                print(f"Error processing PDF {url}: {e}")

    # Process web URLs using the existing tiered approach
    if web_urls:
        # Track URLs that failed to scrape with the first two methods
        failed_urls = []

        # Method 1: Try LangChain AsyncChromiumLoader + BeautifulSoupTransformer
        try:
            # Check if we're running on Python 3.12, which has known issues with Playwright
            if sys.version_info >= (3, 12):
                print(
                    "Python 3.12+ detected - skipping AsyncChromiumLoader due to compatibility issues"
                )
                raise NotImplementedError(
                    "Python 3.12 is not fully compatible with Playwright"
                )

            print(
                f"Attempting to scrape {len(web_urls)} URLs with LangChain AsyncChromiumLoader"
            )
            # Load HTML with AsyncChromiumLoader
            loader = AsyncChromiumLoader(web_urls)
            html_documents = loader.load()

            if html_documents:
                # Transform with BeautifulSoupTransformer
                bs_transformer = BeautifulSoupTransformer()
                docs_transformed = bs_transformer.transform_documents(
                    html_documents,
                    tags_to_extract=[
                        "p",
                        "h1",
                        "h2",
                        "h3",
                        "h4",
                        "h5",
                        "li",
                        "div",
                        "span",
                        "article",
                    ],
                )

                # Process the transformed documents
                for doc in docs_transformed:
                    url = doc.metadata.get("source", "")
                    # Check if content is sufficient
                    if doc.page_content and len(doc.page_content) > 500:
                        # Extract title from metadata or content
                        title = ""
                        for line in doc.page_content.split("\n")[:10]:
                            if len(line) > 20 and len(line) < 100:
                                title = line.strip()
                                break

                        # Create embeddings for the content
                        embedding = create_embeddings(doc.page_content[:20000])

                        # Extract author information
                        authors = []
                        author_meta = doc.metadata.get("author", [])
                        if isinstance(author_meta, list):
                            authors.extend(author_meta)
                        elif isinstance(author_meta, str):
                            authors.append(author_meta)

                        # Extract date published
                        date_meta = doc.metadata.get("date")
                        date_published = (
                            date_meta if isinstance(date_meta, str) else None
                        )

                        # Add to results
                        results.append(
                            {
                                "url": url,
                                "title": title,
                                "content": doc.page_content[
                                    :20000
                                ],  # Limit content length
                                "authors": authors,
                                "date_published": date_published,
                                "embedding": embedding,
                            }
                        )
                        print(f"Successfully processed URL with LangChain: {url}")
                    else:
                        failed_urls.append(url)
                        print(f"Insufficient content from LangChain for {url}")
            else:
                failed_urls = web_urls.copy()
                print("No documents returned from AsyncChromiumLoader")
        except (NotImplementedError, ImportError) as e:
            # This will catch both the Python 3.12 issue and missing dependencies
            failed_urls = web_urls.copy()
            print(f"AsyncChromiumLoader not available or compatible: {e}")
            print("Falling back to BeautifulSoup directly.")
        except Exception as lc_error:
            failed_urls = web_urls.copy()
            print(f"Error using LangChain scraping: {lc_error}")

        # Method 2: Fall back to direct BeautifulSoup for failed URLs
        if failed_urls:
            print(f"Falling back to BeautifulSoup for {len(failed_urls)} URLs")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            still_failed_urls = []
            for url in failed_urls:
                try:
                    # Send a request to the URL
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()

                    # Parse the HTML content
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Extract title
                    title = soup.title.text.strip() if soup.title else ""

                    # Get the main content - this is a simplified approach
                    paragraphs = soup.find_all("p")
                    content = "\n\n".join([p.get_text().strip() for p in paragraphs])

                    # Remove very short paragraphs which might be noise
                    content_paragraphs = [
                        p for p in content.split("\n\n") if len(p) > 50
                    ]
                    content = "\n\n".join(content_paragraphs)

                    # Try to find author information
                    authors = []
                    author_meta = soup.find(
                        "meta", attrs={"name": "author"}
                    ) or soup.find("meta", attrs={"property": "article:author"})
                    if author_meta and author_meta.get("content"):
                        authors.append(author_meta.get("content"))

                    # Try to find date published
                    date_meta = soup.find("meta", attrs={"name": "date"}) or soup.find(
                        "meta", attrs={"property": "article:published_time"}
                    )
                    date_published = date_meta.get("content") if date_meta else None

                    # Check if BeautifulSoup managed to extract meaningful content
                    if content and len(content) > 200:
                        # Create embeddings for the content
                        embedding = create_embeddings(content[:20000])

                        # Add to results
                        results.append(
                            {
                                "url": url,
                                "title": title,
                                "content": content[:20000],  # Limit content length
                                "authors": authors,
                                "date_published": date_published,
                                "embedding": embedding,
                            }
                        )
                        print(f"Successfully processed URL with BeautifulSoup: {url}")
                    else:
                        still_failed_urls.append(url)
                        print(f"Insufficient content from BeautifulSoup for {url}")
                except Exception as bs_error:
                    still_failed_urls.append(url)
                    print(f"BeautifulSoup error scraping {url}: {bs_error}")

            # Method 3: Last resort - Try Firecrawl for URLs that still failed (if enabled)
            if still_failed_urls and firecrawl_client and USE_FIRECRAWL:
                print(f"Falling back to Firecrawl for {len(still_failed_urls)} URLs")
                for url in still_failed_urls:
                    try:
                        print(f"Attempting to scrape {url} with Firecrawl")
                        firecrawl_result = firecrawl_client.scrape_url(
                            url, params={"formats": ["markdown"]}
                        )

                        # Extract content from Firecrawl result
                        if firecrawl_result and "markdown" in firecrawl_result:
                            markdown_content = firecrawl_result["markdown"]
                            metadata = firecrawl_result.get("metadata", {})

                            # Create embeddings for the content
                            embedding = create_embeddings(markdown_content[:20000])

                            # Extract author information
                            authors = []
                            author_meta = metadata.get("author", [])
                            if isinstance(author_meta, list):
                                authors.extend(author_meta)
                            elif isinstance(author_meta, str):
                                authors.append(author_meta)

                            # Extract date published
                            date_meta = metadata.get("date")
                            date_published = (
                                date_meta if isinstance(date_meta, str) else None
                            )

                            # Add to results
                            results.append(
                                {
                                    "url": url,
                                    "title": metadata.get("title", ""),
                                    "content": markdown_content[
                                        :20000
                                    ],  # Limit content length
                                    "authors": authors,
                                    "date_published": date_published,
                                    "embedding": embedding,
                                }
                            )
                            print(f"Successfully processed URL with Firecrawl: {url}")
                        else:
                            print(f"Firecrawl didn't return usable content for {url}")

                    except Exception as fc_error:
                        print(f"Firecrawl error scraping {url}: {fc_error}")
            elif still_failed_urls and USE_FIRECRAWL:
                print("Firecrawl not available - API key not configured")
            elif still_failed_urls:
                print("Firecrawl is disabled - skipping Firecrawl scraping")

    return results
