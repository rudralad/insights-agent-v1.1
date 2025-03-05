from typing import List, Dict, Any
import os
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.output_parsers import BaseOutputParser
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import uuid
import sys
import re

# Common model options for each provider
OPENAI_MODELS = ["o1-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
GROQ_MODELS = [
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
    "deepseek-r1-distill-llama-70b",
]
GEMINI_MODELS = [
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-pro",
    "gemini-pro-vision",
]

# Import LLM providers
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

# Load environment variables
load_dotenv()


# Custom output parser that extracts content after think tags.
# Ignores text between <think> and </think> tags, and returns only text after the closing tag.
# Supports variations like </think> and </ think>.
# If these tags are not present, it returns the original content.
class ThinkTagsOutputParser(BaseOutputParser):
    """
    Custom output parser that extracts content after think tags.
    Ignores text between <think> and </think> tags, and returns only text after the closing tag.
    Supports variations like </think> and </ think>.
    If these tags are not present, it returns the original content.
    """

    def parse(self, text):
        # Check for <think>...</think> format
        if "<think>" in text and ("</think>" in text or "</ think>" in text):
            # Find closing tag position - check both possible formats
            closing_tag_pos = -1
            if "</think>" in text:
                closing_tag_pos = text.find("</think>") + len("</think>")
            elif "</ think>" in text:
                closing_tag_pos = text.find("</ think>") + len("</ think>")

            if closing_tag_pos != -1:
                # Extract everything after the closing tag
                content_after_tags = text[closing_tag_pos:].strip()
                print(
                    "Found <think>...</think> tags, extracting content after closing tag"
                )
                return content_after_tags

        # Check for ,think><think/> format (legacy format)
        elif ",think><think/>" in text:
            # Extract everything after the tags
            content_after_tags = text.split(",think><think/", 1)[1].strip()
            print("Found ,think><think/> tags, extracting content after them")
            return content_after_tags

        # No recognized tag pattern found, return original text
        return text


# Initialize LLM based on provider configuration
def initialize_llm():
    """
    Initialize LLM based on provider configuration in environment variables.
    Supports OpenAI, Groq, and Gemini.

    Returns:
        The initialized LLM instance with output parser for think tags
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("LLM_MODEL", "o1-mini")

    print(f"Initializing LLM with provider: {provider}, model: {model}")

    # Initialize the base LLM based on provider
    base_llm = None
    if provider == "openai":
        base_llm = ChatOpenAI(
            model=model, api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7
        )
    elif provider == "groq":
        if ChatGroq is None:
            print(
                "ERROR: Groq integration not available. Install with 'pip install langchain-groq'."
            )
            print("Falling back to OpenAI.")
            base_llm = ChatOpenAI(
                model="o1-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0
            )
        else:
            base_llm = ChatGroq(
                model=model, api_key=os.getenv("GROQ_API_KEY"), temperature=0
            )
    elif provider == "gemini":
        if ChatGoogleGenerativeAI is None:
            print(
                "ERROR: Google Gemini integration not available. Install with 'pip install langchain-google-genai'."
            )
            print("Falling back to OpenAI.")
            base_llm = ChatOpenAI(
                model="o1-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0
            )
        else:
            base_llm = ChatGoogleGenerativeAI(
                model=model, api_key=os.getenv("GEMINI_API_KEY"), temperature=0
            )
    else:
        print(f"WARNING: Unknown LLM provider '{provider}'. Falling back to OpenAI.")
        base_llm = ChatOpenAI(
            model="o1-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0
        )

    # Wrap the LLM with the ThinkTagsOutputParser
    think_tags_parser = ThinkTagsOutputParser()

    # Create and return a chain that processes the output through the parser
    return base_llm | think_tags_parser


# Initialize LLM
llm = initialize_llm()

# Initialize OpenAI embedding model
print("Initializing OpenAI embeddings...")
openai_embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
print("OpenAI embeddings initialized")

# In-memory storage for document embeddings
document_store = {"ids": [], "texts": [], "embeddings": [], "metadatas": []}


def generate_search_queries(query: str, num_queries: int = 5) -> List[str]:
    """
    Generate diverse search queries based on the original query.
    The ThinkTagsOutputParser extracts content after </think> tags if present.

    Args:
        query: The original query from the user
        num_queries: Number of search queries to generate

    Returns:
        List of search queries
    """
    # Prompt for generating diverse search queries
    prompt = PromptTemplate.from_template(
        """You are a search query expansion expert. Your task is to generate {num_queries} diverse search queries 
        based on the original query to maximize the coverage of relevant information.
        
        Original query: {query}
        
        Generate {num_queries} different search queries that would help gather comprehensive information about this topic.
        Each query should focus on different aspects or use different terminology.
        
        If you want to include your thinking process, put it between <think> and </think> tags. Only text after the </think> tag will be used.
        
        Output the queries as a numbered list, one per line, without any additional text.
        """
    )

    # Create chain
    chain = prompt | llm | StrOutputParser()

    # Generate queries
    result = chain.invoke({"query": query, "num_queries": num_queries})

    # Parse the result into a list of queries
    queries = []
    for line in result.strip().split("\n"):
        # Remove numbering and any extra whitespace
        clean_line = line.strip()
        if clean_line:
            # Remove numbering like "1.", "2.", etc.
            if clean_line[0].isdigit() and clean_line[1:].startswith(". "):
                clean_line = clean_line[clean_line.find(". ") + 2 :]
            queries.append(clean_line)

    return queries


def calculate_similarity(query_embedding, document_embedding):
    """
    Calculate cosine similarity between query and document embeddings.

    Args:
        query_embedding: Embedding vector for the query
        document_embedding: Embedding vector for the document

    Returns:
        Similarity score between 0 and 1
    """
    # Reshape embeddings for sklearn's cosine_similarity
    query_embedding = np.array(query_embedding).reshape(1, -1)
    document_embedding = np.array(document_embedding).reshape(1, -1)

    # Calculate cosine similarity
    similarity = cosine_similarity(query_embedding, document_embedding)[0][0]
    return similarity


def embed_text(text: str) -> List[float]:
    """
    Generate embeddings for a text using OpenAI's embedding model.
    Includes error handling and fallback mechanisms.

    Args:
        text: Text to embed

    Returns:
        Embedding vector or None if embedding fails
    """
    if not text or len(text.strip()) == 0:
        print("Warning: Empty text provided for embedding. Returning None.")
        return None

    try:
        # Truncate text if it's too long (OpenAI has token limits)
        # A conservative estimate is 8000 characters ~ 2000 tokens
        if len(text) > 5000:
            print(
                f"Warning: Truncating text from {len(text)} to 5000 characters for embedding."
            )
            text = text[:8000]

        embedding = openai_embeddings.embed_query(text)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # Return None as fallback - the semantic search function will handle this
        return None


def add_to_vector_store(documents: List[Dict[str, Any]]) -> None:
    """
    Add documents to the in-memory vector store.
    Uses pre-computed embeddings when available to avoid duplicate computation.

    Args:
        documents: List of document dictionaries
    """
    reused_embeddings = 0
    generated_embeddings = 0

    for i, doc in enumerate(documents):
        if not doc.get("content"):
            continue

        doc_id = f"doc_{i}_{uuid.uuid4().hex[:8]}"
        document_store["ids"].append(doc_id)

        # Get content for embedding
        content = doc.get("content", "")
        document_store["texts"].append(content)

        # Use pre-computed embedding if available, otherwise generate a new one
        if "embedding" in doc and doc["embedding"] is not None:
            embedding = doc["embedding"]
            reused_embeddings += 1
        else:
            # Generate and store embedding
            embedding = embed_text(content)
            generated_embeddings += 1

        document_store["embeddings"].append(embedding)

        # Store metadata
        metadata = {
            "url": doc.get("url", ""),
            "title": doc.get("title", ""),
            "authors": doc.get("authors", []),
            "date_published": doc.get("date_published", ""),
            "doc_id": i,  # Store original document index
        }
        document_store["metadatas"].append(metadata)

    print(
        f"Vector store updated: {reused_embeddings} pre-computed embeddings reused, {generated_embeddings} new embeddings generated"
    )


def semantic_search(
    query: str, documents: List[Dict[str, Any]], top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Perform semantic search across documents using embeddings and cosine similarity.

    Args:
        query: The search query
        documents: List of documents
        top_k: Number of top results to return

    Returns:
        List of most semantically similar documents
    """
    # Clear previous document store
    document_store["ids"] = []
    document_store["texts"] = []
    document_store["embeddings"] = []
    document_store["metadatas"] = []

    # First, ensure documents are added to the vector store
    add_to_vector_store(documents)

    if not document_store["embeddings"]:
        print(
            "Warning: No embeddings available for semantic search. Falling back to original order."
        )
        return documents[:top_k]  # Fallback to original order

    # Generate embedding for the query
    query_embedding = embed_text(query)
    if query_embedding is None:
        print(
            "Warning: Failed to generate query embedding. Falling back to original order."
        )
        return documents[:top_k]  # Fallback to original order

    # Calculate similarity scores
    similarities = []
    valid_indices = []
    for i, doc_embedding in enumerate(document_store["embeddings"]):
        if doc_embedding is not None:
            similarity = calculate_similarity(query_embedding, doc_embedding)
            similarities.append(similarity)
            valid_indices.append(i)
        else:
            print(
                f"Warning: Document at index {i} has no valid embedding. Skipping in similarity calculation."
            )

    # Get indices of top_k most similar documents
    if not similarities:
        print("Warning: No valid similarity scores. Falling back to original order.")
        return documents[:top_k]

    # Get the top_k indices based on similarity scores
    sorted_indices = np.argsort(similarities)[::-1][:top_k]

    # Map back to the original document indices
    top_indices = [valid_indices[i] for i in sorted_indices]

    # Get the original document indices from metadata
    result_documents = []
    for idx in top_indices:
        doc_idx = document_store["metadatas"][idx]["doc_id"]
        result_documents.append(documents[doc_idx])

    return result_documents


def extract_insights(scraped_content: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract key insights from scraped content using LLM.
    Uses semantic search to rank content by relevance before analysis.
    The ThinkTagsOutputParser extracts content after </think> tags if present.
    Makes multiple LLM calls for large content to handle token limitations.

    Args:
        scraped_content: List of scraped content dictionaries

    Returns:
        Dictionary with extracted insights
    """
    # Use semantic search to rank content by relevance to common research themes
    relevance_query = "key facts, statistics, methodologies, evidence, expert opinions, and recent developments in this field"

    # Rank content by semantic relevance
    ranked_content = semantic_search(
        relevance_query, scraped_content, top_k=len(scraped_content)
    )

    # Check current LLM provider to handle token limits appropriately
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    # Define a max batch size based on the provider
    # Groq has a lower token limit (~6000 tokens), so we use a smaller batch size
    max_chars_per_batch = 12000 if provider == "groq" else 20000

    # Divide sources into batches to avoid token limits
    all_insights = []
    current_batch_text = ""
    current_batch_sources = []
    batch_number = 1

    for i, item in enumerate(ranked_content):
        title = item.get("title", f"Document {i+1}")
        content = item.get("content", "")
        url = item.get("url", "").lower()

        # Add source information with priority indicator
        priority_label = f"[Priority Source {i+1}]" if i < 3 else f"[Source {i+1}]"

        # Add PDF indicator if content appears to be from a PDF
        pdf_indicator = ""
        if "pdf" in url or url.endswith(".pdf"):
            pdf_indicator = "[PDF]"

        if content:
            # Give more content space to higher-ranked sources
            content_length = 3000 if i < 3 else 1500
            truncated_content = content[:content_length]

            # Format the document text
            doc_text = f"\n\n--- {priority_label} {title} {pdf_indicator} ---\n{truncated_content}..."

            # Check if adding this document would exceed the batch size
            if (
                len(current_batch_text) + len(doc_text) > max_chars_per_batch
                and current_batch_text
            ):
                # Process the current batch before starting a new one
                batch_insights = process_content_batch(current_batch_text, batch_number)
                all_insights.append(batch_insights)

                # Start a new batch
                current_batch_text = doc_text
                current_batch_sources = [title]
                batch_number += 1
            else:
                # Add to the current batch
                current_batch_text += doc_text
                current_batch_sources.append(title)

    # Process the final batch if it contains any sources
    if current_batch_text:
        batch_insights = process_content_batch(current_batch_text, batch_number)
        all_insights.append(batch_insights)

    # Combine all insights from different batches
    combined_insights = ""
    if len(all_insights) == 1:
        combined_insights = all_insights[0]
    else:
        # If we had multiple batches, compile them together
        combined_insights = compile_insights(all_insights)

    print(
        f"Extracted insights from {len(ranked_content)} sources across {batch_number} batches"
    )

    # Ensure there's no markdown formatting in the final insights
    final_insights = clean_report_formatting(combined_insights)

    return {"insights": final_insights, "sources": scraped_content}


def generate_report(query: str, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a comprehensive research report of atleast 750-800 words based on extracted insights.
    Uses semantic search to ensure the most relevant sources are prioritized.
    The ThinkTagsOutputParser extracts content after </think> tags if present.
    Handles token limitations for different LLM providers.

    Args:
        query: The original research query
        extraction_results: Results from the extraction step

    Returns:
        Dictionary with report content and citations
    """
    insights = extraction_results.get("insights", "")
    sources = extraction_results.get("sources", [])

    # Use semantic search to rank sources by relevance to the query
    # This ensures the most relevant sources are cited in the report
    ranked_sources = semantic_search(query, sources, top_k=len(sources))

    # Create source information for citations
    source_info = []
    for i, source in enumerate(ranked_sources):
        source_info.append(
            {
                "id": i + 1,
                "title": source.get("title", f"Source {i+1}"),
                "url": source.get("url", ""),
                "authors": source.get("authors", []),
                "date_published": source.get("date_published", ""),
                "relevance": (
                    "high" if i < 5 else "medium" if i < 10 else "low"
                ),  # Add relevance rating
            }
        )

    # Format source info as a string
    sources_text = ""
    for source in source_info:
        relevance_indicator = (
            "[HIGH RELEVANCE]" if source["relevance"] == "high" else ""
        )
        sources_text += f"\nSource {source['id']}: {source['title']} {relevance_indicator} - {source['url']}"

    # Check current LLM provider to handle token limits appropriately
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    # Check if we need to split the report generation due to token limits
    # Estimate tokens based on character count (rough approximation)
    total_chars = len(query) + len(insights) + len(sources_text)

    # For Groq, handle the token limitation
    if provider == "groq" and total_chars > 12000:  # ~4000-5000 tokens
        return generate_report_in_parts(query, insights, sources_text, source_info)
    else:
        # Standard approach for models that can handle the full content
        return generate_single_report(query, insights, sources_text, source_info)


def generate_single_report(
    query: str, insights: str, sources_text: str, source_info: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate a report in a single LLM call.

    Args:
        query: The original research query
        insights: Extracted insights
        sources_text: Formatted source information
        source_info: List of source metadata

    Returns:
        Dictionary with report content and citations
    """
    # Prompt for generating the report
    prompt = PromptTemplate.from_template(
        """You are a professional research analyst tasked with creating a comprehensive report on a topic.
        
        Research Query: {query}
        
        Extracted Insights:
        {insights}
        
        Available Sources:
        {sources}
        
        Create a well-structured, professional research report that addresses the query comprehensively.
        The report should include:
        
        1. An abstract (instead of executive summary)
        2. Introduction to the topic
        3. Main findings organized by themes or categories
        4. Analysis and implications
        5. Conclusion
        
        If you want to include your thinking process, put it between <think> and </think> tags. Only text after the </think> tag will be used.
        
        Important guidelines:
        - Maintain a neutral, analytical tone
        - Cite sources using [Source X] notation when referencing specific information
        - Prioritize information from sources marked as [HIGH RELEVANCE]
        - Ensure the report is well-organized with clear headings and subheadings
        - Focus on providing valuable insights rather than just summarizing the sources
        - Be concise but comprehensive
        - Do NOT include a separate references or bibliography section - citations will be added separately
        - Do NOT use markdown code formatting (```), this will cause rendering issues
        - Format section titles using plain text with line breaks instead of markdown formatting
        - Use plain text formatting only
        
        Your report should be suitable for a professional audience seeking to understand this topic in depth.
        """
    )

    # Create chain
    chain = prompt | llm | StrOutputParser()

    # Generate report
    report_content = chain.invoke(
        {"query": query, "insights": insights, "sources": sources_text}
    )

    # Process the report content to remove any markdown code blocks or triple backticks
    # This prevents the report from being displayed as code
    processed_report = clean_report_formatting(report_content)

    # Format citations in AMA style
    formatted_citations = []
    for source in source_info:
        # Generate citation for this source
        citation = format_citation(source)
        formatted_citations.append(citation)

    return {"report_content": processed_report, "citations": formatted_citations}


def clean_report_formatting(report_text: str) -> str:
    """
    Clean up report formatting to prevent display issues.

    Args:
        report_text: The raw report text

    Returns:
        Cleaned report text
    """
    # Remove any triple backticks (code blocks)
    cleaned_text = re.sub(r"```[a-zA-Z]*\n", "", report_text)
    cleaned_text = cleaned_text.replace("```", "")

    # Make sure headings are properly formatted
    # Replace markdown headings with plain text headings if present
    cleaned_text = re.sub(r"^#+ (.*?)$", r"\1", cleaned_text, flags=re.MULTILINE)

    return cleaned_text


def generate_report_in_parts(
    query: str, insights: str, sources_text: str, source_info: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate a report in multiple parts to handle token limitations.

    Args:
        query: The original research query
        insights: Extracted insights
        sources_text: Formatted source information
        source_info: List of source metadata

    Returns:
        Dictionary with report content and citations
    """
    print("Generating report in parts due to token limitations")

    # First, generate the report structure and abstract
    structure_prompt = PromptTemplate.from_template(
        """You are a professional research analyst tasked with planning a comprehensive report.
        
        Research Query: {query}
        
        Based on the query, create a detailed outline for a professional research report that includes:
        
        1. An abstract (brief overview of the topic and findings)
        2. Introduction (background and context)
        3. Main findings (organized by 3-5 key themes or categories)
        4. Analysis and implications
        5. Conclusion
        
        For each section, provide a brief description of what should be included.
        
        Use plain text formatting only. Do NOT use markdown code formatting (```).
        
        If you want to include your thinking process, put it between <think> and </think> tags. Only text after the </think> tag will be used.
        """
    )

    # Create chain
    structure_chain = structure_prompt | llm | StrOutputParser()

    # Generate report structure
    report_structure = structure_chain.invoke({"query": query})

    # Next, generate the main content sections
    # We'll do this in a more focused way by providing just the relevant insights

    # Parse the insights into sections if possible
    insight_sections = {}
    current_section = "General"
    current_content = []

    for line in insights.split("\n"):
        if line.strip() and any(
            line.strip().startswith(h)
            for h in ["#", "Key", "Main", "Controversies", "Expert", "Recent"]
        ):
            # This looks like a section header
            if current_content:
                insight_sections[current_section] = "\n".join(current_content)
            current_section = line.strip()
            current_content = []
        else:
            current_content.append(line)

    # Add the last section
    if current_content:
        insight_sections[current_section] = "\n".join(current_content)

    # Generate each part of the report
    sections = {
        "abstract_intro": "Abstract and Introduction",
        "main_findings": "Main Findings",
        "analysis": "Analysis and Implications",
        "conclusion": "Conclusion",
    }

    section_contents = {}

    for section_key, section_name in sections.items():
        section_prompt = PromptTemplate.from_template(
            """You are generating the {section_name} section of a research report.
            
            Research Query: {query}
            
            Report Structure:
            {structure}
            
            Relevant Insights:
            {relevant_insights}
            
            Available Sources:
            {sources}
            
            Write a high-quality {section_name} section based on the provided information.
            
            If you want to include your thinking process, put it between <think> and </think> tags. Only text after the </think> tag will be used.
            
            Guidelines:
            - Maintain a neutral, analytical tone
            - Cite sources using [Source X] notation when referencing specific information
            - Prioritize information from sources marked as [HIGH RELEVANCE]
            - Be concise but comprehensive
            - Do NOT use markdown code formatting (```), this will cause rendering issues
            - Format section titles using plain text with line breaks instead of markdown formatting
            - Use plain text formatting only
            """
        )

        # Select relevant insights for this section
        relevant_insights = insights
        if section_key == "main_findings":
            # For main findings, try to include only the fact/arguments sections of insights
            relevant_insights = "\n\n".join(
                [
                    content
                    for title, content in insight_sections.items()
                    if any(
                        keyword in title.lower()
                        for keyword in [
                            "fact",
                            "statistic",
                            "argument",
                            "perspective",
                            "finding",
                        ]
                    )
                ]
            )
        elif section_key == "analysis":
            # For analysis, focus on controversies and expert opinions
            relevant_insights = "\n\n".join(
                [
                    content
                    for title, content in insight_sections.items()
                    if any(
                        keyword in title.lower()
                        for keyword in [
                            "controvers",
                            "debate",
                            "expert",
                            "opinion",
                            "analysis",
                            "implication",
                        ]
                    )
                ]
            )

        # Create chain
        section_chain = section_prompt | llm | StrOutputParser()

        # Generate section content
        section_content = section_chain.invoke(
            {
                "section_name": section_name,
                "query": query,
                "structure": report_structure,
                "relevant_insights": relevant_insights
                or insights,  # Fall back to all insights if specific section extraction fails
                "sources": sources_text,
            }
        )

        # Clean the formatting of each section
        section_contents[section_key] = clean_report_formatting(section_content)

    # Combine all sections into a complete report
    final_report = f"{section_contents['abstract_intro']}\n\n{section_contents['main_findings']}\n\n{section_contents['analysis']}\n\n{section_contents['conclusion']}"

    # Format citations in AMA style
    formatted_citations = []
    for source in source_info:
        # Generate citation for this source
        citation = format_citation(source)
        formatted_citations.append(citation)

    print(
        f"Report generation complete. Generated {len(formatted_citations)} citations."
    )
    return {"report_content": final_report, "citations": formatted_citations}


def format_citation(source: Dict[str, Any]) -> str:
    """
    Format a source into a citation.

    Args:
        source: Source metadata

    Returns:
        Formatted citation
    """
    # AMA style: Authors. Title. Publication date. URL.
    citation = ""

    # Authors (Last name First initial.)
    if source["authors"]:
        author_list = []
        for author in source["authors"]:
            # Try to format author names if possible
            parts = author.split()
            if len(parts) > 1:
                # Last name, first initial
                last_name = parts[-1]
                first_initial = parts[0][0] if parts[0] else ""
                author_list.append(f"{last_name} {first_initial}.")
            else:
                author_list.append(author)

        if len(author_list) > 6:
            # If more than 6 authors, use "et al."
            citation += ", ".join(author_list[:6]) + ", et al. "
        else:
            citation += ", ".join(author_list) + ". "

    # Title
    citation += f"{source['title']}. "

    # Publication date
    if source["date_published"]:
        citation += f"{source['date_published']}. "

    # URL
    citation += source["url"]

    return citation


def compile_insights(batch_insights: List[str]) -> str:
    """
    Compile insights from multiple batches into a coherent summary.

    Args:
        batch_insights: List of insights from different batches

    Returns:
        A compiled and summarized set of insights
    """
    # Join all insights with clear separators
    all_insights_text = "\n\n".join(
        [
            f"--- INSIGHTS BATCH {i+1} ---\n{insights}"
            for i, insights in enumerate(batch_insights)
        ]
    )

    # Create a prompt to compile and summarize the insights
    prompt = PromptTemplate.from_template(
        """You are a research analyst compiling insights from multiple analysis batches.
        
        Below are insights that were extracted from different batches of source materials.
        Your task is to compile them into a single, coherent set of insights, removing any duplicates
        and organizing the information logically.
        
        Batched Insights:
        {all_insights}
        
        Compile these insights into a single comprehensive analysis with these sections:
        1. Key facts and statistics
        2. Main arguments or perspectives
        3. Potential controversies or debates
        4. Expert opinions
        5. Recent developments
        
        If you want to include your thinking process, put it between <think> and </think> tags. Only text after the </think> tag will be used.
        
        Focus on creating a coherent narrative that incorporates the most important information from all batches.
        Eliminate redundancies and organize related information together.
        
        Important formatting guidelines:
        - Do NOT use markdown code formatting (```), this will cause rendering issues
        - Format section titles using plain text with line breaks instead of markdown formatting
        - Use plain text formatting only
        """
    )

    # Create chain
    chain = prompt | llm | StrOutputParser()

    # Compile insights
    print(f"Compiling insights from {len(batch_insights)} batches")
    compiled_insights = chain.invoke({"all_insights": all_insights_text})

    # Clean the formatting to prevent display issues
    cleaned_insights = clean_report_formatting(compiled_insights)

    return cleaned_insights
