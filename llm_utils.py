from typing import List, Dict, Any
import os
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import uuid

# Load environment variables
load_dotenv()


# Initialize LLM
llm = ChatOpenAI(model="o1-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Initialize OpenAI embedding model
print("Initializing OpenAI embeddings...")
openai_embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
print("OpenAI embeddings initialized")

# In-memory storage for document embeddings
document_store = {"ids": [], "texts": [], "embeddings": [], "metadatas": []}


def generate_search_queries(query: str, num_queries: int = 5) -> List[str]:
    """
    Generate diverse search queries based on the original query.

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
        if len(text) > 8000:
            print(
                f"Warning: Truncating text from {len(text)} to 8000 characters for embedding."
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

    # Prepare content for analysis - now using the ranked content
    combined_text = ""
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
            combined_text += f"\n\n--- {priority_label} {title} {pdf_indicator} ---\n{content[:content_length]}..."

    # Prompt for extracting insights
    prompt = PromptTemplate.from_template(
        """You are a research assistant tasked with extracting key insights from multiple sources.
        
        Below is content from various web sources, ranked by relevance. Sources marked as [Priority Source] and [PDF] 
        should be given more weight as they are likely more relevant and reliable.
        
        Extract the following:
        1. Key facts and statistics
        2. Main arguments or perspectives
        3. Potential controversies or debates
        4. Expert opinions
        5. Recent developments
        
        Content:
        {content}
        
        Provide your analysis in a structured format with clear headings for each category.
        Focus on extracting the most important and relevant information.
        Prioritize information from sources marked as [Priority Source] and [PDF].
        """
    )

    # Create chain
    chain = prompt | llm | StrOutputParser()

    # Extract insights
    insights = chain.invoke({"content": combined_text})

    print(f"Extracted insights from {len(ranked_content)} sources")

    return {"insights": insights, "sources": scraped_content}


def generate_report(query: str, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a comprehensive research report based on extracted insights.
    Uses semantic search to ensure the most relevant sources are prioritized.

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
        
        Important guidelines:
        - Maintain a neutral, analytical tone
        - Cite sources using [Source X] notation when referencing specific information
        - Prioritize information from sources marked as [HIGH RELEVANCE]
        - Ensure the report is well-organized with clear headings and subheadings
        - Focus on providing valuable insights rather than just summarizing the sources
        - Be concise but comprehensive
        - Do NOT include a separate references or bibliography section - citations will be added separately
        
        Your report should be suitable for a professional audience seeking to understand this topic in depth.
        """
    )

    # Create chain
    chain = prompt | llm | StrOutputParser()

    # Generate report
    report_content = chain.invoke(
        {"query": query, "insights": insights, "sources": sources_text}
    )

    # Format citations in AMA style
    formatted_citations = []
    for source in source_info:
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
                # AMA style uses "et al" for more than 6 authors
                citation += ", ".join(author_list[:6]) + ", et al. "
            else:
                citation += ", ".join(author_list) + ". "

        # Title
        citation += f"{source['title']}. "

        # Publication date
        if source["date_published"]:
            # Try to format date if it's a full date
            try:
                date_parts = source["date_published"].split("-")
                if len(date_parts) == 3:
                    # Format as Month Day, Year
                    year, month, day = date_parts
                    months = [
                        "January",
                        "February",
                        "March",
                        "April",
                        "May",
                        "June",
                        "July",
                        "August",
                        "September",
                        "October",
                        "November",
                        "December",
                    ]
                    month_name = (
                        months[int(month) - 1] if 1 <= int(month) <= 12 else month
                    )
                    citation += f"{month_name} {int(day)}, {year}. "
                else:
                    citation += f"{source['date_published']}. "
            except:
                citation += f"{source['date_published']}. "

        # URL
        citation += f"Accessed online at: {source['url']}"

        formatted_citations.append(citation)

    print(
        f"Report generation complete. Generated {len(formatted_citations)} citations."
    )
    return {"report_content": report_content, "citations": formatted_citations}
