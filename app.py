import streamlit as st
import os
import io
import base64
from dotenv import load_dotenv
from search_utils import search_tavily, scrape_content
from llm_utils import generate_search_queries, extract_insights, generate_report
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import markdown
import re
from datetime import datetime
import traceback

# Import Airtable utilities
import airtable_utils as airtable

# App Configuration
ENABLE_PDF_GENERATION = False  # Set to True to enable PDF generation functionality

# Load environment variables
load_dotenv()

# Initialize Airtable - will create the base and tables if they don't exist
try:
    airtable_initialized = airtable.initialize_airtable()
    if airtable_initialized:
        print("Airtable initialized successfully")
    else:
        print("Failed to initialize Airtable")
except Exception as e:
    print(f"Error initializing Airtable: {str(e)}")
    airtable_initialized = False


# Class for PDF creation with headers, footers, and markdown support
class ReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.footer_text = "Insights Agent (Beta) | Powered by Ability Labs"
        # Add custom page settings
        self.set_margins(15, 15, 15)  # Left, top, right margins
        # Add the DejaVu fonts with correct filenames and error handling
        try:
            self.add_font("DejaVuSans", "", "fonts/DejaVu-Sans-ExtraLight.ttf")
        except Exception as e:
            print(f"Warning: Could not load DejaVuSans font: {e}")
            # Fall back to a built-in font
            self.set_font("helvetica", "", 12)

        try:
            self.add_font("DejaVuSerif", "", "fonts/DejaVuSerif.ttf")
        except Exception as e:
            print(f"Warning: Could not load DejaVuSerif font: {e}")
            # Fall back to a built-in font
            self.set_font("times", "", 12)

        # For bold and italic versions, we'll just use the regular versions since they're not available
        try:
            # Use the same font for bold since we don't have a specific bold version
            self.add_font("DejaVuSans", "B", "fonts/DejaVu-Sans-ExtraLight.ttf")
        except Exception as e:
            print(f"Warning: Could not load DejaVuSans Bold font: {e}")

        try:
            # Use the same font for italic since we don't have a specific italic version
            self.add_font("DejaVuSans", "I", "fonts/DejaVu-Sans-ExtraLight.ttf")
        except Exception as e:
            print(f"Warning: Could not load DejaVuSans Italic font: {e}")

        try:
            # Use the regular serif font for bold since we don't have a specific bold version
            self.add_font("DejaVuSerif", "B", "fonts/DejaVuSerif.ttf")
        except Exception as e:
            print(f"Warning: Could not load DejaVuSerif Bold font: {e}")

        try:
            # Use the regular serif font for italic since we don't have a specific italic version
            self.add_font("DejaVuSerif", "I", "fonts/DejaVuSerif.ttf")
        except Exception as e:
            print(f"Warning: Could not load DejaVuSerif Italic font: {e}")

    def header(self):
        # Add a subtle header with a line
        self.set_font("DejaVuSans", "I", 10)
        self.cell(
            0, 10, "Insights Agent Research Report", 0, new_x=XPos.RIGHT, new_y=YPos.TOP
        )
        self.ln(7)
        self.line(15, 15, 195, 15)
        self.ln(10)

    def footer(self):
        # Position footer at 15mm from bottom
        self.set_y(-15)
        # Draw a subtle line
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(1)
        # Set font
        self.set_font("DejaVuSans", "I", 8)
        # Add page number and footer text
        self.cell(
            0,
            10,
            f"Page {self.page_no()}/{{nb}} - {self.footer_text}",
            0,
            new_x=XPos.RIGHT,
            new_y=YPos.TOP,
            align="C",
        )


def markdown_to_pdf(title, content, citations):
    try:
        print(f"Starting PDF generation for: {title}")
        # Create PDF object
        pdf = ReportPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.alias_nb_pages()  # For page numbering
        pdf.add_page()
        print("PDF object created successfully")

        # Add title with better styling
        pdf.set_font("DejaVuSerif", "B", 18)
        pdf.set_text_color(44, 62, 80)

        # Use multi_cell for title to handle long titles with proper wrapping
        # Calculate available width (accounting for margins)
        available_width = pdf.w - 2 * pdf.l_margin

        # Set text alignment to center
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(available_width, 10, title, align="C")
        pdf.ln(5)

        # Add horizontal line after title
        pdf.line(30, pdf.get_y(), 180, pdf.get_y())
        pdf.ln(10)

        # Add table of contents with improved styling
        pdf.set_font("DejaVuSerif", "B", 14)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 10, "Table of Contents", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

        # Style table of contents entries
        pdf.set_font("DejaVuSans", "", 12)
        pdf.set_text_color(0, 0, 0)
        toc_items = [
            "1. Abstract",
            "2. Introduction",
            "3. Main Findings",
            "4. Discussion",
            "5. Conclusion",
            "6. Citations",
        ]

        for item in toc_items:
            pdf.cell(0, 8, item, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)
        print("Table of contents added successfully")

        # Convert markdown to simple HTML
        html_content = markdown.markdown(content)

        # Debug: Print the first 100 characters of content to see format
        print(f"Content preview: {content[:100]}...")

        # Improved section extraction - more robust pattern matching
        # Define section keywords to look for (case insensitive)
        section_keywords = [
            "abstract",
            "introduction",
            "main findings",
            "discussion|analysis",  # Allow for either Discussion or Analysis
            "conclusion",
        ]

        # Find all level 2 headers (## headers) in the content
        header_pattern = re.compile(r"^##\s+(.*?)$", re.MULTILINE)
        headers = header_pattern.findall(content)

        # Debug: Print all found headers
        print(f"Found headers: {headers}")

        # Map found headers to our expected sections based on keywords
        section_map = {}
        for i, keyword in enumerate(section_keywords):
            keyword_parts = keyword.split("|")
            for header in headers:
                matched = False
                for part in keyword_parts:
                    if part.lower() in header.lower():
                        section_map[i] = f"## {header}"
                        matched = True
                        break
                if matched:
                    break

            # If no matching header found, use the default section name
            if i not in section_map:
                # For keywords with alternatives, use the first one
                default_keyword = keyword_parts[0].title()
                section_map[i] = f"## {default_keyword}"

        # Debug: Print section mapping
        print(f"Section mapping: {section_map}")

        # Track processed sections for summary
        processed_sections = []
        empty_sections = []

        # Extract content for each section using the mapped headers
        for i, keyword in enumerate(section_keywords):
            section_header = section_map[i]
            keyword_display = keyword.split("|")[0].title()

            # Find the current section in markdown format
            section_pattern = re.escape(section_header)

            # Determine the pattern end - either next section or end of content
            if i < len(section_keywords) - 1:
                # Find the next section that actually exists in the content
                next_headers = []
                for j in range(i + 1, len(section_keywords)):
                    if j in section_map:
                        next_headers.append(re.escape(section_map[j]))

                if next_headers:
                    # Use alternation to match any of the next headers
                    next_pattern = "|".join(next_headers)
                    section_match = re.search(
                        f"{section_pattern}(.*?)(?:{next_pattern})",
                        content,
                        re.DOTALL,
                    )
                else:
                    # If no next headers found, look for any ## header
                    section_match = re.search(
                        f"{section_pattern}(.*?)(?:##|$)", content, re.DOTALL
                    )
            else:
                section_match = re.search(
                    f"{section_pattern}(.*?)$", content, re.DOTALL
                )

            # Debug: Print the regex pattern being used
            print(f"Section {i+1} ({keyword_display}) pattern: {section_match.pattern}")

            if section_match:
                section_content = section_match.group(1).strip()
                # Debug: Print content length
                print(
                    f"Section {i+1} ({keyword_display}) content length: {len(section_content)} chars"
                )
                processed_sections.append(keyword_display)

                # Add section heading with improved styling
                pdf.add_page()

                # Add section number and title with better styling
                pdf.set_font("DejaVuSerif", "B", 16)
                pdf.set_text_color(44, 62, 80)
                section_title = section_header.replace("##", "").strip()

                # Use multi_cell for section titles to handle long titles
                available_width = pdf.w - 2 * pdf.l_margin
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(
                    available_width, 10, f"{i+1}. {section_title}", align="L"
                )

                # Add decorative line under section title
                pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                pdf.ln(7)

                # Add section content with improved formatting
                pdf.set_font("DejaVuSans", "", 12)
                pdf.set_text_color(0, 0, 0)

                # Split content into paragraphs
                paragraphs = section_content.split("\n\n")
                for paragraph in paragraphs:
                    if paragraph.strip():
                        # Handle subsections (if any)
                        if paragraph.strip().startswith("###"):
                            pdf.set_font("DejaVuSerif", "B", 13)
                            pdf.set_text_color(70, 70, 70)
                            pdf.ln(3)
                            pdf.multi_cell(0, 10, paragraph.replace("###", "").strip())
                            pdf.set_font("DejaVuSans", "", 12)
                            pdf.set_text_color(0, 0, 0)
                            pdf.ln(2)
                        # Handle simple list items
                        elif paragraph.strip().startswith("- "):
                            items = paragraph.strip().split("\n- ")
                            for item in items:
                                if item.strip():
                                    pdf.set_text_color(0, 0, 0)
                                    # Add some indentation for list items
                                    pdf.set_x(20)
                                    # Use bullet character with Unicode-compatible font
                                    pdf.multi_cell(0, 10, f"• {item.strip()}")
                                    pdf.ln(1)
                        else:
                            # Handle normal paragraphs with better line spacing
                            pdf.multi_cell(0, 10, paragraph.strip())
                            pdf.ln(3)  # Better spacing between paragraphs
            else:
                # Debug: Print when section not found
                print(f"Section {i+1} ({keyword_display}) not found in content")
                empty_sections.append(keyword_display)

                # If section not found, add an empty section
                pdf.add_page()
                pdf.set_font("DejaVuSerif", "B", 16)
                pdf.set_text_color(44, 62, 80)

                # Use multi_cell for empty section titles
                available_width = pdf.w - 2 * pdf.l_margin
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(
                    available_width, 10, f"{i+1}. {keyword_display}", align="L"
                )

                pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                pdf.ln(7)
                pdf.set_font("DejaVuSans", "", 12)
                pdf.set_text_color(0, 0, 0)
                pdf.multi_cell(0, 10, "No content available for this section.")

        print(f"Processed sections: {', '.join(processed_sections)}")
        if empty_sections:
            print(f"Empty sections: {', '.join(empty_sections)}")
        else:
            print("All sections were found and processed successfully")

        # Add citations section with improved styling
        pdf.add_page()
        pdf.set_font("DejaVuSerif", "B", 16)
        pdf.set_text_color(44, 62, 80)

        # Use multi_cell for citations title
        available_width = pdf.w - 2 * pdf.l_margin
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(available_width, 10, "6. Citations", align="L")

        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(7)

        pdf.set_font("DejaVuSans", "", 11)
        pdf.set_text_color(0, 0, 0)

        for citation in citations:
            # Add bullet points to citations with proper indentation
            pdf.set_x(20)
            pdf.multi_cell(0, 10, f"• {citation}")
            pdf.ln(1)

        print(f"Added {len(citations)} citations to the PDF")

        # Add disclaimer with improved styling
        pdf.add_page()
        pdf.set_font("DejaVuSerif", "B", 16)
        pdf.set_text_color(44, 62, 80)

        # Use multi_cell for disclaimer title
        available_width = pdf.w - 2 * pdf.l_margin
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(available_width, 10, "Disclaimer", align="L")

        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(7)

        pdf.set_font("DejaVuSans", "I", 12)
        pdf.set_text_color(80, 80, 80)
        disclaimer_text = """This tool is still in development and the information provided should be crosschecked for accuracy. The Insights Agent uses artificial intelligence to generate reports based on available information, which may not always be complete or up-to-date. This report should be used as a starting point for research rather than as definitive information. Always consult qualified healthcare professionals for medical advice."""
        pdf.multi_cell(0, 10, disclaimer_text)

        print("Added disclaimer to the PDF")

        # Return PDF as bytes - handle different output types properly
        output = pdf.output(dest="S")
        # Check if output is already bytes/bytearray, if not encode it
        if isinstance(output, bytearray):
            result = bytes(output)  # Convert bytearray to bytes
        elif isinstance(output, bytes):
            result = output
        else:
            result = output.encode("latin1")  # Return as bytes

        print(f"PDF generation completed successfully. PDF size: {len(result)} bytes")
        return result
    except Exception as e:
        # Log the error for debugging
        print(f"Error in PDF generation: {str(e)}")
        # Create a simple PDF with built-in fonts as fallback
        try:
            print("Attempting fallback PDF generation")
            fallback_pdf = FPDF()
            fallback_pdf.add_page()
            fallback_pdf.set_font("helvetica", "B", 16)
            fallback_pdf.cell(0, 10, "Error creating formatted PDF", ln=True)
            fallback_pdf.set_font("helvetica", "", 12)
            fallback_pdf.multi_cell(
                0,
                10,
                f"A simple version of the report is provided due to font issues: {str(e)}",
            )
            fallback_pdf.ln(10)

            # Add title
            fallback_pdf.set_font("helvetica", "B", 14)

            # Use multi_cell for title to handle long titles with proper wrapping
            available_width = fallback_pdf.w - 2 * fallback_pdf.l_margin
            fallback_pdf.set_x(fallback_pdf.l_margin)
            fallback_pdf.multi_cell(available_width, 10, title, align="C")
            fallback_pdf.ln(5)

            print("Starting fallback section processing")

            # Add content - improved section handling
            fallback_pdf.set_font("helvetica", "", 12)

            # Define section keywords to look for (case insensitive)
            section_keywords = [
                "abstract",
                "introduction",
                "main findings",
                "discussion|analysis",  # Allow for either Discussion or Analysis
                "conclusion",
            ]

            # Find all level 2 headers (## headers) in the content
            header_pattern = re.compile(r"^##\s+(.*?)$", re.MULTILINE)
            headers = header_pattern.findall(content)

            # Track processed sections for fallback
            fallback_processed = []
            fallback_empty = []

            # Process each section
            for i, keyword in enumerate(section_keywords):
                keyword_parts = keyword.split("|")
                keyword_display = keyword_parts[0].title()

                # Find a matching header
                matching_header = None
                for header in headers:
                    for part in keyword_parts:
                        if part.lower() in header.lower():
                            matching_header = header
                            break
                    if matching_header:
                        break

                if matching_header:
                    # Add section header
                    fallback_pdf.add_page()
                    fallback_pdf.set_font("helvetica", "B", 14)

                    # Use multi_cell for section titles
                    available_width = fallback_pdf.w - 2 * fallback_pdf.l_margin
                    fallback_pdf.set_x(fallback_pdf.l_margin)
                    fallback_pdf.multi_cell(
                        available_width, 10, f"{i+1}. {matching_header}", align="L"
                    )

                    fallback_pdf.ln(5)

                    print(f"Processing fallback section: {matching_header}")
                    fallback_processed.append(keyword_display)

                    # Find section content
                    section_pattern = f"## {re.escape(matching_header)}"

                    # Determine the pattern end - either next section or end of content
                    if i < len(section_keywords) - 1:
                        # Look for next headers
                        next_headers = []
                        for j in range(i + 1, len(section_keywords)):
                            for next_header in headers:
                                for next_part in section_keywords[j].split("|"):
                                    if next_part.lower() in next_header.lower():
                                        next_headers.append(f"## {next_header}")
                                        break

                        if next_headers:
                            # Use alternation to match any of the next headers
                            next_pattern = "|".join(
                                [re.escape(h) for h in next_headers]
                            )
                            section_match = re.search(
                                f"{section_pattern}(.*?)(?:{next_pattern})",
                                content,
                                re.DOTALL,
                            )
                        else:
                            # If no next headers found, look for any ## header
                            section_match = re.search(
                                f"{section_pattern}(.*?)(?:##|$)", content, re.DOTALL
                            )
                    else:
                        section_match = re.search(
                            f"{section_pattern}(.*?)$", content, re.DOTALL
                        )

                    if section_match:
                        section_content = section_match.group(1).strip()
                        fallback_pdf.set_font("helvetica", "", 12)

                        print(
                            f"Fallback section {i+1} content length: {len(section_content)} chars"
                        )

                        # Process paragraphs
                        paragraphs = section_content.split("\n\n")
                        for paragraph in paragraphs:
                            if paragraph.strip():
                                fallback_pdf.multi_cell(0, 10, paragraph.strip())
                                fallback_pdf.ln(3)
                else:
                    # Add empty section with default name
                    fallback_pdf.add_page()
                    fallback_pdf.set_font("helvetica", "B", 14)

                    # Use multi_cell for empty section titles
                    available_width = fallback_pdf.w - 2 * fallback_pdf.l_margin
                    fallback_pdf.set_x(fallback_pdf.l_margin)
                    fallback_pdf.multi_cell(
                        available_width, 10, f"{i+1}. {keyword_display}", align="L"
                    )

                    fallback_pdf.ln(5)
                    fallback_pdf.set_font("helvetica", "", 12)
                    fallback_pdf.multi_cell(
                        0, 10, "No content available for this section."
                    )
                    fallback_empty.append(keyword_display)
                    print(f"Fallback section not found: {keyword_display}")

            print(f"Fallback processed sections: {', '.join(fallback_processed)}")
            if fallback_empty:
                print(f"Fallback empty sections: {', '.join(fallback_empty)}")

            # Add citations
            fallback_pdf.add_page()
            fallback_pdf.set_font("helvetica", "B", 14)

            # Use multi_cell for citations title
            available_width = fallback_pdf.w - 2 * fallback_pdf.l_margin
            fallback_pdf.set_x(fallback_pdf.l_margin)
            fallback_pdf.multi_cell(available_width, 10, "Citations", align="L")

            fallback_pdf.ln(5)
            fallback_pdf.set_font("helvetica", "", 12)
            for citation in citations:
                fallback_pdf.multi_cell(0, 10, f"- {citation}")
                fallback_pdf.ln(2)

            print(f"Added {len(citations)} citations to fallback PDF")

            # Add disclaimer
            fallback_pdf.add_page()
            fallback_pdf.set_font("helvetica", "B", 14)

            # Use multi_cell for disclaimer title
            available_width = fallback_pdf.w - 2 * fallback_pdf.l_margin
            fallback_pdf.set_x(fallback_pdf.l_margin)
            fallback_pdf.multi_cell(available_width, 10, "Disclaimer", align="L")

            fallback_pdf.ln(5)
            fallback_pdf.set_font("helvetica", "I", 12)
            disclaimer_text = """This tool is still in development and the information provided should be crosschecked for accuracy. The Insights Agent uses artificial intelligence to generate reports based on available information, which may not always be complete or up-to-date."""
            fallback_pdf.multi_cell(0, 10, disclaimer_text)

            print("Added disclaimer to fallback PDF")

            # Return PDF as bytes - handle different output types properly
            output = fallback_pdf.output(dest="S")
            # Check if output is already bytes/bytearray, if not encode it
            if isinstance(output, bytearray):
                result = bytes(output)  # Convert bytearray to bytes
            elif isinstance(output, bytes):
                result = output
            else:
                result = output.encode("latin1")

            print(
                f"Fallback PDF generation completed successfully. PDF size: {len(result)} bytes"
            )
            return result
        except Exception as fallback_error:
            # If even the fallback fails, raise the original error
            print(f"Fallback PDF also failed: {str(fallback_error)}")
            raise e


# Check if API keys are set
missing_keys = []
if not os.getenv("OPENAI_API_KEY"):
    missing_keys.append("OPENAI_API_KEY")
if not os.getenv("TAVILY_API_KEY"):
    missing_keys.append("TAVILY_API_KEY")

# Set up the Streamlit app
st.set_page_config(page_title="Insights Agent", page_icon="🏥")

st.title("🏥 Insights Agent")
st.markdown(
    """
This research agent helps you find and analyze the latest information on physiotherapy and rehabilitation topics.
Enter your research question below to get insights from multiple sources.
"""
)

# Display warning if API keys are missing
if missing_keys:
    st.warning(
        f"⚠️ The following API keys are missing in your .env file: {', '.join(missing_keys)}. "
        f"Some functionality may be limited."
    )

# Initialize session state for user information
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "user_info_submitted" not in st.session_state:
    st.session_state.user_info_submitted = False
if "user_info_timestamp" not in st.session_state:
    st.session_state.user_info_timestamp = None
if "user_airtable_id" not in st.session_state:
    st.session_state.user_airtable_id = None
if "research_history" not in st.session_state:
    st.session_state.research_history = []
if "report_rated" not in st.session_state:
    st.session_state.report_rated = False
if "current_rating" not in st.session_state:
    st.session_state.current_rating = None
if "rating_feedback" not in st.session_state:
    st.session_state.rating_feedback = ""
# Add new session state for interface reset
if "reset_interface" not in st.session_state:
    st.session_state.reset_interface = False
# Add new session state for report results
if "report_results" not in st.session_state:
    st.session_state.report_results = None
# Add new session state for query
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
# User input
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
# Add new session state for search data storage
if "search_data" not in st.session_state:
    st.session_state.search_data = {}
if "all_search_history" not in st.session_state:
    st.session_state.all_search_history = []
# Add session state for analytics data
if "research_analytics" not in st.session_state:
    st.session_state.research_analytics = {}
# Add session state to control Airtable operations for performance
if "skip_airtable_operations" not in st.session_state:
    st.session_state.skip_airtable_operations = True
# Add session state to track if data has been saved to Airtable
if "data_saved_to_airtable" not in st.session_state:
    st.session_state.data_saved_to_airtable = False

# User information collection
if not st.session_state.user_info_submitted:
    st.subheader("User Information")
    st.markdown("Please provide your information before starting research.")

    user_name = st.text_input("Full Name", value=st.session_state.user_name)
    user_email = st.text_input("Email Address", value=st.session_state.user_email)

    # Validate email format
    email_valid = True
    if user_email and "@" not in user_email:
        st.warning("Please enter a valid email address.")
        email_valid = False

    # Privacy notice
    st.info(
        """
    **Privacy Notice**: Your information will be stored securely and used only for:
    - Tracking your research queries
    - Improving our service
    - Contacting you with important updates
    
    We will never share your information with third parties.
    """
    )

    if st.button(
        "Submit Information", disabled=not (user_name and user_email and email_valid)
    ):
        st.session_state.user_name = user_name
        st.session_state.user_email = user_email
        st.session_state.user_info_submitted = True
        st.session_state.user_info_timestamp = datetime.now().isoformat()

        # Store user data for later Airtable saving
        st.session_state.user_data_for_airtable = {
            "name": user_name,
            "email": user_email,
            "timestamp": st.session_state.user_info_timestamp,
        }

        # Generate a temporary user ID until we save to Airtable
        st.session_state.user_airtable_id = (
            f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        print(
            f"User information stored with temporary ID: {st.session_state.user_airtable_id}"
        )

        st.rerun()

    # Don't show the rest of the app until user info is submitted
    st.stop()


# Function to prepare user data for database storage
def prepare_user_data():
    """
    Prepare user data for future database storage.
    Returns a dictionary with user information.
    """
    return {
        "name": st.session_state.user_name,
        "email": st.session_state.user_email,
        "timestamp": st.session_state.user_info_timestamp,
        "last_query": query if "query" in locals() else None,
    }


# Display user information
st.sidebar.subheader("User Information")
st.sidebar.write(f"**Name:** {st.session_state.user_name}")
st.sidebar.write(f"**Email:** {st.session_state.user_email}")
if st.session_state.user_info_timestamp:
    timestamp = datetime.fromisoformat(st.session_state.user_info_timestamp)
    st.sidebar.write(f"**Registered:** {timestamp.strftime('%Y-%m-%d %H:%M')}")
if st.sidebar.button("Change Information"):
    st.session_state.user_info_submitted = False
    st.rerun()


# User input
def update_query():
    st.session_state.query_input = st.session_state.temp_query


query = st.text_input(
    "What would you like to research?",
    placeholder="e.g., Effectiveness of dry needling for chronic low back pain",
    key="temp_query",
    on_change=update_query,
    value=st.session_state.query_input,
)

# Always use the session state value for processing
query = st.session_state.query_input

# Create a placeholder for the progress bar
progress_placeholder = st.empty()
report_placeholder = st.empty()

# Initialize session state for stop button and research process
if "research_running" not in st.session_state:
    st.session_state.research_running = False
if "stop_research" not in st.session_state:
    st.session_state.stop_research = False
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False


# Function to save all stored data to Airtable
def save_all_data_to_airtable():
    """
    Save all stored data to Airtable at once.
    This includes user data, research sessions, search data, analytics, and ratings.
    """
    print("\n" + "=" * 50)
    print("SAVING ALL DATA TO AIRTABLE")
    print("=" * 50)

    if not airtable_initialized:
        print("Airtable is not initialized. Skipping data saving.")
        return False

    # Print debug information about Airtable configuration
    print(f"Airtable BASE_ID: {airtable.BASE_ID}")
    print(f"Airtable RESEARCH_TABLE_ID: {airtable.RESEARCH_TABLE_ID}")
    print(
        f"Other table IDs: Users={airtable.USERS_TABLE_ID}, Search={airtable.SEARCH_DATA_TABLE_ID}, Analytics={airtable.ANALYTICS_TABLE_ID}"
    )

    # Skip saving if we're configured to avoid Airtable operations for performance
    if (
        hasattr(st.session_state, "skip_airtable_operations")
        and st.session_state.skip_airtable_operations
    ):
        print("Skipping Airtable operations for performance reasons.")
        return False

    saved_data = {
        "user": False,
        "research_sessions": 0,
        "search_data": 0,
        "analytics": 0,
        "ratings": 0,
    }

    try:
        # 1. Save user data if it exists and hasn't been saved yet
        if hasattr(
            st.session_state, "user_data_for_airtable"
        ) and st.session_state.user_airtable_id.startswith("temp_"):
            user_data = st.session_state.user_data_for_airtable

            print(f"Saving user data: {user_data}")

            # Fix: Don't use the User ID field in Airtable, let Airtable generate its own ID
            user_id = airtable.get_or_create_user(
                name=user_data["name"],
                email=user_data["email"],
                timestamp=user_data["timestamp"],
            )

            st.session_state.user_airtable_id = user_id
            saved_data["user"] = True
            print(f"User saved to Airtable with ID: {user_id}")

        # 2. Save research sessions
        if hasattr(st.session_state, "current_research_data"):
            research_data = st.session_state.current_research_data
            # Update user_id if it was just created
            if saved_data["user"]:
                research_data["user_id"] = st.session_state.user_airtable_id

            print(f"Saving research session: {research_data}")

            # Check if we have an existing record to update
            existing_airtable_id = None
            for entry in st.session_state.research_history:
                if entry["id"] == research_data["id"] and "airtable_id" in entry:
                    existing_airtable_id = entry["airtable_id"]
                    break

            if existing_airtable_id:
                # We have an existing record, update it instead of creating a new one
                print(
                    f"Updating existing research session with Airtable ID: {existing_airtable_id}"
                )

                # Create update data
                update_data = {
                    "status": research_data.get("status", "completed"),
                    "completion_timestamp": research_data.get("completion_timestamp"),
                    "total_time_seconds": research_data.get("total_time_seconds"),
                }

                # Include the full report if available
                if "full_report" in research_data:
                    update_data["full_report"] = research_data["full_report"]
                    print(
                        f"Including full report in update (length: {len(update_data['full_report'])} characters)"
                    )

                # Update the existing record
                try:
                    airtable.update_research_session(
                        research_id=research_data["id"], update_data=update_data
                    )
                    saved_data["research_sessions"] += 1
                    print(
                        f"Research session updated in Airtable with ID: {existing_airtable_id}"
                    )
                except Exception as e:
                    print(f"Error updating research session: {str(e)}")
            else:
                # Create a new record
                airtable_id = airtable.create_research_session(research_data)

                # Update the research history entry with the Airtable ID
                for entry in st.session_state.research_history:
                    if entry["id"] == research_data["id"]:
                        entry["airtable_id"] = airtable_id
                        print(
                            f"Updated research history entry with Airtable ID: {airtable_id}"
                        )
                        break

                saved_data["research_sessions"] += 1
                print(f"Research session saved to Airtable with ID: {airtable_id}")

        # 3. Save search data
        if (
            hasattr(st.session_state, "all_search_history")
            and st.session_state.all_search_history
        ):
            print(
                f"Saving {len(st.session_state.all_search_history)} search history entries"
            )
            for search_data in st.session_state.all_search_history:
                # Check if this search data has already been saved to Airtable
                if "airtable_id" not in search_data:
                    print(
                        f"Saving search data for research ID: {search_data.get('research_id')}"
                    )
                    search_data_id = airtable.create_search_data(search_data)
                    search_data["airtable_id"] = search_data_id
                    saved_data["search_data"] += 1
                    print(f"Search data saved to Airtable with ID: {search_data_id}")

        # 4. Save analytics data
        if (
            hasattr(st.session_state, "analytics_history")
            and st.session_state.analytics_history
        ):
            print(f"Saving {len(st.session_state.analytics_history)} analytics entries")
            for analytics_data in st.session_state.analytics_history:
                # Check if this analytics data has already been saved to Airtable
                if "airtable_id" not in analytics_data:
                    print(
                        f"Saving analytics data for research ID: {analytics_data.get('research_id')}"
                    )
                    analytics_id = airtable.create_analytics_data(analytics_data)
                    analytics_data["airtable_id"] = analytics_id
                    saved_data["analytics"] += 1
                    print(f"Analytics data saved to Airtable with ID: {analytics_id}")

        # 5. Save ratings
        if (
            hasattr(st.session_state, "ratings_to_save")
            and st.session_state.ratings_to_save
        ):
            print(f"Saving {len(st.session_state.ratings_to_save)} ratings")
            for rating_data in st.session_state.ratings_to_save:
                print(f"Saving rating for research ID: {rating_data['research_id']}")

                # Ensure rating is a number
                try:
                    rating_value = float(rating_data["rating"])
                except (ValueError, TypeError):
                    print(f"Warning: Invalid rating value: {rating_data['rating']}")
                    rating_value = 3  # Default to 3 if invalid

                # Update the rating data with the validated rating value
                rating_update = {
                    "rating": rating_value,
                    "rating_feedback": rating_data["rating_feedback"],
                    "rating_timestamp": rating_data["rating_timestamp"],
                }

                print(f"Rating update data: {rating_update}")

                try:
                    airtable.update_research_session(
                        research_id=rating_data["research_id"],
                        update_data=rating_update,
                    )
                    saved_data["ratings"] += 1
                    print(
                        f"Rating saved to Airtable for research ID: {rating_data['research_id']}"
                    )
                except Exception as e:
                    print(f"Error saving rating to Airtable: {str(e)}")

                    # Try to find the research session in the history and get its Airtable ID
                    airtable_id = None
                    for entry in st.session_state.research_history:
                        if (
                            entry["id"] == rating_data["research_id"]
                            and "airtable_id" in entry
                        ):
                            airtable_id = entry["airtable_id"]
                            break

                    if airtable_id:
                        print(
                            f"Found Airtable ID {airtable_id} for research ID {rating_data['research_id']}"
                        )
                        try:
                            # Try direct update using the Airtable record ID
                            url = f"{airtable.AIRTABLE_API_URL}/{airtable.BASE_ID}/{airtable.RESEARCH_TABLE_ID}/{airtable_id}"
                            payload = {
                                "fields": {
                                    "Rating": rating_value,
                                    "Rating Feedback": rating_data["rating_feedback"],
                                    "Rating Timestamp": rating_data["rating_timestamp"],
                                }
                            }

                            response = requests.patch(
                                url, headers=airtable.get_headers(), json=payload
                            )

                            if response.status_code == 200:
                                saved_data["ratings"] += 1
                                print(
                                    f"Rating saved directly to Airtable record ID: {airtable_id}"
                                )
                            else:
                                print(
                                    f"Failed to save rating directly: {response.text}"
                                )
                        except Exception as direct_error:
                            print(f"Error saving rating directly: {str(direct_error)}")
                    else:
                        print(
                            f"Could not find Airtable ID for research ID: {rating_data['research_id']}"
                        )

            # Clear the ratings to save only if we successfully saved them
            if saved_data["ratings"] > 0:
                try:
                    st.session_state.ratings_to_save = []
                    print("Cleared ratings_to_save after successful saving")
                except Exception as e:
                    print(f"Error clearing ratings_to_save: {str(e)}")

        print(f"All data saved to Airtable: {saved_data}")
        print("=" * 50 + "\n")
        return True
    except Exception as e:
        print(f"Error saving data to Airtable: {str(e)}")
        traceback.print_exc()
        print("=" * 50 + "\n")
        return False


# Function to handle rating submission
def submit_rating():
    print("\n" + "=" * 50)
    print("SUBMITTING RATING")
    print("=" * 50)

    try:
        # Update the research history with the rating
        if st.session_state.research_history:
            last_entry = st.session_state.research_history[-1]

            # Ensure rating is a number
            try:
                rating_value = float(st.session_state.current_rating)
                print(f"Rating value: {rating_value}")
            except (ValueError, TypeError):
                print(
                    f"Warning: Invalid rating value: {st.session_state.current_rating}"
                )
                rating_value = 3  # Default to 3 if invalid

            last_entry["rating"] = rating_value
            last_entry["rating_feedback"] = st.session_state.rating_feedback

            # Add rating timestamp
            rating_timestamp = datetime.now().isoformat()
            last_entry["rating_timestamp"] = rating_timestamp

            print(f"Research entry to update: {last_entry['id']}")
            print(
                f"Rating: {rating_value}, Feedback: {st.session_state.rating_feedback}"
            )

            # Update analytics if available
            if (
                "analytics_history" in st.session_state
                and st.session_state.analytics_history
            ):
                # Find matching analytics entry
                for analytics in st.session_state.analytics_history:
                    if analytics.get("research_id") == last_entry.get("id"):
                        # Update rating information
                        analytics["result_metrics"]["rating"] = rating_value
                        analytics["result_metrics"][
                            "rating_feedback"
                        ] = st.session_state.rating_feedback

                        # Record rating timestamp
                        analytics["result_metrics"][
                            "rating_timestamp"
                        ] = rating_timestamp

                        print(f"Analytics updated with rating: {rating_value}/5")
                        break

            # Create rating data for Airtable saving
            rating_data = {
                "research_id": last_entry.get("id"),
                "rating": rating_value,
                "rating_feedback": st.session_state.rating_feedback,
                "rating_timestamp": rating_timestamp,
            }

            print(f"Rating data prepared for Airtable: {rating_data}")

            # Store rating data for later Airtable saving
            if "ratings_to_save" not in st.session_state:
                st.session_state.ratings_to_save = []

            st.session_state.ratings_to_save.append(rating_data)

            print(f"Rating stored for later saving to Airtable: {rating_value}/5")

            # Check if we have an Airtable ID for this research session
            if "airtable_id" in last_entry:
                print(f"Research session has Airtable ID: {last_entry['airtable_id']}")
            else:
                print("Warning: Research session does not have an Airtable ID")

            # Enable Airtable operations for rating submission
            previous_skip_setting = st.session_state.skip_airtable_operations
            st.session_state.skip_airtable_operations = False

            print("Attempting to save rating to Airtable...")

            # Try to save directly to Airtable first
            try:
                # Direct update to Airtable
                airtable.update_research_session(
                    research_id=last_entry.get("id"),
                    update_data={
                        "rating": rating_value,
                        "rating_feedback": st.session_state.rating_feedback,
                        "rating_timestamp": rating_timestamp,
                    },
                )
                print(
                    f"Rating directly saved to Airtable for research ID: {last_entry.get('id')}"
                )
                st.session_state.data_saved_to_airtable = True
            except Exception as e:
                print(f"Error directly saving rating to Airtable: {str(e)}")
                print("Falling back to save_all_data_to_airtable method...")

                # Fall back to the general save method
                try:
                    save_result = save_all_data_to_airtable()
                    if save_result:
                        print("Successfully saved all data to Airtable")
                        st.session_state.data_saved_to_airtable = True
                    else:
                        print("Failed to save data to Airtable")
                        st.session_state.data_saved_to_airtable = False
                except Exception as e2:
                    print(f"Error in fallback save method: {str(e2)}")
                    st.session_state.data_saved_to_airtable = False

            # Restore previous setting
            st.session_state.skip_airtable_operations = previous_skip_setting

        # Mark the report as rated
        st.session_state.report_rated = True

        # Log the rating
        print(f"Report rated: {st.session_state.current_rating}/5")
        if st.session_state.rating_feedback:
            print(f"Rating feedback: {st.session_state.rating_feedback}")

    except Exception as e:
        print(f"Error in submit_rating function: {str(e)}")
        import traceback

        traceback.print_exc()
        # Mark as rated anyway to prevent multiple submissions
        st.session_state.report_rated = True
        # Show error to user
        st.error(
            "There was an error submitting your rating, but it's been saved and will be processed later."
        )

    print("=" * 50 + "\n")


# Function to update rating
def update_rating():
    if "rating_value" in st.session_state:
        st.session_state.current_rating = st.session_state.rating_value


# Function to reset rating for a new report
def reset_rating():
    st.session_state.report_rated = False
    st.session_state.current_rating = None
    st.session_state.rating_feedback = ""


# Function to reset the interface
def reset_interface():
    # Only save to Airtable if we have ratings and haven't saved them yet
    if (
        hasattr(st.session_state, "ratings_to_save")
        and st.session_state.ratings_to_save
        and not st.session_state.data_saved_to_airtable
    ):
        print("Saving pending ratings before reset...")
        try:
            # Enable Airtable operations temporarily
            previous_skip_setting = st.session_state.skip_airtable_operations
            st.session_state.skip_airtable_operations = False
            save_result = save_all_data_to_airtable()
            # Restore previous setting
            st.session_state.skip_airtable_operations = previous_skip_setting

            if save_result:
                st.session_state.data_saved_to_airtable = True
                print("Successfully saved data to Airtable before reset")
                # Only clear ratings_to_save if saving was successful
                st.session_state.ratings_to_save = []
            else:
                print("Failed to save data to Airtable before reset")
                # Don't clear the ratings so they can be saved next time
        except Exception as e:
            print(f"Error while saving ratings before reset: {str(e)}")
            # Restore previous setting in case of error
            st.session_state.skip_airtable_operations = previous_skip_setting

    st.session_state.reset_interface = True
    st.session_state.research_running = False
    st.session_state.stop_research = False
    st.session_state.report_results = None
    st.session_state.current_query = ""
    st.session_state.query_input = ""
    st.session_state.processing_complete = False
    # Clear search data
    st.session_state.search_data = {}
    # Reset data saved flag only if we didn't have ratings to save
    if (
        not hasattr(st.session_state, "ratings_to_save")
        or not st.session_state.ratings_to_save
    ):
        st.session_state.data_saved_to_airtable = False

    # Clear PDF cache if PDF generation is enabled
    if ENABLE_PDF_GENERATION:
        if "pdf_bytes" in st.session_state:
            st.session_state.pdf_bytes = None
        if "fallback_report_text" in st.session_state:
            st.session_state.fallback_report_text = None

    reset_rating()
    # Clear the query and results
    return True


# Function to format search data for database storage
def prepare_search_data_for_db(search_data):
    """
    Format search data for future database storage.
    Returns a structured dictionary ready for database insertion.
    """
    formatted_data = {
        "user_id": f"{st.session_state.user_name}_{st.session_state.user_email}",  # Placeholder for actual user ID
        "query": search_data["query"],
        "timestamp": search_data["timestamp"],
        "search_queries": search_data["search_queries"],
        "num_results": sum(
            len(query_result["results"])
            for query_result in search_data["search_results"]
        ),
        "unique_urls": search_data.get("unique_urls", []),
        "detailed_results": [],
        "scraped_sources": search_data.get("scraped_content", []),
    }

    # Format the detailed results
    for query_result in search_data["search_results"]:
        for result in query_result["results"]:
            formatted_data["detailed_results"].append(
                {
                    "search_query": query_result["query"],
                    "url": result["url"],
                    "title": result["title"],
                }
            )

    return formatted_data


# Function to prepare comprehensive analytics data for database storage
def prepare_analytics_data_for_db(research_id):
    """
    Prepare comprehensive analytics data about the research process for database storage.
    Returns a structured dictionary with detailed metrics.
    """
    analytics = st.session_state.research_analytics.get(research_id, {})

    # Find the corresponding research entry
    research_entry = None
    for entry in st.session_state.research_history:
        if entry.get("id") == research_id:
            research_entry = entry
            break

    # Find the corresponding search data
    search_data = None
    for data in st.session_state.all_search_history:
        if data.get("research_id") == research_id:
            search_data = data
            break

    if not research_entry:
        return None

    # Create comprehensive analytics record
    analytics_data = {
        "research_id": research_id,
        "user_data": {
            "user_id": f"{st.session_state.user_name}_{st.session_state.user_email}",
            "user_name": st.session_state.user_name,
            "user_email": st.session_state.user_email,
            "user_registered": st.session_state.user_info_timestamp,
        },
        "query_data": {
            "original_query": research_entry.get("query", ""),
            "timestamp_started": research_entry.get("timestamp", ""),
            "timestamp_completed": analytics.get("completion_timestamp", ""),
            "total_time_seconds": analytics.get("total_time_seconds", 0),
            "status": research_entry.get("status", "unknown"),
        },
        "search_metrics": {
            "num_search_queries": len(analytics.get("search_queries", [])),
            "search_queries": analytics.get("search_queries", []),
            "total_results_found": analytics.get("total_results_found", 0),
            "unique_urls_found": len(analytics.get("unique_urls", [])),
        },
        "scraping_metrics": {
            "total_urls_attempted": analytics.get("total_urls_attempted", 0),
            "successful_scrapes": analytics.get("successful_scrapes", 0),
            "failed_scrapes": analytics.get("failed_scrapes", 0),
            "total_content_chars": analytics.get("total_content_chars", 0),
            "average_content_length": analytics.get("average_content_length", 0),
        },
        "processing_metrics": {
            "query_generation_time": analytics.get("query_generation_time", 0),
            "search_time": analytics.get("search_time", 0),
            "scraping_time": analytics.get("scraping_time", 0),
            "insight_extraction_time": analytics.get("insight_extraction_time", 0),
            "report_generation_time": analytics.get("report_generation_time", 0),
        },
        "result_metrics": {
            "num_citations": analytics.get("num_citations", 0),
            "report_content_length": analytics.get("report_content_length", 0),
            "rating": research_entry.get("rating"),
            "rating_feedback": research_entry.get("rating_feedback", ""),
        },
    }

    return analytics_data


# Function to start research
def start_research():
    # Only start if not already running
    if not st.session_state.research_running:
        st.session_state.research_running = True
        st.session_state.stop_research = False
        st.session_state.current_query = (
            st.session_state.query_input
        )  # Save from session state
        st.session_state.processing_complete = False

        # Reset rating for new research
        reset_rating()

        # Generate unique ID for this research
        search_id = f"search_{datetime.now().strftime('%Y%m%d%H%M%S')}_{st.session_state.user_name.replace(' ', '_')}"

        # Record start time for performance tracking
        start_time = datetime.now()
        start_time_iso = start_time.isoformat()

        # Use regular variables for analytics during processing
        # Only store in session state at the end
        research_analytics = {
            "start_time": start_time,
            "search_queries": [],
            "total_results_found": 0,
            "unique_urls": [],
            "total_urls_attempted": 0,
            "successful_scrapes": 0,
            "failed_scrapes": 0,
            "total_content_chars": 0,
            "query_generation_time": 0,
            "search_time": 0,
            "scraping_time": 0,
            "insight_extraction_time": 0,
            "report_generation_time": 0,
        }

        # Create status elements in the UI
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Add to research history
        research_entry = {
            "id": search_id,
            "query": st.session_state.query_input,
            "timestamp": start_time_iso,
            "status": "started",
            "rating": None,
            "rating_feedback": "",
        }
        st.session_state.research_history.append(research_entry)

        # Store the research data for later Airtable saving
        research_data = {
            "id": search_id,
            "user_id": st.session_state.user_airtable_id,
            "query": st.session_state.query_input,
            "timestamp": start_time_iso,
            "status": "started",
        }

        # Initialize search data dictionary for this query
        search_data = {
            "query": st.session_state.query_input,
            "timestamp": start_time_iso,
            "search_queries": [],
            "search_results": [],
            "research_id": search_id,  # Add the research ID to link with analytics
        }

        # Step 1: Generate search queries
        status_text.text("Generating search queries...")
        try:
            step_start_time = datetime.now()
            search_queries = generate_search_queries(st.session_state.query_input)
            progress_bar.progress(10)

            # Calculate time taken for this step
            query_generation_time = (datetime.now() - step_start_time).total_seconds()

            # Update analytics in local variable
            research_analytics["query_generation_time"] = query_generation_time
            research_analytics["search_queries"] = search_queries

            # Store search queries
            search_data["search_queries"] = search_queries

            # Log completion of step 1
            print(
                f"Step 1 complete: Generated {len(search_queries)} search queries for '{st.session_state.query_input}' in {query_generation_time:.2f} seconds"
            )

            # Check if stop button was clicked
            if st.session_state.stop_research:
                # Update status in analytics
                research_analytics["stop_reason"] = "user_stopped"
                # Store analytics in session state only at the end
                st.session_state.research_analytics[search_id] = research_analytics
                st.info("Research stopped by user.")
                st.session_state.research_running = False
                st.stop()

        except Exception as e:
            # Update analytics with error
            research_analytics["error"] = f"Error generating search queries: {str(e)}"
            research_analytics["stop_reason"] = "error"
            # Store analytics in session state only at the end
            st.session_state.research_analytics[search_id] = research_analytics

            st.error(f"Error generating search queries: {str(e)}")
            st.session_state.research_running = False
            st.stop()

        # Step 2: Search for information
        status_text.text("Searching for relevant information...")
        search_results = []
        try:
            step_start_time = datetime.now()
            total_results = 0

            for i, search_query in enumerate(search_queries):
                # Check if stop button was clicked
                if st.session_state.stop_research:
                    research_analytics["stop_reason"] = "user_stopped"
                    # Store analytics in session state only at the end
                    st.session_state.research_analytics[search_id] = research_analytics
                    st.info("Research stopped by user.")
                    st.session_state.research_running = False
                    st.stop()

                # Process the search query without showing expanders
                status_text.text(
                    f"Searching query {i+1}/{len(search_queries)}: {search_query}"
                )
                result = search_tavily(search_query)
                search_results.extend(result)
                total_results += len(result)

                # Store results for this search query
                query_results = []
                for item in result:
                    if "url" in item and "title" in item:
                        query_results.append(
                            {"url": item["url"], "title": item["title"]}
                        )

                # Add to search data
                search_data["search_results"].append(
                    {"query": search_query, "results": query_results}
                )

            # Calculate time taken for this step
            search_time = (datetime.now() - step_start_time).total_seconds()

            # Update analytics in local variable
            research_analytics["search_time"] = search_time
            research_analytics["total_results_found"] = total_results

            progress_bar.progress(30)

            # Log completion of step 2
            print(
                f"Step 2 complete: Found {total_results} search results across {len(search_queries)} queries in {search_time:.2f} seconds"
            )

        except Exception as e:
            # Update analytics with error
            research_analytics["error"] = f"Error searching with Tavily: {str(e)}"
            research_analytics["stop_reason"] = "error"
            # Store analytics in session state only at the end
            st.session_state.research_analytics[search_id] = research_analytics

            st.error(f"Error searching with Tavily: {str(e)}")
            if not os.getenv("TAVILY_API_KEY"):
                st.error(
                    "Please make sure your Tavily API key is set in the .env file."
                )
            st.session_state.research_running = False
            st.stop()

        # Extract unique URLs
        urls = list(
            set([item.get("url", "") for item in search_results if item.get("url")])
        )
        if not urls:
            # Update analytics with error
            research_analytics["error"] = "No URLs found from search results"
            research_analytics["stop_reason"] = "no_urls"
            # Store analytics in session state only at the end
            st.session_state.research_analytics[search_id] = research_analytics

            st.error("No URLs found from search results. Please try a different query.")
            st.session_state.research_running = False
            st.stop()

        print(f"Extracted {len(urls)} unique URLs from search results")

        # Store unique URLs in search data and analytics
        search_data["unique_urls"] = urls
        research_analytics["unique_urls"] = urls
        research_analytics["total_urls_attempted"] = len(urls[:20])  # We limit to 20

        # Step 3: Scrape content
        status_text.text("Studying Sources...")
        try:
            step_start_time = datetime.now()

            # Check if stop button was clicked
            if st.session_state.stop_research:
                research_analytics["stop_reason"] = "user_stopped"
                # Store analytics in session state only at the end
                st.session_state.research_analytics[search_id] = research_analytics
                st.info("Research stopped by user.")
                st.session_state.research_running = False
                st.stop()

            # Limit to top 20 URLs to avoid overloading
            scraped_content = scrape_content(urls[:20])
            if not scraped_content:
                # Update analytics with error
                research_analytics["error"] = "Could not scrape content from any URLs"
                research_analytics["stop_reason"] = "scrape_failed"
                # Store analytics in session state only at the end
                st.session_state.research_analytics[search_id] = research_analytics

                st.error("Could not scrape content from any of the URLs.")
                st.session_state.research_running = False
                st.stop()

            # Calculate scraping metrics
            successful_scrapes = len(scraped_content)
            failed_scrapes = len(urls[:20]) - successful_scrapes
            total_content_chars = sum(
                len(item.get("content", "")) for item in scraped_content
            )
            avg_content_length = (
                total_content_chars / successful_scrapes
                if successful_scrapes > 0
                else 0
            )

            # Update analytics in local variable
            research_analytics["scraping_time"] = (
                datetime.now() - step_start_time
            ).total_seconds()
            research_analytics["successful_scrapes"] = successful_scrapes
            research_analytics["failed_scrapes"] = failed_scrapes
            research_analytics["total_content_chars"] = total_content_chars
            research_analytics["average_content_length"] = avg_content_length

            # Store scraped content information in search data
            search_data["scraped_content"] = []
            for item in scraped_content:
                if "url" in item and "content" in item:
                    # Store metadata about scraped content, but not the full content (too large)
                    content_preview = (
                        item["content"][:200] + "..."
                        if len(item["content"]) > 200
                        else item["content"]
                    )
                    search_data["scraped_content"].append(
                        {
                            "url": item["url"],
                            "content_length": len(item["content"]),
                            "content_preview": content_preview,
                            "scraped_successfully": True,
                        }
                    )

            # Update search data in session state only at the end
            # st.session_state.search_data = search_data

            # Remove the "Scraping Methods Used" expander section
            progress_bar.progress(50)

            # Log completion of step 3
            print(
                f"Step 3 complete: Successfully scraped {successful_scrapes} sources out of {len(urls[:20])} URLs in {research_analytics['scraping_time']:.2f} seconds"
            )

        except Exception as e:
            # Update analytics with error
            research_analytics["error"] = f"Error scraping content: {str(e)}"
            research_analytics["stop_reason"] = "error"
            # Store analytics in session state only at the end
            st.session_state.research_analytics[search_id] = research_analytics

            st.error(f"Error scraping content: {str(e)}")
            st.session_state.research_running = False
            st.stop()

        # Step 4: Extract insights
        status_text.text("Extracting insights from content...")
        try:
            step_start_time = datetime.now()

            # Check if stop button was clicked
            if st.session_state.stop_research:
                research_analytics["stop_reason"] = "user_stopped"
                # Store analytics in session state only at the end
                st.session_state.research_analytics[search_id] = research_analytics
                st.info("Research stopped by user.")
                st.session_state.research_running = False
                st.stop()

            extraction_results = extract_insights(scraped_content)

            # Update analytics in local variable
            insight_extraction_time = (datetime.now() - step_start_time).total_seconds()
            research_analytics["insight_extraction_time"] = insight_extraction_time

            progress_bar.progress(75)

            # Log completion of step 4
            print(
                f"Step 4 complete: Successfully extracted insights from {len(scraped_content)} sources in {insight_extraction_time:.2f} seconds"
            )

        except Exception as e:
            # Update analytics with error
            research_analytics["error"] = f"Error extracting insights: {str(e)}"
            research_analytics["stop_reason"] = "error"
            # Store analytics in session state only at the end
            st.session_state.research_analytics[search_id] = research_analytics

            st.error(f"Error extracting insights: {str(e)}")
            st.session_state.research_running = False
            st.stop()

        # Step 5: Generate report
        status_text.text("Generating final report...")
        try:
            step_start_time = datetime.now()

            # Check if stop button was clicked
            if st.session_state.stop_research:
                research_analytics["stop_reason"] = "user_stopped"
                # Store analytics in session state only at the end
                st.session_state.research_analytics[search_id] = research_analytics
                st.info("Research stopped by user.")
                st.session_state.research_running = False
                st.stop()

            report_results = generate_report(
                st.session_state.query_input, extraction_results
            )

            # Calculate report generation time and overall completion time
            report_generation_time = (datetime.now() - step_start_time).total_seconds()
            completion_time = datetime.now()
            total_time_seconds = (
                completion_time - research_analytics["start_time"]
            ).total_seconds()

            # Update analytics in local variable
            research_analytics["report_generation_time"] = report_generation_time
            research_analytics["completion_timestamp"] = completion_time.isoformat()
            research_analytics["total_time_seconds"] = total_time_seconds
            research_analytics["num_citations"] = len(report_results["citations"])
            research_analytics["report_content_length"] = len(
                report_results["report_content"]
            )

            # Save report results to session state
            st.session_state.report_results = report_results
            st.session_state.processing_complete = True

            progress_bar.progress(100)

            # Log completion of step 5
            print(
                f"Step 5 complete: Successfully generated report with {len(report_results['citations'])} citations in {report_generation_time:.2f} seconds"
            )

        except Exception as e:
            # Update analytics with error
            research_analytics["error"] = f"Error generating report: {str(e)}"
            research_analytics["stop_reason"] = "error"
            # Store analytics in session state only at the end
            st.session_state.research_analytics[search_id] = research_analytics

            st.error(f"Error generating report: {str(e)}")
            st.session_state.research_running = False
            st.stop()

        # Clear the progress container
        progress_placeholder.empty()

        # Reset the research running state
        st.session_state.research_running = False

        # Update research history status
        if st.session_state.research_history:
            st.session_state.research_history[-1]["status"] = "completed"

        # Store all search data in session state at the end
        search_data["id"] = search_id
        st.session_state.search_data = search_data

        # Format for database and add to search history
        db_ready_data = prepare_search_data_for_db(search_data)
        db_ready_data["research_id"] = search_id

        # Add scraping metrics to the search data
        db_ready_data["successful_scrapes"] = research_analytics["successful_scrapes"]
        db_ready_data["failed_scrapes"] = research_analytics["failed_scrapes"]

        # Store in session state only at the end
        st.session_state.all_search_history.append(db_ready_data)

        # Store analytics in session state only at the end
        st.session_state.research_analytics[search_id] = research_analytics

        # Create analytics data ready for database
        analytics_data = prepare_analytics_data_for_db(search_id)
        if "analytics_history" not in st.session_state:
            st.session_state.analytics_history = []
        st.session_state.analytics_history.append(analytics_data)

        # Update the research data with completion information
        completion_time = datetime.now()
        completion_time_iso = completion_time.isoformat()
        total_time_seconds = (completion_time - start_time).total_seconds()

        # Update the current research data for later saving
        research_data["status"] = "completed"
        research_data["completion_timestamp"] = completion_time_iso
        research_data["total_time_seconds"] = total_time_seconds

        # Add the full report content to the research data
        if (
            st.session_state.report_results
            and "report_content" in st.session_state.report_results
        ):
            # Create a complete report with content and citations
            full_report = st.session_state.report_results["report_content"]

            # Add citations if available
            if "citations" in st.session_state.report_results:
                full_report += "\n\n## Citations\n"
                for citation in st.session_state.report_results["citations"]:
                    full_report += f"- {citation}\n"

            # Store the complete report
            research_data["full_report"] = full_report
            print(
                f"Added full report content with citations to research data (length: {len(research_data['full_report'])} characters)"
            )

        # Store in session state only at the end
        st.session_state.current_research_data = research_data

        # Now that the report is generated, save all data to Airtable at once
        # Enable Airtable operations for report completion
        st.session_state.skip_airtable_operations = False
        save_all_data_to_airtable()
        # Reset flag to skip Airtable operations again
        st.session_state.skip_airtable_operations = True
        st.session_state.data_saved_to_airtable = True

        # Update research history with reference to search data
        if st.session_state.research_history:
            st.session_state.research_history[-1]["search_data_id"] = search_id
            st.session_state.research_history[-1]["analytics_id"] = search_id

        print(
            f"Research process complete for query: '{st.session_state.query_input}' in {total_time_seconds:.2f} seconds"
        )
        print(
            f"Search data and analytics prepared for database storage with ID: {search_id}"
        )


# Display the final report if we have results - do this regardless of current research state
if st.session_state.report_results:
    with report_placeholder.container():
        st.header(f"Research Report: {st.session_state.current_query}")

        # Add table of contents
        st.markdown("## Table of Contents")
        st.markdown("1. [Abstract](#abstract)")
        st.markdown("2. [Introduction](#introduction)")
        st.markdown("3. [Main Findings](#main-findings)")
        st.markdown("4. [Discussion](#discussion)")
        st.markdown("5. [Conclusion](#conclusion)")
        st.markdown("6. [Citations](#citations)")

        # Display the report content
        st.markdown(st.session_state.report_results["report_content"])

        # Display citations
        st.markdown("## Citations")
        for citation in st.session_state.report_results["citations"]:
            st.markdown(f"- {citation}")

        # Rating system
        st.markdown("---")
        st.subheader("Rate this Report")

        # Star rating system
        st.write("Please rate the quality of this research report:")

        # Use radio buttons styled as stars
        rating_options = ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]
        rating_values = [1, 2, 3, 4, 5]

        # Create columns for better spacing
        col1, col2 = st.columns([3, 2])

        with col1:
            # Initialize with current rating if available
            default_index = (
                rating_values.index(st.session_state.current_rating)
                if st.session_state.current_rating in rating_values
                else 2
            )

            # Use horizontal radio buttons for stars
            selected_stars = st.radio(
                "Select your rating:",
                options=rating_options,
                index=default_index,
                horizontal=True,
                label_visibility="collapsed",
                key="rating_value",
                on_change=update_rating,
            )

            # Update current_rating when selected_stars changes
            if "rating_value" in st.session_state:
                rating_index = rating_options.index(st.session_state["rating_value"])
                st.session_state.current_rating = rating_values[rating_index]

            # Display the numeric rating
            st.write(f"Your rating: **{st.session_state.current_rating}/5**")

        # Feedback text area with key to maintain state
        feedback = st.text_area(
            "Additional feedback (optional)",
            value=st.session_state.rating_feedback,
            placeholder="What did you like or dislike about this report?",
            key="feedback_area",
        )
        st.session_state.rating_feedback = feedback

        # Submit button
        if st.button("Submit Rating", key="submit_rating_button"):
            submit_rating()

        # Show thank you message if rated
        if st.session_state.report_rated:
            st.success(
                f"Thank you for your rating: **{st.session_state.current_rating}/5** ⭐"
            )
            if st.session_state.rating_feedback:
                st.write(f"Your feedback: *{st.session_state.rating_feedback}*")

        # Generate PDF for download - only if enabled in configuration
        if ENABLE_PDF_GENERATION:
            try:
                # Only generate PDF if it hasn't been generated yet or if we don't have pdf_bytes in session state
                if (
                    "pdf_bytes" not in st.session_state
                    or st.session_state.pdf_bytes is None
                ):
                    print(
                        f"Starting PDF generation for report on: '{st.session_state.current_query}'"
                    )

                    # Store PDF in session state to avoid regenerating on each interaction
                    st.session_state.pdf_bytes = markdown_to_pdf(
                        f"Research Report: {st.session_state.current_query}",
                        st.session_state.report_results["report_content"],
                        st.session_state.report_results["citations"],
                    )

                    # Ensure pdf_bytes is bytes, not bytearray
                    if isinstance(st.session_state.pdf_bytes, bytearray):
                        st.session_state.pdf_bytes = bytes(st.session_state.pdf_bytes)

                    print(
                        f"PDF generation successful. PDF size: {len(st.session_state.pdf_bytes)} bytes"
                    )
                else:
                    print(
                        f"Using cached PDF, size: {len(st.session_state.pdf_bytes)} bytes"
                    )

                # Add download button for PDF - always active
                st.download_button(
                    label="Download Report as PDF",
                    data=st.session_state.pdf_bytes,
                    file_name="research_report.pdf",
                    mime="application/pdf",
                    key="download_pdf_button",
                )

            except Exception as e:
                print(f"Error in PDF generation: {str(e)}")
                st.error(f"Error generating PDF: {str(e)}")

                # Only generate fallback markdown if not already done
                if "fallback_report_text" not in st.session_state:
                    # Fallback to markdown if PDF generation fails
                    st.warning(
                        "PDF generation failed. Offering markdown download instead."
                    )
                    st.session_state.fallback_report_text = f"# Research Report: {st.session_state.current_query}\n\n{st.session_state.report_results['report_content']}\n\n## Citations\n"

                    for citation in st.session_state.report_results["citations"]:
                        st.session_state.fallback_report_text += f"- {citation}\n"

                    st.session_state.fallback_report_text += "\n\n## Disclaimer\nThis tool is still in development and the information provided should be crosschecked for accuracy."

                    print("Providing markdown download as fallback")

                st.download_button(
                    label="Download Report as Markdown",
                    data=st.session_state.fallback_report_text,
                    file_name="research_report.md",
                    mime="text/markdown",
                    key="download_md_button",
                )

        # Always add disclaimer to the webpage, regardless of PDF generation
        st.markdown("---")
        st.markdown(
            "**Disclaimer:** This tool is still in development and the information provided should be crosschecked for accuracy."
        )

elif st.session_state.stop_research:
    progress_placeholder.empty()
    progress_placeholder.info("Research stopped by user.")

# Display research history in sidebar
if st.session_state.research_history:
    st.sidebar.subheader("Research History")
    for i, entry in enumerate(
        st.session_state.research_history[-5:]
    ):  # Show last 5 entries
        timestamp = datetime.fromisoformat(entry["timestamp"])
        formatted_time = timestamp.strftime("%Y-%m-%d %H:%M")
        status_text = entry["status"]

        # Add rating to the display if available
        if entry.get("rating"):
            status_text += f" (Rated: {entry['rating']}/5)"

        st.sidebar.markdown(
            f"**{i+1}.** {entry['query'][:30]}... ({formatted_time}) - {status_text}"
        )

    if len(st.session_state.research_history) > 5:
        st.sidebar.markdown(
            f"*...and {len(st.session_state.research_history) - 5} more*"
        )

# Footer
st.markdown("---")
st.markdown("Insights Agent v0.0.1(Closed Beta) | Powered by Ability Labs")


# Function to stop research
def stop_research():
    st.session_state.stop_research = True
    st.session_state.research_running = False
    st.session_state.processing_complete = True

    # Update research history status
    if st.session_state.research_history:
        st.session_state.research_history[-1]["status"] = "stopped"

        # Mark the research as stopped in the current_research_data for later saving
        if hasattr(st.session_state, "current_research_data"):
            # Create a local copy of the research data
            research_data = st.session_state.current_research_data.copy()
            research_data["status"] = "stopped"

            # Add timestamp for when research was stopped
            stop_time = datetime.now()
            stop_time_iso = stop_time.isoformat()

            # Calculate total time if we have a start time
            if "timestamp" in research_data:
                start_time = datetime.fromisoformat(research_data["timestamp"])
                total_time_seconds = (stop_time - start_time).total_seconds()
                research_data["total_time_seconds"] = total_time_seconds

            research_data["stopped_timestamp"] = stop_time_iso

            # Update the session state with the modified data
            st.session_state.current_research_data = research_data

            # We'll store the data but not save to Airtable until report generation or rating
            print("Research stopped by user - data will be saved later")

    print("Research stopped by user")


# Create button columns for start and stop
col1, col2, col3, col4 = st.columns([1, 1, 1, 4])
with col1:
    start_button = st.button(
        "🚀 Start Research",
        on_click=start_research,
        disabled=not st.session_state.query_input,
    )
with col2:
    stop_button = st.button(
        "🛑 Stop",
        on_click=stop_research,
        disabled=not st.session_state.research_running,
    )
with col3:
    reset_button = st.button("🔄 Reset", on_click=reset_interface)

# Check if reset button was pressed
if st.session_state.reset_interface:
    # Clear the placeholders
    progress_placeholder.empty()
    report_placeholder.empty()
    # Reset the flag
    st.session_state.reset_interface = False
    # Stop execution to prevent showing any results
    st.stop()

# Process the query when start button is clicked and not yet processed
if (
    st.session_state.query_input
    and st.session_state.research_running
    and not st.session_state.stop_research
    and not st.session_state.processing_complete
):
    with progress_placeholder.container():
        # Create progress bar
        progress_bar = st.progress(0)
        st.subheader("Processing...")
        status_text = st.empty()

        # Update research history status
        if st.session_state.research_history:
            st.session_state.research_history[-1]["status"] = "processing"

        # Initialize search data dictionary for this query
        search_data = {
            "query": st.session_state.query_input,
            "timestamp": datetime.now().isoformat(),
            "search_queries": [],
            "search_results": [],
            "research_id": search_id,  # Add the research ID to link with analytics
        }
