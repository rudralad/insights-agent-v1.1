import os
import requests
import json
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Any, Optional
import pathlib

# Load environment variables
load_dotenv()

# Airtable configuration
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
BASE_ID = os.getenv("AIRTABLE_BASE_ID")
USERS_TABLE_ID = os.getenv("AIRTABLE_USERS_TABLE_ID")
RESEARCH_TABLE_ID = os.getenv("AIRTABLE_RESEARCH_TABLE_ID")
SEARCH_DATA_TABLE_ID = os.getenv("AIRTABLE_SEARCH_DATA_TABLE_ID")
ANALYTICS_TABLE_ID = os.getenv("AIRTABLE_ANALYTICS_TABLE_ID")

# Airtable API base URL - using the standard API endpoint
AIRTABLE_API_URL = "https://api.airtable.com/v0"


# Headers for API requests
def get_headers():
    return {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json",
    }


def validate_airtable_config():
    """
    Validate Airtable configuration from environment variables

    Returns:
        bool: True if configuration is valid, False otherwise
    """
    global BASE_ID, USERS_TABLE_ID, RESEARCH_TABLE_ID, SEARCH_DATA_TABLE_ID, ANALYTICS_TABLE_ID

    # Verify base ID format
    if not BASE_ID:
        print("No Airtable base ID found in environment variables")
        print("Please add AIRTABLE_BASE_ID to your .env file")
        return False

    if not BASE_ID.startswith("app"):
        print(f"Invalid base ID format: {BASE_ID}. Base ID should start with 'app'")
        return False

    # Check if all table IDs are present
    missing_tables = []
    if not USERS_TABLE_ID:
        missing_tables.append("AIRTABLE_USERS_TABLE_ID")
    if not RESEARCH_TABLE_ID:
        missing_tables.append("AIRTABLE_RESEARCH_TABLE_ID")
    if not SEARCH_DATA_TABLE_ID:
        missing_tables.append("AIRTABLE_SEARCH_DATA_TABLE_ID")
    if not ANALYTICS_TABLE_ID:
        missing_tables.append("AIRTABLE_ANALYTICS_TABLE_ID")

    if missing_tables:
        print(
            f"Missing table IDs in environment variables: {', '.join(missing_tables)}"
        )
        print("\nPlease add the following to your .env file:")
        for missing_table in missing_tables:
            print(f"{missing_table}=your_{missing_table.lower()}_here")
        return False

    print(f"Loaded base ID: {BASE_ID}")
    print(
        f"Loaded table IDs: Users={USERS_TABLE_ID}, Research={RESEARCH_TABLE_ID}, Search={SEARCH_DATA_TABLE_ID}, Analytics={ANALYTICS_TABLE_ID}"
    )

    # Verify that we can connect to the Airtable base
    try:
        # Try to access one of the tables directly
        url = f"{AIRTABLE_API_URL}/{BASE_ID}/{USERS_TABLE_ID}"
        print(f"Verifying base access with URL: {url}")

        headers = get_headers()

        # Try to get just one record to verify access
        params = {"maxRecords": 1}
        response = requests.get(url, headers=headers, params=params)
        print(f"Base verification response code: {response.status_code}")

        if response.status_code != 200:
            print(f"Could not access Airtable table: {response.text}")
            print("This could be due to:")
            print("1. Invalid API key")
            print("2. API key doesn't have permission to access this base")
            print("3. Base ID or table ID is incorrect")
            return False

        print(f"Successfully connected to Airtable base")
        return True
    except Exception as e:
        print(f"Error verifying Airtable configuration: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def initialize_airtable() -> bool:
    """
    Initialize the Airtable connection.

    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global AIRTABLE_API_KEY, BASE_ID, USERS_TABLE_ID, RESEARCH_TABLE_ID, SEARCH_DATA_TABLE_ID, ANALYTICS_TABLE_ID

    print("\n" + "=" * 80)
    print("INITIALIZING AIRTABLE CONNECTION")
    print("=" * 80)

    # Check if Airtable is temporarily disabled
    if os.path.exists("DISABLE_AIRTABLE"):
        print("Airtable integration temporarily disabled by DISABLE_AIRTABLE file")
        return False

    try:
        # Load API key from .env file
        load_dotenv()
        AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")

        if not AIRTABLE_API_KEY:
            print("No Airtable API key found in .env file")
            print(
                "Please add your Airtable API key to the .env file as AIRTABLE_API_KEY=your_key_here"
            )
            return False
        else:
            print(
                f"Found Airtable API key in .env file: {'*' * (len(AIRTABLE_API_KEY) - 4) + AIRTABLE_API_KEY[-4:]}"
            )

        # Test API key validity by checking user info
        auth_test_url = "https://api.airtable.com/v0/meta/whoami"
        auth_headers = {
            "Authorization": f"Bearer {AIRTABLE_API_KEY}",
            "Content-Type": "application/json",
        }

        print(f"Testing API key validity with URL: {auth_test_url}")

        try:
            auth_response = requests.get(auth_test_url, headers=auth_headers)
            print(f"API key validation response code: {auth_response.status_code}")

            if auth_response.status_code != 200:
                print(f"API key validation failed: {auth_response.text}")
                print(
                    "Please check that your Airtable API key is valid and has the correct permissions."
                )
                return False
            else:
                user_info = auth_response.json()
                print(
                    f"Airtable API key validated. Connected as: {user_info.get('name', 'Unknown user')}"
                )
        except Exception as auth_error:
            print(f"Error validating API key: {str(auth_error)}")
            return False

        # Validate the Airtable configuration
        if not validate_airtable_config():
            print("\n" + "=" * 80)
            print("AIRTABLE SETUP REQUIRED")
            print("=" * 80)
            print("\nTo set up Airtable integration, follow these steps:")
            print("1. Create a new base in Airtable through the web interface")
            print(
                "2. Create the required tables (Users, Research Sessions, Search Data, Analytics)"
            )
            print(
                "3. Copy the base ID from the URL (it looks like 'appXXXXXXXXXXXXXX')"
            )
            print(
                "4. Copy the table IDs from the URLs (they look like 'tblXXXXXXXXXXXXXX')"
            )
            print("5. Add these IDs to your .env file as follows:")
            print("   AIRTABLE_BASE_ID=appXXXXXXXXXXXXXX")
            print("   AIRTABLE_USERS_TABLE_ID=tblXXXXXXXXXXXXXX")
            print("   AIRTABLE_RESEARCH_TABLE_ID=tblXXXXXXXXXXXXXX")
            print("   AIRTABLE_SEARCH_DATA_TABLE_ID=tblXXXXXXXXXXXXXX")
            print("   AIRTABLE_ANALYTICS_TABLE_ID=tblXXXXXXXXXXXXXX")
            print("6. Restart the application\n")
            print(
                "The app will continue to function without Airtable integration for now."
            )
            print("=" * 80 + "\n")
            return False

        return True

    except Exception as e:
        print(f"Error initializing Airtable: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def get_or_create_user(name: str, email: str, timestamp: str) -> str:
    """
    Get a user record by email or create it if it doesn't exist.

    Args:
        name: User's full name
        email: User's email
        timestamp: Registration timestamp

    Returns:
        User record ID
    """
    # First check if user exists
    url = f"{AIRTABLE_API_URL}/{BASE_ID}/{USERS_TABLE_ID}"
    params = {"filterByFormula": f"{{Email}} = '{email}'"}

    response = requests.get(url, headers=get_headers(), params=params)

    if response.status_code != 200:
        raise Exception(f"Failed to query users: {response.text}")

    data = response.json()

    # If user exists, return their ID
    if data.get("records"):
        return data["records"][0]["id"]

    # Otherwise create a new user
    payload = {
        "records": [
            {
                "fields": {
                    "Name": name,
                    "Email": email,
                    "Registration Timestamp": timestamp,
                }
            }
        ]
    }

    response = requests.post(url, headers=get_headers(), json=payload)

    if response.status_code != 200:
        raise Exception(f"Failed to create user: {response.text}")

    return response.json()["records"][0]["id"]


def create_research_session(research_data: Dict[str, Any]) -> str:
    """
    Create a new research session record.

    Args:
        research_data: Dictionary containing research session data

    Returns:
        Research session record ID
    """
    print(f"Creating research session with data: {research_data}")
    url = f"{AIRTABLE_API_URL}/{BASE_ID}/{RESEARCH_TABLE_ID}"

    # Format the data for Airtable
    fields = {
        "Research ID": research_data.get("id"),
        "User": research_data.get("user_id"),
        "Original Query": research_data.get("query"),
        "Timestamp Started": research_data.get("timestamp"),
        "Status": research_data.get("status", "started"),
    }

    # Add full report if available
    if "full_report" in research_data:
        fields["Full Report"] = research_data["full_report"]
        print(
            f"Including full report in new record (length: {len(fields['Full Report'])} characters)"
        )

    payload = {"records": [{"fields": fields}]}

    print(f"Research session payload fields: {list(fields.keys())}")

    response = requests.post(url, headers=get_headers(), json=payload)

    print(f"Create research session response status: {response.status_code}")

    if response.status_code != 200:
        print(f"Failed to create research session: {response.text}")
        raise Exception(f"Failed to create research session: {response.text}")

    record_id = response.json()["records"][0]["id"]
    print(f"Research session created with Airtable ID: {record_id}")

    # Verify the record was created correctly
    try:
        verify_url = f"{AIRTABLE_API_URL}/{BASE_ID}/{RESEARCH_TABLE_ID}/{record_id}"
        verify_response = requests.get(verify_url, headers=get_headers())

        if verify_response.status_code == 200:
            record_data = verify_response.json()
            research_id = record_data.get("fields", {}).get("Research ID")
            print(
                f"Verified research session creation. Research ID in Airtable: {research_id}"
            )
        else:
            print(
                f"Warning: Could not verify research session creation: {verify_response.status_code}"
            )
    except Exception as e:
        print(f"Warning: Error verifying research session creation: {str(e)}")

    return record_id


def update_research_session(research_id: str, update_data: Dict[str, Any]) -> None:
    """
    Update an existing research session record.

    Args:
        research_id: The ID of the research session to update
        update_data: Dictionary containing fields to update
    """
    print(f"Updating research session with ID: {research_id}")
    print(f"Update data: {update_data}")

    # First, we need to find the Airtable record ID for this research ID
    url = f"{AIRTABLE_API_URL}/{BASE_ID}/{RESEARCH_TABLE_ID}"

    # Use SEARCH() formula to handle potential special characters in research_id
    formula = f"SEARCH('{research_id}', {{Research ID}}) > 0"
    params = {"filterByFormula": formula}

    print(f"Looking up research session with formula: {formula}")

    response = requests.get(url, headers=get_headers(), params=params)

    print(f"Lookup response status: {response.status_code}")

    if response.status_code != 200:
        raise Exception(f"Failed to query research session: {response.text}")

    data = response.json()

    print(f"Found {len(data.get('records', []))} matching records")

    if not data.get("records"):
        # Try a more direct approach if the SEARCH formula didn't work
        print("No records found with SEARCH formula, trying exact match...")
        params = {"filterByFormula": f"{{Research ID}} = '{research_id}'"}
        response = requests.get(url, headers=get_headers(), params=params)

        if response.status_code != 200:
            raise Exception(f"Failed to query research session: {response.text}")

        data = response.json()
        print(f"Found {len(data.get('records', []))} matching records with exact match")

        if not data.get("records"):
            # List all research sessions for debugging
            print("Listing all research sessions for debugging...")
            all_response = requests.get(url, headers=get_headers())
            if all_response.status_code == 200:
                all_data = all_response.json()
                all_records = all_data.get("records", [])
                print(f"Total research sessions in Airtable: {len(all_records)}")
                for record in all_records[:5]:  # Show first 5 for debugging
                    print(
                        f"Record ID: {record.get('id')}, Research ID: {record.get('fields', {}).get('Research ID', 'N/A')}"
                    )

            raise Exception(f"Research session with ID {research_id} not found")

    record_id = data["records"][0]["id"]
    print(f"Found Airtable record ID: {record_id}")

    # Now update the record
    url = f"{AIRTABLE_API_URL}/{BASE_ID}/{RESEARCH_TABLE_ID}/{record_id}"

    # Format the update data
    fields = {}
    if "status" in update_data:
        fields["Status"] = update_data["status"]
    if "rating" in update_data:
        # Ensure rating is an integer or float
        try:
            fields["Rating"] = float(update_data["rating"])
            print(f"Setting Rating field to: {fields['Rating']}")
        except (ValueError, TypeError):
            print(f"Warning: Invalid rating value: {update_data['rating']}")
            # Use a default value if conversion fails
            fields["Rating"] = 3
    if "rating_feedback" in update_data:
        fields["Rating Feedback"] = update_data["rating_feedback"]
    if "rating_timestamp" in update_data:
        fields["Rating Timestamp"] = update_data["rating_timestamp"]
    if "completion_timestamp" in update_data:
        fields["Timestamp Completed"] = update_data["completion_timestamp"]
    if "total_time_seconds" in update_data:
        fields["Total Time (seconds)"] = update_data["total_time_seconds"]
    if "full_report" in update_data:
        # Try both field name options
        fields["Full Report"] = update_data["full_report"]
        print(
            f"Adding Full Report to Airtable (length: {len(fields['Full Report'])} characters)"
        )
        # For debugging
        print(f"Research table ID being used: {RESEARCH_TABLE_ID}")
        print(
            f"Airtable API URL: {AIRTABLE_API_URL}/{BASE_ID}/{RESEARCH_TABLE_ID}/{record_id}"
        )

    payload = {"fields": fields}
    print(f"Update payload fields: {list(fields.keys())}")

    # For debugging, let's try to get the table structure first
    try:
        table_info_url = f"{AIRTABLE_API_URL}/{BASE_ID}/{RESEARCH_TABLE_ID}"
        info_response = requests.get(
            table_info_url, headers=get_headers(), params={"maxRecords": 1}
        )

        if info_response.status_code == 200:
            sample_record = info_response.json().get("records", [{}])[0]
            if "fields" in sample_record:
                print(
                    f"Available fields in table: {list(sample_record['fields'].keys())}"
                )
        else:
            print(
                f"Failed to get table structure: {info_response.status_code} - {info_response.text}"
            )
    except Exception as e:
        print(f"Error checking table structure: {str(e)}")

    # Now update the record
    response = requests.patch(url, headers=get_headers(), json=payload)

    print(f"Update response status: {response.status_code}")

    if response.status_code != 200:
        print(f"Failed to update research session: {response.text}")
        raise Exception(f"Failed to update research session: {response.text}")
    else:
        print(f"Successfully updated research session with ID: {research_id}")
        print(f"Updated fields: {', '.join(fields.keys())}")


def create_search_data(search_data: Dict[str, Any]) -> str:
    """
    Create a new search data record.

    Args:
        search_data: Dictionary containing search data

    Returns:
        Search data record ID
    """
    url = f"{AIRTABLE_API_URL}/{BASE_ID}/{SEARCH_DATA_TABLE_ID}"

    # Format the search queries as a string
    search_queries_str = "\n".join(search_data.get("search_queries", []))

    # Format unique URLs as a string
    urls_str = "\n".join(search_data.get("unique_urls", []))

    # Format the data for Airtable
    payload = {
        "records": [
            {
                "fields": {
                    "Research ID": search_data.get("research_id"),
                    "Search Queries": search_queries_str,
                    "Total Results": search_data.get("num_results", 0),
                    "Unique URLs": len(search_data.get("unique_urls", [])),
                    "URLs List": urls_str,
                    "Successful Scrapes": search_data.get("successful_scrapes", 0),
                    "Failed Scrapes": search_data.get("failed_scrapes", 0),
                    "Timestamp": search_data.get("timestamp"),
                }
            }
        ]
    }

    response = requests.post(url, headers=get_headers(), json=payload)

    if response.status_code != 200:
        raise Exception(f"Failed to create search data: {response.text}")

    return response.json()["records"][0]["id"]


def create_analytics_data(analytics_data: Dict[str, Any]) -> str:
    """
    Create a new analytics data record.

    Args:
        analytics_data: Dictionary containing analytics data

    Returns:
        Analytics data record ID
    """
    url = f"{AIRTABLE_API_URL}/{BASE_ID}/{ANALYTICS_TABLE_ID}"

    # Extract processing metrics
    processing_metrics = analytics_data.get("processing_metrics", {})

    # Extract scraping metrics
    scraping_metrics = analytics_data.get("scraping_metrics", {})

    # Extract result metrics
    result_metrics = analytics_data.get("result_metrics", {})

    # Format the data for Airtable
    payload = {
        "records": [
            {
                "fields": {
                    "Research ID": analytics_data.get("research_id"),
                    "Query Generation Time": processing_metrics.get(
                        "query_generation_time", 0
                    ),
                    "Search Time": processing_metrics.get("search_time", 0),
                    "Scraping Time": processing_metrics.get("scraping_time", 0),
                    "Insight Extraction Time": processing_metrics.get(
                        "insight_extraction_time", 0
                    ),
                    "Report Generation Time": processing_metrics.get(
                        "report_generation_time", 0
                    ),
                    "Total Content Chars": scraping_metrics.get(
                        "total_content_chars", 0
                    ),
                    "Average Content Length": scraping_metrics.get(
                        "average_content_length", 0
                    ),
                    "Number of Citations": result_metrics.get("num_citations", 0),
                    "Report Content Length": result_metrics.get(
                        "report_content_length", 0
                    ),
                }
            }
        ]
    }

    response = requests.post(url, headers=get_headers(), json=payload)

    if response.status_code != 200:
        raise Exception(f"Failed to create analytics data: {response.text}")

    return response.json()["records"][0]["id"]
