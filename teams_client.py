"""Microsoft Teams client for fetching chat messages."""
import requests
from msal import ConfidentialClientApplication
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from config import Config


class TeamsClient:
    """Client for interacting with Microsoft Teams via Graph API."""

    def __init__(self):
        """Initialize the Teams client with authentication."""
        self.config = Config()
        self.access_token = None
        self._authenticate()

    def _authenticate(self):
        """Authenticate with Azure AD to get access token."""
        app = ConfidentialClientApplication(
            client_id=self.config.AZURE_CLIENT_ID,
            client_credential=self.config.AZURE_CLIENT_SECRET,
            authority=f"https://login.microsoftonline.com/{self.config.AZURE_TENANT_ID}"
        )

        # Acquire token for Microsoft Graph
        result = app.acquire_token_for_client(scopes=self.config.GRAPH_SCOPE)

        if "access_token" in result:
            self.access_token = result["access_token"]
            print("✓ Successfully authenticated with Microsoft Graph API")
        else:
            error_msg = result.get("error_description", result.get("error", "Unknown error"))
            raise Exception(f"Failed to acquire access token: {error_msg}")

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make an authenticated request to the Graph API."""
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

        url = f"{self.config.GRAPH_API_ENDPOINT}/{endpoint}"
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")

    def find_chat_by_name(self, chat_name: str) -> Optional[str]:
        """Find a chat by its topic/name.

        Args:
            chat_name: The name/topic of the chat to find

        Returns:
            The chat ID if found, None otherwise
        """
        print(f"Searching for chat: '{chat_name}'...")

        # Get all chats for the user
        # Note: This requires delegated permissions or specific application permissions
        try:
            result = self._make_request("chats", params={"$filter": f"topic eq '{chat_name}'"})
            chats = result.get("value", [])

            if chats:
                chat_id = chats[0]["id"]
                print(f"✓ Found chat with ID: {chat_id}")
                return chat_id
            else:
                print(f"✗ No chat found with name '{chat_name}'")
                return None
        except Exception as e:
            print(f"✗ Error searching for chat: {e}")
            # If filtering doesn't work, try listing all chats
            print("Attempting to list all chats...")
            result = self._make_request("chats")
            chats = result.get("value", [])

            for chat in chats:
                if chat.get("topic") == chat_name:
                    chat_id = chat["id"]
                    print(f"✓ Found chat with ID: {chat_id}")
                    return chat_id

            print(f"✗ No chat found with name '{chat_name}'")
            return None

    def get_chat_messages(
        self,
        chat_id: str,
        days_back: int = 7,
        limit: int = 50
    ) -> List[Dict]:
        """Get messages from a specific chat.

        Args:
            chat_id: The ID of the chat
            days_back: Number of days of history to fetch
            limit: Maximum number of messages to retrieve

        Returns:
            List of message dictionaries
        """
        print(f"Fetching messages from the last {days_back} days (max {limit} messages)...")

        # Calculate the date filter
        since_date = datetime.utcnow() - timedelta(days=days_back)
        date_filter = since_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "$top": limit,
            "$orderby": "createdDateTime desc",
            "$filter": f"createdDateTime gt {date_filter}"
        }

        try:
            result = self._make_request(f"chats/{chat_id}/messages", params=params)
            messages = result.get("value", [])
            print(f"✓ Retrieved {len(messages)} messages")
            return messages
        except Exception as e:
            print(f"✗ Error fetching messages: {e}")
            # Try without filter if it fails
            print("Retrying without date filter...")
            params = {
                "$top": limit,
                "$orderby": "createdDateTime desc"
            }
            result = self._make_request(f"chats/{chat_id}/messages", params=params)
            messages = result.get("value", [])
            print(f"✓ Retrieved {len(messages)} messages")
            return messages

    def format_messages(self, messages: List[Dict]) -> str:
        """Format messages into a readable text format.

        Args:
            messages: List of message dictionaries from Graph API

        Returns:
            Formatted string of messages
        """
        if not messages:
            return "No messages found."

        formatted_lines = []
        formatted_lines.append(f"=== {self.config.TEAMS_CHAT_NAME} ===\n")

        # Reverse to show chronological order (oldest first)
        for msg in reversed(messages):
            sender = msg.get("from", {}).get("user", {}).get("displayName", "Unknown")
            timestamp = msg.get("createdDateTime", "")

            # Parse timestamp for better formatting
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    timestamp_str = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    timestamp_str = timestamp

            # Get message body
            body = msg.get("body", {}).get("content", "")

            # Clean up HTML if present (basic cleanup)
            if body:
                # Remove common HTML tags (basic approach)
                import re
                body = re.sub(r'<[^>]+>', '', body)
                body = body.strip()

            if body:  # Only include messages with content
                formatted_lines.append(f"[{timestamp_str}] {sender}:")
                formatted_lines.append(f"{body}\n")

        return "\n".join(formatted_lines)

    def get_chat_summary_content(
        self,
        chat_name: Optional[str] = None,
        chat_id: Optional[str] = None,
        days_back: int = 7,
        limit: int = 50
    ) -> str:
        """Get formatted chat content ready for summarization.

        Args:
            chat_name: Name of the chat to summarize (uses config default if not provided)
            chat_id: Specific chat ID (optional, will search by name if not provided)
            days_back: Number of days of history to fetch
            limit: Maximum number of messages

        Returns:
            Formatted string of chat messages
        """
        # Use provided chat_id, or search for it by name
        if not chat_id:
            search_name = chat_name or self.config.TEAMS_CHAT_NAME
            chat_id = self.find_chat_by_name(search_name)

            if not chat_id:
                raise ValueError(f"Could not find chat: {search_name}")

        # Get messages
        messages = self.get_chat_messages(chat_id, days_back, limit)

        # Format messages
        return self.format_messages(messages)
