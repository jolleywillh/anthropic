"""Configuration management for Teams Chat Summarization."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration."""

    # Azure AD / Teams Configuration
    AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
    AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
    AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")

    # Microsoft Graph API
    GRAPH_API_ENDPOINT = os.getenv("GRAPH_API_ENDPOINT", "https://graph.microsoft.com/v1.0")
    GRAPH_SCOPE = ["https://graph.microsoft.com/.default"]

    # Claude API
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    # Teams Chat
    TEAMS_CHAT_NAME = os.getenv("TEAMS_CHAT_NAME", "Energy & Market Pulse")
    TEAMS_CHAT_ID = os.getenv("TEAMS_CHAT_ID")  # Optional: specific chat ID

    @classmethod
    def validate(cls):
        """Validate that required configuration is present."""
        errors = []

        if not cls.AZURE_CLIENT_ID:
            errors.append("AZURE_CLIENT_ID is required")
        if not cls.AZURE_TENANT_ID:
            errors.append("AZURE_TENANT_ID is required")
        if not cls.AZURE_CLIENT_SECRET:
            errors.append("AZURE_CLIENT_SECRET is required")
        if not cls.ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY is required")

        if errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

        return True
