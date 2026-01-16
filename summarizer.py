"""Chat summarization using Claude API."""
from anthropic import Anthropic
from config import Config


class ChatSummarizer:
    """Summarize chat content using Claude API."""

    def __init__(self):
        """Initialize the summarizer with Claude API."""
        self.config = Config()
        self.client = Anthropic(api_key=self.config.ANTHROPIC_API_KEY)
        print("✓ Claude API client initialized")

    def summarize(
        self,
        chat_content: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2048
    ) -> str:
        """Summarize the provided chat content.

        Args:
            chat_content: The formatted chat messages to summarize
            model: Claude model to use for summarization
            max_tokens: Maximum tokens in the response

        Returns:
            Summarized content as a string
        """
        if not chat_content or chat_content == "No messages found.":
            return "No messages available to summarize."

        print(f"Generating summary using {model}...")

        # Create the summarization prompt
        system_prompt = """You are an expert at analyzing and summarizing Teams chat conversations.
Your goal is to provide clear, actionable summaries that help people quickly understand:
1. Key topics and themes discussed
2. Important decisions made
3. Action items or next steps
4. Critical information or updates shared
5. Any concerns or issues raised

Format your summary in a clear, structured way with appropriate headings."""

        user_prompt = f"""Please provide a comprehensive summary of the following Teams chat conversation from "{self.config.TEAMS_CHAT_NAME}":

{chat_content}

Provide a well-structured summary that captures the key points, decisions, and action items."""

        try:
            # Call Claude API
            message = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )

            summary = message.content[0].text
            print("✓ Summary generated successfully")
            return summary

        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            print(f"✗ {error_msg}")
            raise Exception(error_msg)

    def summarize_with_custom_prompt(
        self,
        chat_content: str,
        custom_instructions: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2048
    ) -> str:
        """Summarize chat with custom instructions.

        Args:
            chat_content: The formatted chat messages to summarize
            custom_instructions: Custom instructions for the summary
            model: Claude model to use
            max_tokens: Maximum tokens in the response

        Returns:
            Summarized content based on custom instructions
        """
        if not chat_content or chat_content == "No messages found.":
            return "No messages available to summarize."

        print(f"Generating custom summary using {model}...")

        user_prompt = f"""Please analyze the following Teams chat conversation and {custom_instructions}

Chat content:
{chat_content}"""

        try:
            message = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )

            summary = message.content[0].text
            print("✓ Custom summary generated successfully")
            return summary

        except Exception as e:
            error_msg = f"Error generating custom summary: {str(e)}"
            print(f"✗ {error_msg}")
            raise Exception(error_msg)
