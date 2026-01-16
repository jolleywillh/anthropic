#!/usr/bin/env python3
"""Teams Chat Summarization Tool

This script fetches messages from a Microsoft Teams chat and generates
a summary using Claude API.
"""
import argparse
import sys
from datetime import datetime
from config import Config
from teams_client import TeamsClient
from summarizer import ChatSummarizer


def save_output(content: str, filename: str):
    """Save content to a file.

    Args:
        content: Content to save
        filename: Output filename
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Saved output to: {filename}")
    except Exception as e:
        print(f"✗ Error saving file: {e}")


def main():
    """Main entry point for the chat summarization tool."""
    parser = argparse.ArgumentParser(
        description="Summarize Microsoft Teams chat using Claude API"
    )
    parser.add_argument(
        "--chat-name",
        type=str,
        help="Name of the Teams chat to summarize (default: from .env)"
    )
    parser.add_argument(
        "--chat-id",
        type=str,
        help="Specific Teams chat ID (optional, will search by name if not provided)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days of chat history to fetch (default: 7)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of messages to fetch (default: 50)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save summary to file (optional)"
    )
    parser.add_argument(
        "--save-messages",
        type=str,
        help="Save raw messages to file (optional)"
    )
    parser.add_argument(
        "--custom-prompt",
        type=str,
        help="Custom instructions for summarization (optional)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Teams Chat Summarization Tool")
    print("=" * 60)
    print()

    try:
        # Validate configuration
        Config.validate()

        # Initialize clients
        print("Initializing...")
        teams_client = TeamsClient()
        summarizer = ChatSummarizer()
        print()

        # Fetch chat content
        print("Fetching chat messages...")
        chat_content = teams_client.get_chat_summary_content(
            chat_name=args.chat_name,
            chat_id=args.chat_id,
            days_back=args.days,
            limit=args.limit
        )
        print()

        # Save raw messages if requested
        if args.save_messages:
            save_output(chat_content, args.save_messages)
            print()

        # Generate summary
        print("Generating summary...")
        if args.custom_prompt:
            summary = summarizer.summarize_with_custom_prompt(
                chat_content=chat_content,
                custom_instructions=args.custom_prompt,
                model=args.model
            )
        else:
            summary = summarizer.summarize(
                chat_content=chat_content,
                model=args.model
            )
        print()

        # Output summary
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print()
        print(summary)
        print()
        print("=" * 60)

        # Save summary if requested
        if args.output:
            # Add metadata
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output_content = f"""Teams Chat Summary
Generated: {timestamp}
Chat: {args.chat_name or Config.TEAMS_CHAT_NAME}
Period: Last {args.days} days
Model: {args.model}

{'=' * 60}

{summary}
"""
            save_output(output_content, args.output)

        print()
        print("✓ Summarization complete!")

    except ValueError as e:
        print(f"\n✗ Configuration Error: {e}")
        print("\nPlease ensure you have:")
        print("  1. Created a .env file (copy from .env.example)")
        print("  2. Set all required credentials")
        print("  3. See README.md for setup instructions")
        sys.exit(1)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
