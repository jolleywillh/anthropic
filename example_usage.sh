#!/bin/bash
# Example usage scripts for Teams Chat Summarization

# Make sure you're in the virtual environment
# source venv/bin/activate

echo "Teams Chat Summarization - Example Usage"
echo "========================================"
echo ""

# Example 1: Basic summary
echo "Example 1: Basic summary of default chat"
echo "Command: python main.py"
echo ""
# Uncomment to run:
# python main.py

# Example 2: Last 24 hours with output file
echo "Example 2: Daily summary (last 24 hours)"
echo "Command: python main.py --days 1 --output daily-summary.txt"
echo ""
# Uncomment to run:
# python main.py --days 1 --output daily-summary.txt

# Example 3: Weekly summary with more messages
echo "Example 3: Weekly summary with up to 100 messages"
echo "Command: python main.py --days 7 --limit 100 --output weekly-summary.txt"
echo ""
# Uncomment to run:
# python main.py --days 7 --limit 100 --output weekly-summary.txt

# Example 4: Custom prompt for action items
echo "Example 4: Extract action items and decisions"
echo "Command: python main.py --custom-prompt 'list all action items, decisions, and responsible parties' --output actions.txt"
echo ""
# Uncomment to run:
# python main.py --custom-prompt "list all action items, decisions, and responsible parties" --output actions.txt

# Example 5: Save both messages and summary
echo "Example 5: Save raw messages and summary"
echo "Command: python main.py --save-messages raw-messages.txt --output summary.txt"
echo ""
# Uncomment to run:
# python main.py --save-messages raw-messages.txt --output summary.txt

# Example 6: Different chat
echo "Example 6: Summarize a different chat"
echo "Command: python main.py --chat-name 'Your Other Chat Name' --output other-chat-summary.txt"
echo ""
# Uncomment to run:
# python main.py --chat-name "Your Other Chat Name" --output other-chat-summary.txt

echo ""
echo "To run any example, uncomment the corresponding line in this script"
echo "or copy the command and run it directly."
