# Teams Chat Summarization Tool

A Python tool that fetches messages from Microsoft Teams chats and generates intelligent summaries using Claude API.

## Features

- Fetch messages from any Microsoft Teams chat
- Search for chats by name or use specific chat IDs
- Configurable time range and message limits
- Intelligent summarization using Claude AI
- Support for custom summarization prompts
- Save summaries and raw messages to files
- Command-line interface for easy automation

## Prerequisites

Before you begin, you'll need:

1. **Microsoft Azure Account** with access to Azure Active Directory
2. **Microsoft Teams** with access to the chat you want to summarize
3. **Anthropic API Key** for Claude API access
4. **Python 3.8+** installed on your system

## Setup Instructions

### Step 1: Azure AD App Registration

To access Teams chats via Microsoft Graph API, you need to create an Azure AD application:

1. **Go to Azure Portal**
   - Navigate to https://portal.azure.com
   - Sign in with your Microsoft account

2. **Create App Registration**
   - Go to "Azure Active Directory" > "App registrations"
   - Click "New registration"
   - Enter a name (e.g., "Teams Chat Summarizer")
   - For "Supported account types", select "Accounts in this organizational directory only"
   - Click "Register"

3. **Note Your Application IDs**
   - After registration, copy the following values (you'll need them later):
     - **Application (client) ID** → `AZURE_CLIENT_ID`
     - **Directory (tenant) ID** → `AZURE_TENANT_ID`

4. **Create a Client Secret**
   - In your app registration, go to "Certificates & secrets"
   - Click "New client secret"
   - Add a description (e.g., "Chat Summarizer Secret")
   - Choose an expiration period
   - Click "Add"
   - **IMPORTANT**: Copy the secret **Value** immediately → `AZURE_CLIENT_SECRET`
   - (You won't be able to see it again!)

5. **Configure API Permissions**
   - Go to "API permissions" in your app registration
   - Click "Add a permission"
   - Select "Microsoft Graph"
   - Select "Application permissions" (not Delegated)
   - Add the following permissions:
     - `Chat.Read.All` - Read all chat messages
     - `Chat.ReadBasic.All` - Read names and members of all chat threads
   - Click "Add permissions"
   - **IMPORTANT**: Click "Grant admin consent" and confirm
     - (This requires admin privileges in your organization)
     - If you don't have admin rights, ask your IT admin to grant consent

### Step 2: Get Anthropic API Key

1. **Sign up for Anthropic**
   - Go to https://console.anthropic.com
   - Create an account or sign in

2. **Create an API Key**
   - Navigate to "API Keys" in the console
   - Click "Create Key"
   - Copy your API key → `ANTHROPIC_API_KEY`
   - **IMPORTANT**: Store this securely, you won't see it again

### Step 3: Install the Tool

1. **Clone or download this repository**
   ```bash
   cd /path/to/teams-chat-summarizer
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Step 4: Configure Environment Variables

1. **Copy the example environment file**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your credentials**
   ```bash
   nano .env  # or use your preferred editor
   ```

3. **Fill in all required values:**
   ```env
   # From Azure App Registration (Step 1)
   AZURE_CLIENT_ID=your-client-id-here
   AZURE_TENANT_ID=your-tenant-id-here
   AZURE_CLIENT_SECRET=your-client-secret-here

   # From Anthropic Console (Step 2)
   ANTHROPIC_API_KEY=your-anthropic-api-key-here

   # Your Teams chat name
   TEAMS_CHAT_NAME=Energy & Market Pulse
   ```

4. **Save the file**

## Usage

### Basic Usage

Summarize the default chat (configured in `.env`):

```bash
python main.py
```

### Advanced Options

**Specify a different chat:**
```bash
python main.py --chat-name "Project Updates"
```

**Fetch more history:**
```bash
python main.py --days 14 --limit 100
```

**Save summary to file:**
```bash
python main.py --output summary.txt
```

**Save raw messages:**
```bash
python main.py --save-messages messages.txt --output summary.txt
```

**Use a specific chat ID:**
```bash
python main.py --chat-id "19:meeting_XXXXX"
```

**Custom summarization prompt:**
```bash
python main.py --custom-prompt "extract all action items and assign them to specific people"
```

**Use a different Claude model:**
```bash
python main.py --model "claude-opus-4-20250514"
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--chat-name` | Name of the Teams chat to summarize | From `.env` |
| `--chat-id` | Specific Teams chat ID | None (searches by name) |
| `--days` | Number of days of history to fetch | 7 |
| `--limit` | Maximum number of messages to fetch | 50 |
| `--model` | Claude model to use | claude-sonnet-4-20250514 |
| `--output` | Save summary to file | None (prints to console) |
| `--save-messages` | Save raw messages to file | None |
| `--custom-prompt` | Custom instructions for summarization | None |

## Examples

### Example 1: Daily Team Summary

```bash
python main.py --days 1 --output daily-summary-$(date +%Y-%m-%d).txt
```

### Example 2: Weekly Report

```bash
python main.py --days 7 --limit 200 --output weekly-report.txt
```

### Example 3: Extract Action Items

```bash
python main.py --custom-prompt "list all action items, decisions, and who is responsible for each task" --output actions.txt
```

### Example 4: Multiple Chats

```bash
python main.py --chat-name "Energy & Market Pulse" --output energy-summary.txt
python main.py --chat-name "Project Updates" --output project-summary.txt
```

## Automation

You can automate daily or weekly summaries using cron (Linux/Mac) or Task Scheduler (Windows).

### Cron Example (Daily at 5 PM)

```bash
0 17 * * * cd /path/to/teams-chat-summarizer && ./venv/bin/python main.py --output /path/to/summaries/daily-$(date +\%Y-\%m-\%d).txt
```

## Troubleshooting

### Authentication Errors

**Problem**: `Failed to acquire access token`

**Solutions**:
- Verify your `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, and `AZURE_CLIENT_SECRET` are correct
- Ensure admin consent was granted for API permissions
- Check that your client secret hasn't expired

### Permission Errors

**Problem**: `Access denied` or `Insufficient permissions`

**Solutions**:
- Verify you've added `Chat.Read.All` and `Chat.ReadBasic.All` permissions
- Ensure admin consent was granted (requires admin privileges)
- Check that you're using "Application permissions" not "Delegated permissions"

### Chat Not Found

**Problem**: `No chat found with name 'XXX'`

**Solutions**:
- Verify the exact chat name (case-sensitive)
- Try using the chat ID directly with `--chat-id`
- Ensure your app has permission to access the chat

### Claude API Errors

**Problem**: `Error generating summary`

**Solutions**:
- Verify your `ANTHROPIC_API_KEY` is correct
- Check you have sufficient API credits
- Ensure you're using a valid model name

## Security Best Practices

1. **Never commit `.env` file** - It contains sensitive credentials
2. **Rotate secrets regularly** - Update client secrets and API keys periodically
3. **Use least privilege** - Only grant necessary permissions
4. **Secure storage** - Store API keys securely (consider using a secrets manager)
5. **Monitor usage** - Check Azure and Anthropic consoles for unusual activity

## API Costs

- **Microsoft Graph API**: Generally free for most organizations with Microsoft 365
- **Claude API**: Pay-per-token pricing, varies by model
  - Check current pricing at https://www.anthropic.com/pricing
  - Estimate: ~$0.01-0.05 per summary depending on chat length and model

## File Structure

```
.
├── main.py              # Main entry point
├── teams_client.py      # Teams/Graph API client
├── summarizer.py        # Claude API integration
├── config.py            # Configuration management
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
├── .env                 # Your credentials (not in git)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Contributing

Feel free to submit issues or pull requests to improve this tool!

## License

MIT License - feel free to use and modify as needed.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Azure and Anthropic documentation
3. Open an issue in this repository

## Changelog

### v1.0.0 (2026-01-16)
- Initial release
- Basic Teams chat fetching
- Claude API summarization
- Command-line interface
- Custom prompts support
