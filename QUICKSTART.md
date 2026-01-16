# Quick Start Guide

Get started with Teams Chat Summarization in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] Access to Microsoft Teams
- [ ] Azure AD admin access (or admin to grant permissions)
- [ ] Anthropic API account

## Step-by-Step Setup

### 1. Azure App Registration (5 minutes)

1. Go to https://portal.azure.com
2. Azure Active Directory → App registrations → New registration
3. Name it "Teams Chat Summarizer"
4. Register and copy:
   - Application (client) ID
   - Directory (tenant) ID
5. Go to "Certificates & secrets" → New client secret
   - Copy the secret value immediately!
6. Go to "API permissions" → Add permission → Microsoft Graph → Application permissions
   - Add: `Chat.Read.All` and `Chat.ReadBasic.All`
   - Click "Grant admin consent" ⚠️ (Requires admin)

### 2. Get Claude API Key (2 minutes)

1. Go to https://console.anthropic.com
2. Create account / Sign in
3. API Keys → Create Key
4. Copy your API key

### 3. Install Tool (2 minutes)

```bash
# Navigate to the tool directory
cd /path/to/teams-chat-summarizer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### 4. Configure Credentials (1 minute)

Edit `.env` file:

```env
AZURE_CLIENT_ID=paste-your-client-id
AZURE_TENANT_ID=paste-your-tenant-id
AZURE_CLIENT_SECRET=paste-your-client-secret
ANTHROPIC_API_KEY=paste-your-api-key
TEAMS_CHAT_NAME=Energy & Market Pulse
```

### 5. Run Your First Summary! (30 seconds)

```bash
python main.py
```

That's it! 🎉

## Common First-Run Issues

### "Failed to acquire access token"
- Double-check your Azure credentials in `.env`
- Ensure no extra spaces in the `.env` file

### "Insufficient permissions"
- Admin consent not granted → Ask your IT admin
- Wrong permission type → Use "Application permissions" not "Delegated"

### "Chat not found"
- Verify exact chat name (case-sensitive)
- Try listing chats first to see available names

### "ANTHROPIC_API_KEY is required"
- Make sure `.env` file exists in the same directory
- Check that you copied `.env.example` to `.env`

## Next Steps

- [Full Documentation](README.md) - Complete guide
- Try different options: `python main.py --help`
- Automate with cron/scheduled tasks
- Experiment with custom prompts

## Quick Examples

```bash
# Last 24 hours
python main.py --days 1

# Save to file
python main.py --output summary.txt

# Different chat
python main.py --chat-name "Project Alpha"

# Custom analysis
python main.py --custom-prompt "extract all deadlines and deliverables"
```

## Need Help?

1. Check [README.md](README.md) troubleshooting section
2. Verify all credentials are correct
3. Ensure admin consent was granted
4. Check that chat name exactly matches Teams

Happy summarizing! 🚀
