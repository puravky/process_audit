# DMart Process Audit Assistant

NLP-powered audit tool — describe what you observe, AI maps it to the exact checkpoint.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (free)

1. Push this folder to a GitHub repo
2. Go to share.streamlit.io
3. Connect your repo → select app.py → Deploy

Your dad opens the URL on his phone browser. Done.

## Get Gemini API Key (free)

1. Go to aistudio.google.com/app/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Paste it in the app on startup

## How It Works

1. Enter store name, auditor name, date
2. Describe what you observed in plain English
3. AI finds the matching checkpoint (or asks one clarifying question)
4. Confirm and log the deduction
5. End session → Download Excel report

## Files

- app.py — main Streamlit app
- checkpoints.json — all 122 audit points parsed from the official sheet
- requirements.txt — Python dependencies
