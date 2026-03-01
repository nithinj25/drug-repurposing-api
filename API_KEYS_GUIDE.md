# 🔑 API Keys Setup Guide

## Quick Summary - What You NEED to Get Started

### ✅ **REQUIRED (Free)**
1. **Groq API Key** - For AI processing (FREE, Fast)
   - Sign up: https://console.groq.com/keys
   - Get 30 requests/minute free

2. **Your Email** - For NCBI courtesy
   - Just use your regular email

### 📝 **Where to Put API Keys**

1. **Copy the template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file** (created in project root folder)
   - Open: `drug-repurposing-assistant\.env`
   - Fill in your keys

3. **Minimum Configuration:**
   ```env
   USE_GROQ=true
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxx
   NCBI_EMAIL=your_email@example.com
   ```

---

## 📋 Complete API List & Status

### 🤖 AI/LLM APIs (Pick ONE)

| API | Cost | Speed | Quality | Rate Limit | Get It Here |
|-----|------|-------|---------|------------|-------------|
| **Groq** ⭐ RECOMMENDED | FREE | ⚡ Very Fast | Good | 30/min | https://console.groq.com/keys |
| **OpenAI** | Paid | Fast | Excellent | 3,500/min | https://platform.openai.com/api-keys |

**Cost Comparison:**
- Groq: $0 (free tier is generous)
- OpenAI: ~$0.50-2 per 100 queries (gpt-4o-mini)

---

### 📚 Literature & Research APIs (All FREE)

| API | Required? | Rate Limit | Key Needed? | Where to Get |
|-----|-----------|------------|-------------|--------------|
| **PubMed/NCBI** | Optional | 3/sec (10 with key) | No (but helps) | https://www.ncbi.nlm.nih.gov/account/ |
| **PubMed Central** | No | Same as above | No | Same as above |

**What it does:** Fetches scientific papers about drugs and diseases

---

### 🏥 Clinical Trials APIs (FREE)

| API | Required? | Rate Limit | Key Needed? | URL |
|-----|-----------|------------|-------------|-----|
| **ClinicalTrials.gov** | No | ~1/sec | No | Public API |

**What it does:** Gets clinical trial data (phases, outcomes, safety data)

---

### 🧬 Molecular & Chemical APIs (FREE)

| API | Required? | Rate Limit | Key Needed? |
|-----|-----------|------------|-------------|
| **PubChem** | No | 5/sec | No |
| **ChEMBL** | No | ~5/sec | No |
| **DrugBank** | Optional | Varies | Yes (academic) |

**What it does:** Chemical structures, targets, pathways, drug interactions

---

### 📋 Patent APIs (FREE with limits)

| API | Required? | Rate Limit | Key Needed? |
|-----|-----------|------------|-------------|
| **PatentsView** | No | Limited | No |
| **Google Patents** | Optional | Varies | Yes (Google Cloud) |

**What it does:** Patent search, claims analysis, IP freedom-to-operate

---

### ⚠️ Safety & Regulatory APIs (FREE)

| API | Required? | Rate Limit | Key Needed? | Where to Get |
|-----|-----------|------------|-------------|--------------|
| **FDA openFDA** | Optional | 240/min | No (but recommended) | https://open.fda.gov/apis/authentication/ |
| **DailyMed** | No | Reasonable use | No | Public |

**What it does:** Drug labels, adverse events, safety warnings, recalls

---

### 💰 Market Intelligence (PAID - Not Required)

| API | Cost | Required? |
|-----|------|-----------|
| **IQVIA** | $$$$ Enterprise | No - will use mock data |
| **Bloomberg** | $$$ | No - will use mock data |

**What it does:** Market size, competitor analysis, pricing
**Note:** These are enterprise-only. App will use synthetic data if not available.

---

## 🚀 Quick Start Instructions

### Step 1: Get Groq API Key (2 minutes)
1. Go to: https://console.groq.com
2. Sign up with Google/GitHub
3. Click "Create API Key"
4. Copy the key (starts with `gsk_`)

### Step 2: Create .env File
```bash
# In project root folder
cp .env.example .env
```

### Step 3: Edit .env File
Open `.env` and paste your keys:
```env
USE_GROQ=true
GROQ_API_KEY=gsk_your_actual_key_here
NCBI_EMAIL=yourname@email.com
```

### Step 4: Test It
```bash
python src/api.py
```

---

## 🎯 Recommended Setup (Free Tier)

For testing and development, you only need:

```env
# Minimum viable configuration
USE_GROQ=true
GROQ_API_KEY=gsk_xxxxxxxxxxxxx
NCBI_EMAIL=your@email.com
PORT=8000
LOG_LEVEL=INFO
CORS_ENABLED=true
```

This will:
- ✅ Enable AI-powered analysis (Groq)
- ✅ Access PubMed (no key needed, with courtesy email)
- ✅ Access ClinicalTrials.gov (public API)
- ✅ Access PubChem/ChEMBL (public APIs)
- ✅ Use mock data for IQVIA market intelligence

---

## 💡 Pro Tips

1. **Start with Groq** - It's free and fast enough
2. **Add NCBI key later** - Only if you need higher rate limits
3. **Skip market APIs** - Mock data works fine for testing
4. **FDA key optional** - Only for production with heavy usage

---

## 🔒 Security Notes

1. **Never commit `.env` to git** - It's in .gitignore
2. **Don't share API keys** - They're personal to you
3. **Rotate keys regularly** - Especially in production
4. **Use different keys** - Dev vs Production

---

## ❓ FAQ

**Q: Do I need ALL these APIs?**
A: No! Just Groq API is enough to start.

**Q: Why use Groq over OpenAI?**
A: Groq is free, faster, and good enough for most tasks. OpenAI is better but costs money.

**Q: Can I use mock data?**
A: Yes! If you don't set API keys, the app will use synthetic/mock data.

**Q: What's the total cost?**
A: With Groq free tier: $0/month. With OpenAI: ~$5-20/month depending on usage.

**Q: How do I know if my keys work?**
A: Run the API server and make a test query. Check logs for connection errors.

---

## 🆘 Troubleshooting

**Error: "GROQ_API_KEY not found"**
- Make sure `.env` file exists in project root
- Check that the key is set correctly
- Restart the API server

**Error: "Rate limit exceeded"**
- Wait 60 seconds and try again
- Get an NCBI API key for higher limits
- Add delays between requests

**Error: "Connection refused"**
- Check your internet connection
- Verify API endpoints are accessible
- Check if firewall is blocking requests

---

## 📞 Need Help?

1. Check the logs: `drug_repurposing_assistant.log`
2. Test individual APIs using provided test scripts
3. Verify .env file is in correct location
4. Ensure python-dotenv is installed: `pip install python-dotenv`
