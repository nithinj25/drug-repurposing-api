# 🚀 Deploy to Render - Step by Step Guide

## ✅ Why Render is Perfect for This API

- **Free Tier**: 750 hours/month free
- **Python Support**: Native FastAPI/Uvicorn support
- **Always On**: No cold starts (on paid plans)
- **Auto Deploy**: Deploys from GitHub automatically
- **Easy Setup**: 5 minutes to deploy

---

## 📋 Prerequisites

1. **GitHub Account** (to store your code)
2. **Render Account** (free at render.com)
3. **API Keys** (GROQ_API_KEY, PUBMED_API_KEY)

---

## 🎯 Step-by-Step Deployment

### Step 1: Push Code to GitHub

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit - Drug Repurposing API"

# Create a new repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/drug-repurposing-api.git
git branch -M main
git push -u origin main
```

### Step 2: Create Render Account

1. Go to https://render.com
2. Sign up with GitHub (recommended)
3. Authorize Render to access your repositories

### Step 3: Create New Web Service

1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repository: `drug-repurposing-api`
3. Configure settings:

**Basic Settings:**
```
Name: drug-repurposing-api
Region: Oregon (US West) or closest to you
Branch: main
Runtime: Python 3
```

**Build Settings:**
```
Build Command: pip install -r requirements.txt
Start Command: python src/api.py
```

**Instance Settings:**
```
Plan: Free (or Starter $7/month for better performance)
```

### Step 4: Add Environment Variables

In Render dashboard, go to **Environment** tab and add:

```
GROQ_API_KEY = your_groq_api_key_here
PUBMED_API_KEY = your_pubmed_api_key_here
PORT = 8000
```

### Step 5: Deploy!

Click **"Create Web Service"**

Render will:
- ✅ Clone your repository
- ✅ Install dependencies
- ✅ Start your API
- ✅ Provide a public URL

**Your API will be live at:**
```
https://drug-repurposing-api.onrender.com
```

---

## 🔧 Update Your Frontend

After deployment, update your frontend to use the Render URL:

```javascript
// Before (local)
const API_BASE_URL = 'http://localhost:8000';

// After (production)
const API_BASE_URL = 'https://drug-repurposing-api.onrender.com';
```

---

## 🧪 Test Your Deployed API

### Health Check
```bash
curl https://drug-repurposing-api.onrender.com/health
```

### Interactive Docs
```
https://drug-repurposing-api.onrender.com/docs
```

### Test Analysis
```bash
curl -X POST "https://drug-repurposing-api.onrender.com/analyze" \
  -H "Content-Type: application/json" \
  -d '{"drug_name": "aspirin", "indication": "cancer prevention"}'
```

---

## ⚠️ Important Notes for Free Tier

### Free Tier Limitations:
- **Spins down after 15 min of inactivity**
- **First request after sleep takes 30-60 seconds** (cold start)
- **750 hours/month** (enough for development)

### To Avoid Cold Starts:
**Option 1: Upgrade to Starter ($7/month)**
- Always on, no cold starts
- Better for production use

**Option 2: Keep-Alive Service (Free)**
Use a service like UptimeRobot to ping your API every 10 minutes:
```
URL to monitor: https://drug-repurposing-api.onrender.com/health
Interval: 10 minutes
```

---

## 🔄 Auto-Deploy on Push

Render automatically redeploys when you push to GitHub:

```bash
# Make changes to your code
git add .
git commit -m "Update API"
git push

# Render automatically deploys the new version
```

---

## 📊 Monitor Your API

### View Logs
1. Go to Render dashboard
2. Click your service
3. Click **"Logs"** tab
4. See real-time logs

### Check Metrics
- **Metrics** tab shows:
  - CPU usage
  - Memory usage
  - Request count
  - Response times

---

## 🐛 Troubleshooting

### API Not Starting?
**Check logs in Render dashboard for errors**

Common issues:
1. **Missing dependencies**: Add to `requirements.txt`
2. **Wrong start command**: Verify `python src/api.py`
3. **Environment variables**: Check GROQ_API_KEY is set
4. **Port binding**: API should use `PORT` env variable

### Fix Port Binding
Update `src/api.py` if needed:
```python
import os
port = int(os.getenv("PORT", 8000))
uvicorn.run(app, host="0.0.0.0", port=port)
```

### CORS Errors?
The API already has CORS enabled for all origins. If issues persist, check:
```python
# In src/api.py - should already exist
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # All origins allowed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Timeout Issues?
Analysis takes 10-30 seconds. Ensure:
1. Frontend request timeout ≥ 180 seconds
2. Render timeout is sufficient (default is OK)

---

## 💰 Cost Comparison

| Plan | Price | Best For |
|------|-------|----------|
| **Free** | $0/month | Development, testing |
| **Starter** | $7/month | Production, no cold starts |
| **Standard** | $25/month | High traffic, scaling |

**Recommendation**: Start with Free, upgrade to Starter ($7) for production.

---

## 🚀 Alternative Quick Deploy (One-Click)

Click this button after pushing to GitHub:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

---

## 📱 Frontend Deployment Options

Once your API is on Render, deploy your frontend to:

### Vercel (Recommended for React/Next.js)
```bash
npm install -g vercel
vercel
```
Set environment variable:
```
NEXT_PUBLIC_API_URL=https://drug-repurposing-api.onrender.com
```

### Netlify (Good for static sites)
```bash
npm install -g netlify-cli
netlify deploy
```

### Render Static Site (Keep everything on Render)
1. Create new **Static Site** on Render
2. Connect your frontend repo
3. Build command: `npm run build`
4. Publish directory: `dist` or `build`

---

## ✅ Verification Checklist

After deployment:
- [ ] API health check returns `{"status": "healthy"}`
- [ ] `/docs` page loads successfully
- [ ] Test `/analyze` endpoint with sample data
- [ ] Frontend can connect to API
- [ ] Environment variables are set correctly
- [ ] Logs show no errors

---

## 🎉 You're Done!

Your API is now live and accessible worldwide at:
```
https://drug-repurposing-api.onrender.com
```

Update your frontend, test it, and you're ready for production! 🚀

---

## 📞 Need Help?

- **Render Docs**: https://render.com/docs
- **Render Support**: support@render.com
- **Community Forum**: https://community.render.com

Happy deploying! 🎊
