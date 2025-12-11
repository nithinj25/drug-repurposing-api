# Deployment Guide - Drug Repurposing API

## Quick Start - Local Deployment

### Option 1: Run Locally (Development)

**Start the API server:**
```bash
python src/api.py
```

The API will be available at:
- **API Endpoint**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **API Schema**: `http://localhost:8000/openapi.json`

**Frontend Configuration:**
```javascript
// In your frontend code
const API_BASE_URL = 'http://localhost:8000';
```

**CORS is already enabled** for all origins in the API, so your frontend can make requests directly.

---

## Option 2: Run on Network (Same Machine)

To allow access from other devices on your network:

**1. Find your local IP address:**
```powershell
# On Windows
ipconfig
# Look for "IPv4 Address" (e.g., 192.168.1.100)
```

**2. Modify `src/api.py` to bind to all interfaces:**
The API already binds to `0.0.0.0:8000`, which accepts connections from any interface.

**3. Start the server:**
```bash
python src/api.py
```

**4. Frontend Configuration:**
```javascript
// Replace with your actual IP address
const API_BASE_URL = 'http://192.168.1.100:8000';
```

**5. Windows Firewall (if needed):**
```powershell
# Allow port 8000 through firewall
New-NetFirewallRule -DisplayName "Drug Repurposing API" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

---

## Option 3: Production Deployment with Gunicorn

### Install Gunicorn
```bash
pip install gunicorn
```

### Create Gunicorn configuration
```python
# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 300
keepalive = 5
```

### Run with Gunicorn
```bash
gunicorn src.api:app -c gunicorn.conf.py
```

---

## Option 4: Docker Deployment

### Create Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "src/api.py"]
```

### Build and Run
```bash
# Build image
docker build -t drug-repurposing-api .

# Run container
docker run -d -p 8000:8000 --name drug-api drug-repurposing-api
```

### Frontend Configuration
```javascript
const API_BASE_URL = 'http://localhost:8000';  // Or your Docker host IP
```

---

## Option 5: Cloud Deployment

### Deploy to Azure App Service

**1. Create Azure resources:**
```bash
az login
az group create --name drug-api-rg --location eastus
az appservice plan create --name drug-api-plan --resource-group drug-api-rg --sku B1 --is-linux
az webapp create --resource-group drug-api-rg --plan drug-api-plan --name drug-repurposing-api --runtime "PYTHON:3.11"
```

**2. Configure startup:**
```bash
az webapp config set --resource-group drug-api-rg --name drug-repurposing-api --startup-file "python src/api.py"
```

**3. Deploy code:**
```bash
az webapp up --name drug-repurposing-api --resource-group drug-api-rg
```

**Frontend URL:**
```javascript
const API_BASE_URL = 'https://drug-repurposing-api.azurewebsites.net';
```

---

### Deploy to AWS Elastic Beanstalk

**1. Install EB CLI:**
```bash
pip install awsebcli
```

**2. Initialize:**
```bash
eb init -p python-3.11 drug-repurposing-api
```

**3. Create and deploy:**
```bash
eb create drug-api-env
eb deploy
```

**Frontend URL:**
```javascript
const API_BASE_URL = 'http://drug-api-env.us-east-1.elasticbeanstalk.com';
```

---

### Deploy to Google Cloud Run

**1. Create Dockerfile (see Option 4)**

**2. Build and push:**
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/drug-api
```

**3. Deploy:**
```bash
gcloud run deploy drug-api --image gcr.io/YOUR_PROJECT_ID/drug-api --platform managed --allow-unauthenticated
```

**Frontend URL:**
```javascript
const API_BASE_URL = 'https://drug-api-xxxxx.run.app';
```

---

## Frontend Integration Examples

### JavaScript/Fetch
```javascript
const API_BASE_URL = 'http://localhost:8000';

async function analyzeDrug(drugName, indication) {
  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      drug_name: drugName,
      indication: indication
    })
  });
  
  const result = await response.json();
  return result;
}

// Usage
const result = await analyzeDrug('aspirin', 'cancer prevention');
console.log('Job ID:', result.job_id);
console.log('Status:', result.data.status);
console.log('Composite Score:', result.data.reasoning_result.composite_score);
```

### React Example
```javascript
import React, { useState } from 'react';

const API_BASE_URL = 'http://localhost:8000';

function DrugAnalysis() {
  const [drugName, setDrugName] = useState('');
  const [indication, setIndication] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          drug_name: drugName,
          indication: indication
        })
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input 
        value={drugName} 
        onChange={(e) => setDrugName(e.target.value)}
        placeholder="Drug name"
      />
      <input 
        value={indication} 
        onChange={(e) => setIndication(e.target.value)}
        placeholder="Indication"
      />
      <button onClick={handleAnalyze} disabled={loading}>
        {loading ? 'Analyzing...' : 'Analyze'}
      </button>
      
      {result && (
        <div>
          <h3>Results</h3>
          <p>Job ID: {result.job_id}</p>
          <p>Status: {result.data.status}</p>
          <p>Decision: {result.data.reasoning_result.hypotheses[0].decision}</p>
          <p>Score: {result.data.reasoning_result.composite_score}</p>
        </div>
      )}
    </div>
  );
}
```

### Axios Example
```javascript
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json'
  }
});

// Analyze drug
export const analyzeDrug = async (drugName, indication) => {
  const response = await api.post('/analyze', {
    drug_name: drugName,
    indication: indication
  });
  return response.data;
};

// Get job status
export const getJobStatus = async (jobId) => {
  const response = await api.get(`/jobs/${jobId}`);
  return response.data;
};

// Get all jobs
export const getAllJobs = async () => {
  const response = await api.get('/jobs');
  return response.data;
};

// Get available agents
export const getAgents = async () => {
  const response = await api.get('/agents');
  return response.data;
};
```

---

## API Endpoints for Frontend

### POST `/analyze` - Analyze Drug Repurposing
```javascript
// Request
{
  "drug_name": "aspirin",
  "indication": "cancer prevention",
  "options": {
    "include_molecular": true,
    "include_safety": true,
    "max_papers": 20
  }
}

// Response (84KB complete data)
{
  "success": true,
  "job_id": "uuid",
  "data": {
    "status": "completed",
    "drug_name": "aspirin",
    "indication": "cancer prevention",
    "tasks": { /* All 6 agent results */ },
    "reasoning_result": {
      "composite_score": 0.52,
      "decision_level": "review_required",
      "hypotheses": [...]
    }
  }
}
```

### GET `/jobs/{job_id}` - Get Specific Job
```javascript
// Response
{
  "job_id": "uuid",
  "status": "completed",
  "drug_name": "aspirin",
  "indication": "cancer prevention",
  "tasks": { /* Complete results */ },
  "reasoning_result": { /* Full analysis */ }
}
```

### GET `/jobs` - List All Jobs
```javascript
// Response
[
  {
    "job_id": "uuid-1",
    "drug_name": "aspirin",
    "indication": "cancer prevention",
    "status": "completed",
    "created_at": "2025-12-10T18:14:23"
  }
]
```

### GET `/agents` - Get Available Agents
```javascript
// Response
{
  "total_agents": 6,
  "agents": {
    "literature": {
      "name": "Literature Agent",
      "dimension": "clinical_evidence",
      "description": "Searches scientific literature..."
    }
    // ... 5 more agents
  }
}
```

### GET `/health` - Health Check
```javascript
// Response
{
  "status": "healthy",
  "timestamp": "2025-12-10T18:14:23"
}
```

---

## Environment Variables

Create a `.env` file for configuration:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# External API Keys (required)
GROQ_API_KEY=your_groq_api_key_here
PUBMED_API_KEY=your_pubmed_api_key_here

# Optional Configuration
MAX_WORKERS=4
TIMEOUT_SECONDS=300
LOG_LEVEL=INFO
```

Load in Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
```

---

## Security Considerations

### For Production:

1. **API Key Authentication:**
```python
# Add to src/api.py
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# Add to endpoints
@app.post("/analyze", dependencies=[Depends(verify_api_key)])
```

2. **Rate Limiting:**
```bash
pip install slowapi
```

3. **HTTPS:** Use a reverse proxy (nginx) or cloud provider SSL

4. **CORS:** Restrict origins in production
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],  # Specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

## Monitoring & Logs

### View Logs
```bash
# Real-time logs
tail -f api.log

# In Docker
docker logs -f drug-api
```

### Health Monitoring
```bash
# Continuous health check
while true; do curl http://localhost:8000/health; sleep 30; done
```

---

## Troubleshooting

### Port Already in Use
```powershell
# Find process using port 8000
Get-NetTCPConnection -LocalPort 8000 | Select-Object -ExpandProperty OwningProcess

# Kill the process
Stop-Process -Id <PID> -Force
```

### CORS Errors
- Ensure API is running on accessible host
- Check browser console for specific CORS error
- Verify frontend is using correct API URL

### Timeout Issues
- Increase timeout in frontend requests (analysis takes 10-30s)
- Increase gunicorn timeout if using production server

---

## Next Steps

1. **Choose deployment method** (Local for dev, Cloud for production)
2. **Update frontend API URL** with your deployment endpoint
3. **Test API endpoints** using provided examples
4. **Monitor performance** and adjust workers/timeout as needed
5. **Add authentication** if deploying publicly

**Ready to deploy!** Start with local deployment and scale up as needed.
