import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import InputForm from '../components/InputForm';
import ResultsDisplay from '../components/ResultsDisplay';
import DetailsModal from '../components/DetailsModal';

function HomePage() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);
  const [apiUrl, setApiUrl] = useState(() => {
    return localStorage.getItem('apiUrl') || 'http://localhost:8000';
  });
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [pollInterval, setPollInterval] = useState(null);

  const navigate = useNavigate();

  useEffect(() => {
    localStorage.setItem('apiUrl', apiUrl);
  }, [apiUrl]);

  useEffect(() => {
    return () => {
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  }, [pollInterval]);

  const showStatus = (message, type) => {
    setStatus({ message, type });
    if (type === 'success') {
      setTimeout(() => setStatus(null), 5000);
    }
  };

  const storeRecentJob = (jobId, drug, indication) => {
    const existing = JSON.parse(localStorage.getItem('recentJobs') || '[]');
    const updated = [
      { jobId, drug, indication, timestamp: new Date().toISOString() },
      ...existing.filter((job) => job.jobId !== jobId)
    ].slice(0, 10);
    localStorage.setItem('recentJobs', JSON.stringify(updated));
  };

  const handleAnalyze = async (drugName, indication, query) => {
    setLoading(true);
    setResults(null);

    try {
      const payload = {
        drug_name: drugName,
        indication: indication,
        ...(query && { query: query })
      };

      const response = await fetch(`${apiUrl}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to submit analysis');
      }

      const data = await response.json();
      showStatus(`Analysis submitted! Job ID: ${data.job_id}`, 'info');

      pollResults(data.job_id, drugName, indication);
    } catch (error) {
      console.error('Error:', error);
      showStatus(`Error: ${error.message}`, 'error');
      setLoading(false);
    }
  };

  const pollResults = (id, drugName, indication) => {
    let pollCount = 0;
    const maxPolls = 180;

    const interval = setInterval(async () => {
      pollCount++;

      try {
        const response = await fetch(`${apiUrl}/jobs/${id}`);
        if (!response.ok) {
          throw new Error('Failed to fetch job status');
        }

        const data = await response.json();

        if (data.status === 'completed') {
          clearInterval(interval);
          setPollInterval(null);
          setResults(data);
          setLoading(false);
          storeRecentJob(id, drugName, indication);
          showStatus('Analysis complete!', 'success');
          navigate(`/jobs/${id}`, { state: { jobData: data } });
        } else if (data.status === 'failed') {
          clearInterval(interval);
          setPollInterval(null);
          setLoading(false);
          showStatus(`Analysis failed: ${data.error || 'Unknown error'}`, 'error');
        }
      } catch (error) {
        console.error('Poll error:', error);
        if (pollCount >= maxPolls) {
          clearInterval(interval);
          setPollInterval(null);
          setLoading(false);
          showStatus('Analysis timeout - please check job status later', 'error');
        }
      }
    }, 1000);

    setPollInterval(interval);
  };

  const recentJobs = JSON.parse(localStorage.getItem('recentJobs') || '[]');

  return (
    <>
      <section className="hero">
        <div className="hero-content">
          <div className="hero-text">
            <p className="eyebrow">Enterprise Drug Discovery Platform</p>
            <h2>Accelerate repurposing with multi-agent AI.</h2>
            <p className="hero-subtitle">
              Combine literature mining, clinical trial signals, safety profiling, and market intelligence
              to surface viable repurposing candidates in minutes.
            </p>
            <div className="hero-actions">
              <a className="btn btn-primary" href="#analysis">Start Analysis</a>
              <a className="btn btn-secondary" href="#features">View Capabilities</a>
            </div>
            <div className="hero-stats">
              <div>
                <h3>35M+</h3>
                <p>Research articles</p>
              </div>
              <div>
                <h3>500K+</h3>
                <p>Clinical trials indexed</p>
              </div>
              <div>
                <h3>99.9%</h3>
                <p>API uptime</p>
              </div>
            </div>
          </div>
          <div className="hero-panel">
            <div className="hero-card">
              <p className="hero-card-title">Live Signal Snapshot</p>
              <div className="hero-card-grid">
                <div>
                  <span>Literature</span>
                  <strong>High novelty</strong>
                </div>
                <div>
                  <span>Safety</span>
                  <strong>Low risk</strong>
                </div>
                <div>
                  <span>Clinical</span>
                  <strong>Recruiting trials</strong>
                </div>
                <div>
                  <span>Market</span>
                  <strong>Orphan signal</strong>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="features" className="features">
        <div className="section-header">
          <p className="eyebrow">Enterprise-Grade Capabilities</p>
          <h3>AI agents working in parallel.</h3>
          <p>Every analysis includes mechanistic reasoning, clinical evidence, and commercial viability checks.</p>
        </div>
        <div className="feature-grid">
          <div className="feature-card">
            <h4>AI-Powered Analysis</h4>
            <p>Parallel agents synthesize literature, clinical signals, and safety flags.</p>
          </div>
          <div className="feature-card">
            <h4>Molecular Intelligence</h4>
            <p>Targets, pathways, and mechanistic rationale aligned with indication biology.</p>
          </div>
          <div className="feature-card">
            <h4>Safety Profiling</h4>
            <p>CTCAE Grade 3+ vetoes, boxed warnings, and adverse-event signals.</p>
          </div>
          <div className="feature-card">
            <h4>Clinical Evidence</h4>
            <p>Recruitment-weighted trials with efficacy and safety summaries.</p>
          </div>
          <div className="feature-card">
            <h4>Literature Mining</h4>
            <p>Recency-filtered evidence with competition index and sentiment scoring.</p>
          </div>
          <div className="feature-card">
            <h4>Market Intelligence</h4>
            <p>Prevalence-adjusted TAM with opportunity scores and competitive context.</p>
          </div>
        </div>
      </section>

      <section id="workflow" className="workflow">
        <div className="section-header">
          <p className="eyebrow">How It Works</p>
          <h3>Three steps to actionable insight.</h3>
        </div>
        <div className="workflow-steps">
          <div className="step-card">
            <span className="step-number">01</span>
            <h4>Select Drug & Indication</h4>
            <p>Start with an approved drug and define the target indication.</p>
          </div>
          <div className="step-card">
            <span className="step-number">02</span>
            <h4>Run Multi-Agent Analysis</h4>
            <p>Agents gather evidence across literature, trials, safety, and market.</p>
          </div>
          <div className="step-card">
            <span className="step-number">03</span>
            <h4>Review Results</h4>
            <p>Inspect scores, constraints, and structured evidence in one dashboard.</p>
          </div>
        </div>
      </section>

      <main id="analysis" className="main-content">
        <div className="input-section">
          <InputForm
            onAnalyze={handleAnalyze}
            loading={loading}
            apiUrl={apiUrl}
            onApiUrlChange={setApiUrl}
          />
          {status && (
            <div className={`status-message ${status.type}`}>
              {status.message}
            </div>
          )}
          {recentJobs.length > 0 && (
            <div className="result-card">
              <h3>Recent Analyses</h3>
              <ul className="recent-jobs">
                {recentJobs.map((job) => (
                  <li key={job.jobId}>
                    <span>{job.drug} → {job.indication}</span>
                    <button
                      className="btn btn-secondary light"
                      onClick={() => navigate(`/jobs/${job.jobId}`)}
                    >
                      View
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="results-section">
          {loading ? (
            <LoadingSkeleton />
          ) : results ? (
            <ResultsDisplay results={results} onAgentClick={setSelectedAgent} />
          ) : (
            <EmptyState />
          )}
        </div>
      </main>

      {selectedAgent && (
        <DetailsModal
          agent={selectedAgent}
          onClose={() => setSelectedAgent(null)}
        />
      )}
    </>
  );
}

const EmptyState = () => (
  <div className="empty-state">
    <p>Enter a drug name and indication to begin analysis.</p>
  </div>
);

const LoadingSkeleton = () => (
  <div className="loading-skeleton">
    {[1, 2, 3].map((i) => (
      <div key={i} className="skeleton-card">
        <div className="skeleton-title"></div>
        <div className="skeleton-line"></div>
        <div className="skeleton-line"></div>
      </div>
    ))}
  </div>
);

export default HomePage;
