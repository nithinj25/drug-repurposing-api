import React, { useEffect, useState } from 'react';
import { useLocation, useParams, Link, useNavigate } from 'react-router-dom';
import ResultsDisplay from '../components/ResultsDisplay';

function JobDetailPage() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const [jobData, setJobData] = useState(location.state?.jobData || null);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(!location.state?.jobData);
  const [error, setError] = useState(null);

  const apiUrl = localStorage.getItem('apiUrl') || 'http://localhost:8000';

  useEffect(() => {
    if (jobData) {
      console.log('JobDetailPage - Current jobData:', jobData);
      console.log('JobDetailPage - Data structure:', {
        hasData: !!jobData.data,
        hasTasks: !!jobData.data?.tasks,
        taskKeys: jobData.data?.tasks ? Object.keys(jobData.data.tasks) : [],
        tasks: jobData.data?.tasks
      });
      return;
    }

    const fetchJob = async () => {
      try {
        setLoading(true);
        const response = await fetch(`${apiUrl}/jobs/${jobId}`);
        if (!response.ok) {
          throw new Error('Failed to fetch job data');
        }
        const data = await response.json();
        console.log('JobDetailPage - Fetched data:', data);
        setJobData(data);
        setStatus({ message: 'Loaded job response', type: 'success' });
      } catch (err) {
        console.error('JobDetailPage - Fetch error:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchJob();
  }, [apiUrl, jobData, jobId]);

  const agentButtons = [
    { name: 'Literature Evidence', path: 'literature', icon: '📚', agent: 'literature_agent' },
    { name: 'Clinical Trials', path: 'clinical', icon: '🏥', agent: 'clinical_agent' },
    { name: 'Safety Assessment', path: 'safety', icon: '⚠️', agent: 'safety_agent' },
    { name: 'Molecular Data', path: 'molecular', icon: '🧬', agent: 'molecular_agent' },
    { name: 'Patent Analysis', path: 'patent', icon: '📋', agent: 'patent_agent' },
    { name: 'Market Intelligence', path: 'market', icon: '💼', agent: 'market_agent' }
  ];

  const hasAgentData = (agentName) => {
    if (!jobData) {
      console.log(`hasAgentData(${agentName}): No jobData`);
      return false;
    }
    
    // Handle different API response structures
    const data = jobData.data || jobData;
    const tasks = data.tasks || {};
    
    console.log(`hasAgentData(${agentName}):`, {
      hasJobData: !!jobData,
      hasData: !!data,
      hasTasks: !!tasks,
      taskCount: Object.keys(tasks).length,
      taskKeys: Object.keys(tasks),
      tasks: tasks
    });
    
    const hasData = Object.values(tasks).some(task => {
      const matches = task.agent_name === agentName && task.status === 'completed';
      if (matches) {
        console.log(`Found matching task for ${agentName}:`, task);
      }
      return matches;
    });
    
    console.log(`hasAgentData(${agentName}): Result = ${hasData}`);
    return hasData;
  };

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <p className="eyebrow">Job Response</p>
          <h2>Analysis Report</h2>
          <p className="muted">Job ID: {jobId}</p>
        </div>
        <div className="page-actions">
          <Link className="btn btn-secondary light" to="/">Back to Dashboard</Link>
        </div>
      </div>

      {status && <div className={`status-message ${status.type}`}>{status.message}</div>}
      {loading && <p className="muted">Loading job response...</p>}
      {error && <p className="status-message error">{error}</p>}

      {jobData && (
        <>
          {/* Debug Info - Remove after testing */}
          <div style={{ 
            background: '#f0f0f0', 
            padding: '1rem', 
            marginBottom: '1rem', 
            borderRadius: '8px',
            fontSize: '0.85rem',
            fontFamily: 'monospace'
          }}>
            <strong>Debug Info:</strong>
            <div>Job ID: {jobId}</div>
            <div>Has jobData: {jobData ? 'Yes' : 'No'}</div>
            <div>Has jobData.data: {jobData?.data ? 'Yes' : 'No'}</div>
            <div>Has tasks: {jobData?.data?.tasks ? 'Yes' : 'No'}</div>
            <div>Task count: {jobData?.data?.tasks ? Object.keys(jobData.data.tasks).length : 0}</div>
            <div>Task keys: {jobData?.data?.tasks ? Object.keys(jobData.data.tasks).join(', ') : 'none'}</div>
            <details>
              <summary>View full data structure</summary>
              <pre style={{ maxHeight: '200px', overflow: 'auto', fontSize: '0.75rem' }}>
                {JSON.stringify(jobData, null, 2)}
              </pre>
            </details>
          </div>

          {/* Agent Navigation Section */}
          <div className="agent-nav-section">
            <div className="section-header-row">
              <div>
                <h3 className="section-title">📊 View Detailed Agent Results</h3>
                <p className="section-description">
                  Click on any agent card below to explore comprehensive analysis with charts, detailed data, and insights
                </p>
              </div>
              <div className="agent-count-badge">
                {agentButtons.filter(a => hasAgentData(a.agent)).length} / {agentButtons.length} Available
              </div>
            </div>
            <div className="agent-nav-grid">
              {agentButtons.map((agent) => {
                const isEnabled = hasAgentData(agent.agent);
                return (
                  <button
                    key={agent.path}
                    className={`agent-nav-card ${!isEnabled ? 'disabled' : ''}`}
                    onClick={() => isEnabled && navigate(`/jobs/${jobId}/${agent.path}`)}
                    disabled={!isEnabled}
                    title={isEnabled ? `View ${agent.name} details` : `${agent.name} data not available`}
                  >
                    <div className="agent-nav-icon">{agent.icon}</div>
                    <div className="agent-nav-name">{agent.name}</div>
                    {isEnabled && <div className="agent-nav-arrow">→</div>}
                    {!isEnabled && <div className="agent-nav-status">Not Available</div>}
                  </button>
                );
              })}
            </div>
          </div>

          <ResultsDisplay results={jobData} onAgentClick={() => {}} />
        </>
      )}
    </div>
  );
}

export default JobDetailPage;
