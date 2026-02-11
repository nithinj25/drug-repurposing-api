import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, Button, Empty, Spin, Tag, Table, Statistic, Row, Col } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, ScatterChart, Scatter, ZAxis } from 'recharts';
import BreadcrumbNav from '../components/BreadcrumbNav';

const COLORS = ['#00C9A7', '#845EC2', '#FF6F91', '#FFC75F', '#008E9B'];
const STATUS_COLORS = {
  'Recruiting': '#00C9A7',
  'Active, not recruiting': '#FFC75F',
  'Completed': '#845EC2',
  'Terminated': '#FF6F91',
  'Withdrawn': '#F2455C',
  'Unknown': '#a0aec0'
};

function ClinicalAgentPage() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [debugInfo, setDebugInfo] = useState(null);

  useEffect(() => {
    const fetchJobData = async () => {
      try {
        const apiUrl = localStorage.getItem('apiUrl') || 'http://localhost:8000';
        console.log('ClinicalAgentPage - Fetching job:', jobId, 'from:', apiUrl);
        
        const response = await fetch(`${apiUrl}/jobs/${jobId}`);
        
        if (!response.ok) {
          throw new Error(`API returned ${response.status}`);
        }
        
        const jobData = await response.json();
        console.log('ClinicalAgentPage - Job Data:', jobData);
        setDebugInfo(jobData);
        
        const tasks = jobData.tasks || {};
        const clinicalTask = Object.values(tasks).find(task => 
          task.agent_name === 'clinical_agent' || task.name === 'clinical_agent'
        );
        
        console.log('ClinicalAgentPage - Clinical task:', clinicalTask);
        
        if (!clinicalTask) {
          setError('Clinical agent results not found. The analysis may still be in progress.');
          setData({ job: jobData, clinical: null });
        } else if (!clinicalTask.result) {
          setError('Clinical agent has no results yet.');
          setData({ job: jobData, clinical: null });
        } else {
          setData({
            job: jobData,
            clinical: clinicalTask.result
          });
        }
      } catch (error) {
        console.error('ClinicalAgentPage - Error:', error);
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };

    fetchJobData();
  }, [jobId]);

  if (loading) {
    return (
      <div className="page-container">
        <div className="page-header">
          <button className="btn-back" onClick={() => navigate(`/jobs/${jobId}`)}>
            ← Back to Job
          </button>
          <div className="header-content">
            <h1>🏯 Clinical Trials Analysis</h1>
            <p className="page-subtitle">Loading analysis...</p>
          </div>
        </div>
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Fetching clinical trials data...</p>
        </div>
      </div>
    );
  }

  if (error || !data || !data.clinical) {
    return (
      <div className="page-container">
        <div className="page-header">
          <button className="btn-back" onClick={() => navigate(`/jobs/${jobId}`)}>
            ← Back to Job
          </button>
          <div className="header-content">
            <h1>🏯 Clinical Trials Analysis</h1>
            {data?.job && (
              <p className="page-subtitle">
                {data.job.drug_name} for {data.job.indication}
              </p>
            )}
          </div>
        </div>
        <div className="error-state">
          <div className="error-icon">⚠️</div>
          <h2>Unable to Load Clinical Data</h2>
          <p className="error-message">{error || 'Clinical data not found'}</p>
          <p className="error-hint">The analysis may still be in progress. Please try refreshing in a moment.</p>
          <button className="btn-primary" onClick={() => window.location.reload()}>
            Refresh Page
          </button>
        </div>
      </div>
    );
  }

  const { clinical, job } = data;

  // Prepare chart data
  const statusData = {};
  clinical.trials?.forEach(trial => {
    const status = trial.status || 'Unknown';
    statusData[status] = (statusData[status] || 0) + 1;
  });
  const statusChartData = Object.keys(statusData).map(status => ({
    name: status,
    value: statusData[status]
  }));

  const phaseData = {};
  clinical.trials?.forEach(trial => {
    const phase = trial.phase || 'Unknown';
    phaseData[phase] = (phaseData[phase] || 0) + 1;
  });
  const phaseChartData = Object.keys(phaseData).map(phase => ({
    name: phase,
    count: phaseData[phase]
  }));

  const enrollmentData = clinical.trials?.map((trial, idx) => ({
    name: trial.nct_id || `Trial ${idx + 1}`,
    enrollment: trial.enrollment || 0,
    confidence: (trial.confidence || 0) * 100,
    status_weight: trial.status_weight || 0
  })) || [];

  return (
    <div className="page-container">
      <div className="page-header">
        <button className="btn-back" onClick={() => navigate(`/jobs/${jobId}`)}>
          ← Back to Job
        </button>
        <h1>🏥 Clinical Evidence</h1>
        <p className="page-subtitle">
          {job.drug_name} for {job.indication}
        </p>
      </div>

      <div className="agent-detail-grid">
        {/* Summary Cards */}
        <div className="summary-cards">
          <div className="summary-card">
            <div className="summary-icon">🔬</div>
            <div className="summary-content">
              <div className="summary-value">{clinical.trials_found || 0}</div>
              <div className="summary-label">Trials Found</div>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-icon">📊</div>
            <div className="summary-content">
              <div className="summary-value">{clinical.confidence?.toFixed(2) || 'N/A'}</div>
              <div className="summary-label">Overall Confidence</div>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-icon">👥</div>
            <div className="summary-content">
              <div className="summary-value">
                {clinical.trials?.reduce((sum, t) => sum + (t.enrollment || 0), 0) || 0}
              </div>
              <div className="summary-label">Total Enrollment</div>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-icon">✅</div>
            <div className="summary-content">
              <div className="summary-value">
                {clinical.trials?.filter(t => t.status === 'Completed').length || 0}
              </div>
              <div className="summary-label">Completed Trials</div>
            </div>
          </div>
        </div>

        {/* Summary Text */}
        <div className="detail-card">
          <h2>Summary</h2>
          <p>{clinical.summary || 'No summary available'}</p>
        </div>

        {/* Trial Status Distribution */}
        {statusChartData.length > 0 && (
          <div className="detail-card chart-card">
            <h2>Trial Status Distribution</h2>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={statusChartData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {statusChartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={STATUS_COLORS[entry.name] || COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a202c', border: '1px solid #4a5568' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Phase Distribution */}
        {phaseChartData.length > 0 && (
          <div className="detail-card chart-card">
            <h2>Clinical Phase Distribution</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={phaseChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                <XAxis dataKey="name" stroke="#a0aec0" />
                <YAxis stroke="#a0aec0" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a202c', border: '1px solid #4a5568' }}
                  labelStyle={{ color: '#e2e8f0' }}
                />
                <Bar dataKey="count" fill="#845EC2" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Enrollment vs Confidence Scatter */}
        {enrollmentData.length > 0 && (
          <div className="detail-card chart-card full-width">
            <h2>Trial Enrollment vs Confidence & Status Weight</h2>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                <XAxis type="number" dataKey="enrollment" name="Enrollment" stroke="#a0aec0" />
                <YAxis type="number" dataKey="confidence" name="Confidence %" stroke="#a0aec0" />
                <ZAxis type="number" dataKey="status_weight" range={[50, 400]} name="Status Weight" />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }}
                  contentStyle={{ backgroundColor: '#1a202c', border: '1px solid #4a5568' }}
                  labelStyle={{ color: '#e2e8f0' }}
                />
                <Legend />
                <Scatter name="Trials" data={enrollmentData} fill="#00C9A7" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Trials Detail */}
        <div className="detail-card full-width">
          <h2>Clinical Trials ({clinical.trials?.length || 0})</h2>
          <div className="trials-list">
            {clinical.trials?.map((trial, idx) => (
              <div key={idx} className="trial-item">
                <div className="trial-header">
                  <h3>{trial.title || 'Untitled Trial'}</h3>
                  <div className="trial-badges">
                    <span className="badge" style={{ backgroundColor: STATUS_COLORS[trial.status] || '#a0aec0' }}>
                      {trial.status || 'Unknown'}
                    </span>
                    <span className="badge badge-secondary">{trial.phase || 'Phase Unknown'}</span>
                  </div>
                </div>

                {trial.nct_id && (
                  <div className="trial-meta">
                    <strong>NCT ID:</strong> <a href={`https://clinicaltrials.gov/study/${trial.nct_id}`} target="_blank" rel="noopener noreferrer">{trial.nct_id}</a>
                  </div>
                )}

                <div className="trial-stats-grid">
                  <div className="trial-stat">
                    <span className="stat-label">Enrollment</span>
                    <span className="stat-value">{trial.enrollment || 0}</span>
                  </div>
                  <div className="trial-stat">
                    <span className="stat-label">Confidence</span>
                    <span className="stat-value">{((trial.confidence || 0) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="trial-stat">
                    <span className="stat-label">Status Weight</span>
                    <span className="stat-value">{trial.status_weight?.toFixed(2) || 'N/A'}</span>
                  </div>
                  {trial.start_date && (
                    <div className="trial-stat">
                      <span className="stat-label">Start Date</span>
                      <span className="stat-value">{trial.start_date}</span>
                    </div>
                  )}
                  {trial.completion_date && (
                    <div className="trial-stat">
                      <span className="stat-label">Completion Date</span>
                      <span className="stat-value">{trial.completion_date}</span>
                    </div>
                  )}
                </div>

                {trial.brief_summary && (
                  <div className="trial-summary">
                    <strong>Summary:</strong>
                    <p>{trial.brief_summary}</p>
                  </div>
                )}

                {trial.conditions && trial.conditions.length > 0 && (
                  <div className="trial-section">
                    <strong>Conditions:</strong>
                    <div className="tag-list">
                      {trial.conditions.map((condition, cIdx) => (
                        <span key={cIdx} className="tag">{condition}</span>
                      ))}
                    </div>
                  </div>
                )}

                {trial.interventions && trial.interventions.length > 0 && (
                  <div className="trial-section">
                    <strong>Interventions:</strong>
                    <div className="tag-list">
                      {trial.interventions.map((intervention, iIdx) => (
                        <span key={iIdx} className="tag tag-intervention">{intervention}</span>
                      ))}
                    </div>
                  </div>
                )}

                {trial.locations && trial.locations.length > 0 && (
                  <details className="trial-details">
                    <summary>View Locations ({trial.locations.length})</summary>
                    <div className="locations-grid">
                      {trial.locations.map((location, lIdx) => (
                        <div key={lIdx} className="location-item">
                          {location.facility && <div className="location-facility">{location.facility}</div>}
                          {location.city && location.country && (
                            <div className="location-address">{location.city}, {location.country}</div>
                          )}
                        </div>
                      ))}
                    </div>
                  </details>
                )}

                {trial.safety_signals && trial.safety_signals.length > 0 && (
                  <details className="trial-details">
                    <summary>Safety Signals ({trial.safety_signals.length})</summary>
                    <div className="safety-signals">
                      {trial.safety_signals.map((signal, sIdx) => (
                        <div key={sIdx} className="safety-signal-item">
                          <div className="signal-header">
                            <span className="signal-term">{signal.ae_term}</span>
                            <span className={`signal-severity severity-${signal.severity}`}>{signal.severity}</span>
                          </div>
                          <div className="signal-meta">
                            <span>Frequency: {signal.frequency}</span>
                            {signal.outcome && <span>Outcome: {signal.outcome}</span>}
                          </div>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default ClinicalAgentPage;
