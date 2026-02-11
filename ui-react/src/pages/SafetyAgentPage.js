import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, Button, Empty, Spin, Tag, Table, Statistic, Row, Col } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, RadialBarChart, RadialBar } from 'recharts';
import BreadcrumbNav from '../components/BreadcrumbNav';

const COLORS = ['#00C9A7', '#845EC2', '#FF6F91', '#FFC75F', '#008E9B'];
const RISK_COLORS = {
  green: '#00C9A7',
  yellow: '#FFC75F',
  red: '#F2455C'
};

function SafetyAgentPage() {
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
        console.log('SafetyAgentPage - Fetching job:', jobId, 'from:', apiUrl);
        
        const response = await fetch(`${apiUrl}/jobs/${jobId}`);
        
        if (!response.ok) {
          throw new Error(`API returned ${response.status}`);
        }
        
        const jobData = await response.json();
        console.log('SafetyAgentPage - Job Data:', jobData);
        setDebugInfo(jobData);
        
        const tasks = jobData.tasks || {};
        const safetyTask = Object.values(tasks).find(task => 
          task.agent_name === 'safety_agent' || task.name === 'safety_agent'
        );
        
        console.log('SafetyAgentPage - Safety task:', safetyTask);
        
        if (!safetyTask) {
          setError('Safety agent results not found. The analysis may still be in progress.');
          setData({ job: jobData, safety: null });
        } else if (!safetyTask.result) {
          setError('Safety agent has no results yet.');
          setData({ job: jobData, safety: null });
        } else {
          setData({
            job: jobData,
            safety: safetyTask.result
          });
        }
      } catch (error) {
        console.error('SafetyAgentPage - Error:', error);
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
            <h1>🔬 Safety Analysis</h1>
            <p className="page-subtitle">Loading analysis...</p>
          </div>
        </div>
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Fetching safety data...</p>
        </div>
      </div>
    );
  }

  if (error || !data || !data.safety) {
    return (
      <div className="page-container">
        <div className="page-header">
          <button className="btn-back" onClick={() => navigate(`/jobs/${jobId}`)}>
            ← Back to Job
          </button>
          <div className="header-content">
            <h1>🔬 Safety Analysis</h1>
            {data?.job && (
              <p className="page-subtitle">
                {data.job.drug_name} for {data.job.indication}
              </p>
            )}
          </div>
        </div>
        <div className="error-state">
          <div className="error-icon">⚠️</div>
          <h2>Unable to Load Safety Data</h2>
          <p className="error-message">{error || 'Safety data not found'}</p>
          <p className="error-hint">The analysis may still be in progress. Please try refreshing in a moment.</p>
          <button className="btn-primary" onClick={() => window.location.reload()}>
            Refresh Page
          </button>
        </div>
      </div>
    );
  }

  const { safety, job } = data;

  // Prepare chart data
  const severityData = {};
  safety.adverse_events?.forEach(ae => {
    const severity = ae.severity || 'Unknown';
    severityData[severity] = (severityData[severity] || 0) + 1;
  });
  const severityChartData = Object.keys(severityData).map(severity => ({
    name: severity,
    value: severityData[severity]
  }));

  const frequencyData = safety.adverse_events?.map((ae, idx) => ({
    name: ae.event_term?.substring(0, 20) || `AE ${idx + 1}`,
    frequency: parseFloat(ae.frequency?.replace('%', '')) || 0,
    severity: ae.severity
  })) || [];

  const riskData = [
    {
      name: 'Safety Score',
      value: (safety.safety_score || 0) * 100,
      fill: RISK_COLORS[safety.risk_level] || '#a0aec0'
    }
  ];

  const contraindicationsByType = {};
  safety.contraindications?.forEach(ci => {
    const type = ci.contraindication_type || 'Other';
    contraindicationsByType[type] = (contraindicationsByType[type] || 0) + 1;
  });
  const contraindicationChartData = Object.keys(contraindicationsByType).map(type => ({
    name: type,
    count: contraindicationsByType[type]
  }));

  return (
    <div className="page-container">
      <div className="page-header">
        <button className="btn-back" onClick={() => navigate(`/jobs/${jobId}`)}>
          ← Back to Job
        </button>
        <h1>⚠️ Safety Assessment</h1>
        <p className="page-subtitle">
          {job.drug_name} for {job.indication}
        </p>
      </div>

      <div className="agent-detail-grid">
        {/* Summary Cards */}
        <div className="summary-cards">
          <div className="summary-card">
            <div className="summary-icon">📊</div>
            <div className="summary-content">
              <div className="summary-value">{(safety.safety_score * 100).toFixed(0)}%</div>
              <div className="summary-label">Safety Score</div>
            </div>
          </div>
          <div className="summary-card" style={{ backgroundColor: RISK_COLORS[safety.risk_level] + '20' }}>
            <div className="summary-icon">🚦</div>
            <div className="summary-content">
              <div className="summary-value" style={{ color: RISK_COLORS[safety.risk_level] }}>
                {safety.risk_level?.toUpperCase() || 'UNKNOWN'}
              </div>
              <div className="summary-label">Risk Level</div>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-icon">⚠️</div>
            <div className="summary-content">
              <div className="summary-value">{safety.adverse_events?.length || 0}</div>
              <div className="summary-label">Adverse Events</div>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-icon">🚫</div>
            <div className="summary-content">
              <div className="summary-value">{safety.contraindications?.length || 0}</div>
              <div className="summary-label">Contraindications</div>
            </div>
          </div>
        </div>

        {/* Critical Safety Risk Alert */}
        {safety.critical_safety_risk && (
          <div className="alert alert-danger">
            <div className="alert-icon">🚨</div>
            <div className="alert-content">
              <h3>Critical Safety Risk Detected</h3>
              <p>Grade 3+ adverse events with significant frequency have been identified. This drug may not be suitable for the proposed indication.</p>
            </div>
          </div>
        )}

        {/* Summary Text */}
        <div className="detail-card">
          <h2>Safety Summary</h2>
          <p>{safety.summary || 'No summary available'}</p>
        </div>

        {/* Safety Score Radial */}
        <div className="detail-card chart-card">
          <h2>Overall Safety Score</h2>
          <ResponsiveContainer width="100%" height={300}>
            <RadialBarChart 
              cx="50%" 
              cy="50%" 
              innerRadius="60%" 
              outerRadius="90%" 
              barSize={30} 
              data={riskData}
              startAngle={180}
              endAngle={0}
            >
              <RadialBar
                label={{ position: 'insideStart', fill: '#fff', formatter: (value) => `${value.toFixed(0)}%` }}
                background
                dataKey="value"
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1a202c', border: '1px solid #4a5568' }}
              />
            </RadialBarChart>
          </ResponsiveContainer>
          <div className="chart-caption" style={{ textAlign: 'center', marginTop: '1rem' }}>
            Risk Level: <strong style={{ color: RISK_COLORS[safety.risk_level] }}>{safety.risk_level?.toUpperCase()}</strong>
          </div>
        </div>

        {/* Severity Distribution */}
        {severityChartData.length > 0 && (
          <div className="detail-card chart-card">
            <h2>Severity Distribution</h2>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={severityChartData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {severityChartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a202c', border: '1px solid #4a5568' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Frequency Chart */}
        {frequencyData.length > 0 && (
          <div className="detail-card chart-card full-width">
            <h2>Adverse Event Frequency</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={frequencyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                <XAxis dataKey="name" stroke="#a0aec0" angle={-45} textAnchor="end" height={100} />
                <YAxis stroke="#a0aec0" label={{ value: 'Frequency (%)', angle: -90, position: 'insideLeft' }} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a202c', border: '1px solid #4a5568' }}
                  labelStyle={{ color: '#e2e8f0' }}
                />
                <Bar dataKey="frequency" fill="#FF6F91" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Contraindications by Type */}
        {contraindicationChartData.length > 0 && (
          <div className="detail-card chart-card">
            <h2>Contraindications by Type</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={contraindicationChartData}>
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

        {/* Adverse Events Detail */}
        <div className="detail-card full-width">
          <h2>Adverse Events ({safety.adverse_events?.length || 0})</h2>
          <div className="ae-list">
            {safety.adverse_events?.map((ae, idx) => (
              <div key={idx} className={`ae-item severity-${ae.severity}`}>
                <div className="ae-header">
                  <h3>{ae.event_term || 'Unknown Event'}</h3>
                  <div className="ae-badges">
                    <span className={`badge badge-${ae.severity}`}>{ae.severity || 'Unknown'}</span>
                    <span className="badge badge-frequency">{ae.frequency || 'N/A'}</span>
                  </div>
                </div>
                {ae.description && <p className="ae-description">{ae.description}</p>}
                {ae.outcome && (
                  <div className="ae-outcome">
                    <strong>Outcome:</strong> {ae.outcome}
                  </div>
                )}
                {ae.source && (
                  <div className="ae-source">
                    <small>Source: {ae.source}</small>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Contraindications Detail */}
        {safety.contraindications && safety.contraindications.length > 0 && (
          <div className="detail-card full-width">
            <h2>Contraindications ({safety.contraindications.length})</h2>
            <div className="contraindication-list">
              {safety.contraindications.map((ci, idx) => (
                <div key={idx} className="contraindication-item">
                  <div className="ci-header">
                    <h3>{ci.condition || 'Unknown Condition'}</h3>
                    <span className="badge badge-secondary">{ci.contraindication_type || 'General'}</span>
                  </div>
                  {ci.description && <p>{ci.description}</p>}
                  {ci.recommendation && (
                    <div className="ci-recommendation">
                      <strong>Recommendation:</strong> {ci.recommendation}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Red Flags */}
        {safety.red_flags && safety.red_flags.length > 0 && (
          <div className="detail-card full-width alert alert-warning">
            <h2>🚩 Red Flags</h2>
            <ul className="red-flags-list">
              {safety.red_flags.map((flag, idx) => (
                <li key={idx}>{flag}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Boxed Warnings */}
        {safety.boxed_warnings && safety.boxed_warnings.length > 0 && (
          <div className="detail-card full-width alert alert-danger">
            <h2>⚠️ Boxed Warnings (Black Box)</h2>
            <div className="boxed-warnings">
              {safety.boxed_warnings.map((warning, idx) => (
                <div key={idx} className="boxed-warning-item">
                  <p>{warning}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default SafetyAgentPage;
