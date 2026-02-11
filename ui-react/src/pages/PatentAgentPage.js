import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, Button, Empty, Spin, Tag, Table, Statistic, Row, Col } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, RadialBarChart, RadialBar } from 'recharts';
import BreadcrumbNav from '../components/BreadcrumbNav';

const COLORS = ['#00C9A7', '#845EC2', '#FF6F91', '#FFC75F', '#008E9B'];
const RISK_COLORS = {
  low: '#00C9A7',
  medium: '#FFC75F',
  high: '#FF6F91',
  unknown: '#a0aec0'
};

function PatentAgentPage() {
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
        console.log('PatentAgentPage - Fetching job:', jobId, 'from:', apiUrl);
        
        const response = await fetch(`${apiUrl}/jobs/${jobId}`);
        
        if (!response.ok) {
          throw new Error(`API returned ${response.status}`);
        }
        
        const jobData = await response.json();
        console.log('PatentAgentPage - Job Data:', jobData);
        setDebugInfo(jobData);
        
        const tasks = jobData.tasks || {};
        const patentTask = Object.values(tasks).find(task => 
          task.agent_name === 'patent_agent' || task.name === 'patent_agent'
        );
        
        console.log('PatentAgentPage - Patent task:', patentTask);
        
        if (!patentTask) {
          setError('Patent agent results not found. The analysis may still be in progress.');
          setData({ job: jobData, patent: null });
        } else if (!patentTask.result) {
          setError('Patent agent has no results yet.');
          setData({ job: jobData, patent: null });
        } else {
          setData({
            job: jobData,
            patent: patentTask.result
          });
        }
      } catch (error) {
        console.error('PatentAgentPage - Error:', error);
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
            <h1>📜 Patent Analysis</h1>
            <p className="page-subtitle">Loading analysis...</p>
          </div>
        </div>
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Fetching patent data...</p>
        </div>
      </div>
    );
  }

  if (error || !data || !data.patent) {
    return (
      <div className="page-container">
        <div className="page-header">
          <button className="btn-back" onClick={() => navigate(`/jobs/${jobId}`)}>
            ← Back to Job
          </button>
          <div className="header-content">
            <h1>📜 Patent Analysis</h1>
            {data?.job && (
              <p className="page-subtitle">
                {data.job.drug_name} for {data.job.indication}
              </p>
            )}
          </div>
        </div>
        <div className="error-state">
          <div className="error-icon">⚠️</div>
          <h2>Unable to Load Patent Data</h2>
          <p className="error-message">{error || 'Patent data not found'}</p>
          <p className="error-hint">The analysis may still be in progress. Please try refreshing in a moment.</p>
          <button className="btn-primary" onClick={() => window.location.reload()}>
            Refresh Page
          </button>
        </div>
      </div>
    );
  }

  const { patent, job } = data;

  // Prepare chart data
  const ftoData = [
    { name: 'Freedom to Operate', value: (patent.fto_score || 0) * 100, fill: RISK_COLORS[patent.fto_risk] || '#a0aec0' }
  ];

  const patentStatusData = {};
  patent.patents?.forEach(p => {
    const status = p.status || 'Unknown';
    patentStatusData[status] = (patentStatusData[status] || 0) + 1;
  });
  const statusChartData = Object.keys(patentStatusData).map(status => ({
    name: status,
    value: patentStatusData[status]
  }));

  const jurisdictionData = {};
  patent.patents?.forEach(p => {
    const jurisdiction = p.jurisdiction || 'Unknown';
    jurisdictionData[jurisdiction] = (jurisdictionData[jurisdiction] || 0) + 1;
  });
  const jurisdictionChartData = Object.keys(jurisdictionData).map(jurisdiction => ({
    name: jurisdiction,
    count: jurisdictionData[jurisdiction]
  }));

  return (
    <div className="page-container">
      <div className="page-header">
        <button className="btn-back" onClick={() => navigate(`/jobs/${jobId}`)}>
          ← Back to Job
        </button>
        <h1>📋 Patent Analysis</h1>
        <p className="page-subtitle">
          {job.drug_name} for {job.indication}
        </p>
      </div>

      <div className="agent-detail-grid">
        {/* Summary Cards */}
        <div className="summary-cards">
          <div className="summary-card">
            <div className="summary-icon">📄</div>
            <div className="summary-content">
              <div className="summary-value">{patent.patents?.length || 0}</div>
              <div className="summary-label">Patents Found</div>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-icon">🎯</div>
            <div className="summary-content">
              <div className="summary-value">{(patent.fto_score * 100).toFixed(0)}%</div>
              <div className="summary-label">FTO Score</div>
            </div>
          </div>
          <div className="summary-card" style={{ backgroundColor: RISK_COLORS[patent.fto_risk] + '20' }}>
            <div className="summary-icon">⚠️</div>
            <div className="summary-content">
              <div className="summary-value" style={{ color: RISK_COLORS[patent.fto_risk] }}>
                {patent.fto_risk?.toUpperCase() || 'UNKNOWN'}
              </div>
              <div className="summary-label">FTO Risk</div>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-icon">✅</div>
            <div className="summary-content">
              <div className="summary-value">
                {patent.patents?.filter(p => p.status === 'Active').length || 0}
              </div>
              <div className="summary-label">Active Patents</div>
            </div>
          </div>
        </div>

        {/* Summary Text */}
        <div className="detail-card">
          <h2>Patent Summary</h2>
          <p>{patent.summary || 'No summary available'}</p>
        </div>

        {/* FTO Score Radial */}
        <div className="detail-card chart-card">
          <h2>Freedom to Operate Score</h2>
          <ResponsiveContainer width="100%" height={300}>
            <RadialBarChart 
              cx="50%" 
              cy="50%" 
              innerRadius="60%" 
              outerRadius="90%" 
              barSize={30} 
              data={ftoData}
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
            FTO Risk: <strong style={{ color: RISK_COLORS[patent.fto_risk] }}>{patent.fto_risk?.toUpperCase()}</strong>
          </div>
        </div>

        {/* Patent Status Distribution */}
        {statusChartData.length > 0 && (
          <div className="detail-card chart-card">
            <h2>Patent Status Distribution</h2>
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

        {/* Jurisdiction Distribution */}
        {jurisdictionChartData.length > 0 && (
          <div className="detail-card chart-card full-width">
            <h2>Geographic Distribution</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={jurisdictionChartData}>
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

        {/* Patents Detail */}
        <div className="detail-card full-width">
          <h2>Patent Details ({patent.patents?.length || 0})</h2>
          <div className="patents-list">
            {patent.patents?.map((p, idx) => (
              <div key={idx} className="patent-item">
                <div className="patent-header">
                  <h3>{p.title || 'Untitled Patent'}</h3>
                  <div className="patent-badges">
                    <span className={`badge badge-${p.status?.toLowerCase()}`}>
                      {p.status || 'Unknown'}
                    </span>
                    <span className="badge badge-secondary">{p.jurisdiction || 'Unknown'}</span>
                  </div>
                </div>

                <div className="patent-meta">
                  {p.patent_number && (
                    <div className="meta-item">
                      <strong>Patent Number:</strong> <code>{p.patent_number}</code>
                    </div>
                  )}
                  {p.application_number && (
                    <div className="meta-item">
                      <strong>Application Number:</strong> <code>{p.application_number}</code>
                    </div>
                  )}
                </div>

                <div className="patent-dates">
                  {p.filing_date && (
                    <div className="date-item">
                      <span className="date-label">Filed:</span>
                      <span className="date-value">{p.filing_date}</span>
                    </div>
                  )}
                  {p.grant_date && (
                    <div className="date-item">
                      <span className="date-label">Granted:</span>
                      <span className="date-value">{p.grant_date}</span>
                    </div>
                  )}
                  {p.expiry_date && (
                    <div className="date-item">
                      <span className="date-label">Expires:</span>
                      <span className="date-value">{p.expiry_date}</span>
                    </div>
                  )}
                </div>

                {p.abstract && (
                  <div className="patent-abstract">
                    <strong>Abstract:</strong>
                    <p>{p.abstract}</p>
                  </div>
                )}

                {p.assignee && (
                  <div className="patent-assignee">
                    <strong>Assignee:</strong> {p.assignee}
                  </div>
                )}

                {p.inventors && p.inventors.length > 0 && (
                  <div className="patent-inventors">
                    <strong>Inventors:</strong> {p.inventors.join(', ')}
                  </div>
                )}

                {p.claims && p.claims.length > 0 && (
                  <details className="patent-details">
                    <summary>View Claims ({p.claims.length})</summary>
                    <ol className="claims-list">
                      {p.claims.map((claim, cIdx) => (
                        <li key={cIdx}>{claim}</li>
                      ))}
                    </ol>
                  </details>
                )}

                {p.relevance_score !== undefined && (
                  <div className="patent-relevance">
                    <strong>Relevance Score:</strong>
                    <div className="relevance-bar">
                      <div 
                        className="relevance-fill" 
                        style={{ 
                          width: `${(p.relevance_score * 100).toFixed(0)}%`,
                          backgroundColor: COLORS[0]
                        }}
                      >
                        {(p.relevance_score * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>
                )}

                {p.blocking_risk && (
                  <div className="patent-risk">
                    <span className="risk-icon">⚠️</span>
                    <span><strong>Blocking Risk:</strong> {p.blocking_risk}</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Recommendations */}
        {patent.recommendations && patent.recommendations.length > 0 && (
          <div className="detail-card full-width">
            <h2>Recommendations</h2>
            <ul className="recommendations-list">
              {patent.recommendations.map((rec, idx) => (
                <li key={idx} className="recommendation-item">
                  <span className="rec-icon">💡</span>
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Risks */}
        {patent.risks && patent.risks.length > 0 && (
          <div className="detail-card full-width alert alert-warning">
            <h2>⚠️ Patent Risks</h2>
            <ul className="risks-list">
              {patent.risks.map((risk, idx) => (
                <li key={idx}>{risk}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default PatentAgentPage;
