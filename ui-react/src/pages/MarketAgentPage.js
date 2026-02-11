import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, Button, Empty, Spin, Statistic, Row, Col } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import BreadcrumbNav from '../components/BreadcrumbNav';

function MarketAgentPage() {
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
        console.log('MarketAgentPage - Fetching job:', jobId, 'from:', apiUrl);
        
        const response = await fetch(`${apiUrl}/jobs/${jobId}`);
        
        if (!response.ok) {
          throw new Error(`API returned ${response.status}`);
        }
        
        const jobData = await response.json();
        console.log('MarketAgentPage - Job Data:', jobData);
        setDebugInfo(jobData);
        
        const tasks = jobData.tasks || {};
        const marketTask = Object.values(tasks).find(task => 
          task.agent_name === 'market_agent' || task.name === 'market_agent'
        );
        
        console.log('MarketAgentPage - Market task:', marketTask);
        
        if (!marketTask) {
          setError('Market agent results not found. The analysis may still be in progress.');
          setData({ job: jobData, market: null });
        } else if (!marketTask.result) {
          setError('Market agent has no results yet.');
          setData({ job: jobData, market: null });
        } else {
          setData({
            job: jobData,
            market: marketTask.result
          });
        }
      } catch (error) {
        console.error('MarketAgentPage - Error:', error);
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
            <h1>💼 Market Analysis</h1>
            <p className="page-subtitle">Loading analysis...</p>
          </div>
        </div>
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Fetching market data...</p>
        </div>
      </div>
    );
  }

  if (error || !data || !data.market) {
    return (
      <div className="page-container">
        <div className="page-header">
          <button className="btn-back" onClick={() => navigate(`/jobs/${jobId}`)}>
            ← Back to Job
          </button>
          <div className="header-content">
            <h1>💼 Market Analysis</h1>
            {data?.job && (
              <p className="page-subtitle">
                {data.job.drug_name} for {data.job.indication}
              </p>
            )}
          </div>
        </div>
        <div className="error-state">
          <div className="error-icon">⚠️</div>
          <h2>Unable to Load Market Data</h2>
          <p className="error-message">{error || 'Market data not found'}</p>
          <p className="error-hint">The analysis may still be in progress. Please try refreshing in a moment.</p>
          <button className="btn-primary" onClick={() => window.location.reload()}>
            Refresh Page
          </button>
        </div>
      </div>
    );
  }

  const { market, job } = data;

  // Prepare chart data
  const competitorData = market.competitors?.map((comp, idx) => ({
    name: comp.name || `Competitor ${idx + 1}`,
    market_share: comp.market_share || 0,
    revenue: comp.revenue_usd || 0
  })) || [];

  const marketMetrics = [
    { name: 'Total Addressable Market', value: market.total_addressable_market_usd || 0, fill: '#00C9A7' },
    { name: 'Projected Market Share', value: (market.total_addressable_market_usd || 0) * 0.1, fill: '#845EC2' }
  ];

  const opportunityData = [
    { name: 'Market Opportunity', value: (market.market_opportunity_score || 0) * 100, fill: '#FFC75F' },
    { name: 'Prevalence Adjustment', value: (market.prevalence_adjustment || 1) * 100, fill: '#FF6F91' }
  ];

  return (
    <div className="page-container">
      <div className="page-header">
        <button className="btn-back" onClick={() => navigate(`/jobs/${jobId}`)}>
          ← Back to Job
        </button>
        <h1>💼 Market Intelligence</h1>
        <p className="page-subtitle">
          {job.drug_name} for {job.indication}
        </p>
      </div>

      <div className="agent-detail-grid">
        {/* Summary Cards */}
        <div className="summary-cards">
          <div className="summary-card">
            <div className="summary-icon">💰</div>
            <div className="summary-content">
              <div className="summary-value">
                ${((market.total_addressable_market_usd || 0) / 1e9).toFixed(2)}B
              </div>
              <div className="summary-label">Total Addressable Market</div>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-icon">📊</div>
            <div className="summary-content">
              <div className="summary-value">{market.market_opportunity_score?.toFixed(2) || 'N/A'}</div>
              <div className="summary-label">Opportunity Score</div>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-icon">👥</div>
            <div className="summary-content">
              <div className="summary-value">{market.prevalence?.toLocaleString() || 'N/A'}</div>
              <div className="summary-label">Disease Prevalence</div>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-icon">⚔️</div>
            <div className="summary-content">
              <div className="summary-value">{market.competitors?.length || 0}</div>
              <div className="summary-label">Competitors</div>
            </div>
          </div>
        </div>

        {/* Summary Text */}
        <div className="detail-card">
          <h2>Market Summary</h2>
          <p>{market.summary || 'No summary available'}</p>
          {market.prevalence_adjustment && market.prevalence_adjustment !== 1 && (
            <div className="info-box">
              <strong>Note:</strong> Prevalence adjustment factor of {market.prevalence_adjustment} applied due to {market.prevalence > 1000000 ? 'high' : 'low'} disease prevalence.
            </div>
          )}
        </div>

        {/* Market Metrics */}
        <div className="detail-card chart-card">
          <h2>Market Size Metrics</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={marketMetrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
              <XAxis dataKey="name" stroke="#a0aec0" angle={-15} textAnchor="end" height={80} />
              <YAxis stroke="#a0aec0" tickFormatter={(value) => `$${(value / 1e9).toFixed(1)}B`} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1a202c', border: '1px solid #4a5568' }}
                labelStyle={{ color: '#e2e8f0' }}
                formatter={(value) => `$${(value / 1e9).toFixed(2)}B`}
              />
              <Bar dataKey="value" fill="#00C9A7" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Opportunity Metrics */}
        <div className="detail-card chart-card">
          <h2>Opportunity Analysis</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={opportunityData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
              <XAxis type="number" stroke="#a0aec0" />
              <YAxis dataKey="name" type="category" stroke="#a0aec0" width={150} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1a202c', border: '1px solid #4a5568' }}
                labelStyle={{ color: '#e2e8f0' }}
                formatter={(value) => `${value.toFixed(1)}%`}
              />
              <Bar dataKey="value" fill="#845EC2" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Competitor Market Share */}
        {competitorData.length > 0 && (
          <div className="detail-card chart-card full-width">
            <h2>Competitive Landscape</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={competitorData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                <XAxis dataKey="name" stroke="#a0aec0" />
                <YAxis yAxisId="left" stroke="#a0aec0" label={{ value: 'Market Share (%)', angle: -90, position: 'insideLeft' }} />
                <YAxis yAxisId="right" orientation="right" stroke="#a0aec0" tickFormatter={(value) => `$${(value / 1e6).toFixed(0)}M`} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a202c', border: '1px solid #4a5568' }}
                  labelStyle={{ color: '#e2e8f0' }}
                />
                <Legend />
                <Bar yAxisId="left" dataKey="market_share" fill="#FF6F91" name="Market Share %" />
                <Bar yAxisId="right" dataKey="revenue" fill="#FFC75F" name="Revenue (USD)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Competitors Detail */}
        <div className="detail-card full-width">
          <h2>Competitor Analysis ({market.competitors?.length || 0})</h2>
          <div className="competitors-grid">
            {market.competitors?.map((competitor, idx) => (
              <div key={idx} className="competitor-card">
                <h3>{competitor.name || 'Unknown Competitor'}</h3>
                <div className="competitor-stats">
                  {competitor.market_share && (
                    <div className="stat-row">
                      <span className="stat-label">Market Share:</span>
                      <span className="stat-value">{competitor.market_share}%</span>
                    </div>
                  )}
                  {competitor.revenue_usd && (
                    <div className="stat-row">
                      <span className="stat-label">Revenue:</span>
                      <span className="stat-value">${(competitor.revenue_usd / 1e6).toFixed(1)}M</span>
                    </div>
                  )}
                  {competitor.approval_year && (
                    <div className="stat-row">
                      <span className="stat-label">Approval Year:</span>
                      <span className="stat-value">{competitor.approval_year}</span>
                    </div>
                  )}
                </div>
                {competitor.strengths && competitor.strengths.length > 0 && (
                  <div className="competitor-section">
                    <strong>Strengths:</strong>
                    <ul>
                      {competitor.strengths.map((strength, sIdx) => (
                        <li key={sIdx}>{strength}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {competitor.weaknesses && competitor.weaknesses.length > 0 && (
                  <div className="competitor-section">
                    <strong>Weaknesses:</strong>
                    <ul>
                      {competitor.weaknesses.map((weakness, wIdx) => (
                        <li key={wIdx}>{weakness}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Market Insights */}
        {market.insights && market.insights.length > 0 && (
          <div className="detail-card full-width">
            <h2>Market Insights</h2>
            <ul className="insights-list">
              {market.insights.map((insight, idx) => (
                <li key={idx} className="insight-item">
                  <span className="insight-icon">💡</span>
                  <span>{insight}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Pricing Strategy */}
        {market.pricing_strategy && (
          <div className="detail-card">
            <h2>Pricing Strategy</h2>
            <p>{market.pricing_strategy}</p>
          </div>
        )}

        {/* Regulatory Considerations */}
        {market.regulatory_considerations && market.regulatory_considerations.length > 0 && (
          <div className="detail-card">
            <h2>Regulatory Considerations</h2>
            <ul>
              {market.regulatory_considerations.map((consideration, idx) => (
                <li key={idx}>{consideration}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default MarketAgentPage;
