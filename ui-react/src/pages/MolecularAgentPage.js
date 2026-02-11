import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, Button, Empty, Spin, Table, Statistic, Row, Col } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, ZAxis } from 'recharts';
import BreadcrumbNav from '../components/BreadcrumbNav';

function MolecularAgentPage() {
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
        console.log('MolecularAgentPage - Fetching job:', jobId, 'from:', apiUrl);
        
        const response = await fetch(`${apiUrl}/jobs/${jobId}`);
        
        if (!response.ok) {
          throw new Error(`API returned ${response.status}`);
        }
        
        const jobData = await response.json();
        console.log('MolecularAgentPage - Job Data:', jobData);
        setDebugInfo(jobData);
        
        const tasks = jobData.tasks || {};
        const molecularTask = Object.values(tasks).find(task => 
          task.agent_name === 'molecular_agent' || task.name === 'molecular_agent'
        );
        
        console.log('MolecularAgentPage - Molecular task:', molecularTask);
        
        if (!molecularTask) {
          setError('Molecular agent results not found. The analysis may still be in progress.');
          setData({ job: jobData, molecular: null });
        } else if (!molecularTask.result) {
          setError('Molecular agent has no results yet.');
          setData({ job: jobData, molecular: null });
        } else {
          setData({
            job: jobData,
            molecular: molecularTask.result
          });
        }
      } catch (error) {
        console.error('MolecularAgentPage - Error:', error);
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
            <h1>🧬 Molecular Analysis</h1>
            <p className="page-subtitle">Loading analysis...</p>
          </div>
        </div>
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Fetching molecular data...</p>
        </div>
      </div>
    );
  }

  if (error || !data || !data.molecular) {
    return (
      <div className="page-container">
        <div className="page-header">
          <button className="btn-back" onClick={() => navigate(`/jobs/${jobId}`)}>
            ← Back to Job
          </button>
          <div className="header-content">
            <h1>🧬 Molecular Analysis</h1>
            {data?.job && (
              <p className="page-subtitle">
                {data.job.drug_name} for {data.job.indication}
              </p>
            )}
          </div>
        </div>
        <div className="error-state">
          <div className="error-icon">⚠️</div>
          <h2>Unable to Load Molecular Data</h2>
          <p className="error-message">{error || 'Molecular data not found'}</p>
          <p className="error-hint">The analysis may still be in progress. Please try refreshing in a moment.</p>
          <button className="btn-primary" onClick={() => window.location.reload()}>
            Refresh Page
          </button>
        </div>
      </div>
    );
  }

  const { molecular, job } = data;

  // Prepare chart data
  const targetData = molecular.targets?.map((target, idx) => ({
    name: target.target_name || `Target ${idx + 1}`,
    confidence: (target.confidence || 0) * 100,
    interaction_type: target.interaction_type || 'Unknown'
  })) || [];

  const pathwayData = molecular.pathways?.map((pathway, idx) => ({
    name: pathway.pathway_name?.substring(0, 30) || `Pathway ${idx + 1}`,
    genes: pathway.genes_involved?.length || 0,
    relevance: (pathway.relevance_score || 0) * 100
  })) || [];

  return (
    <div className="page-container">
      <div className="page-header">
        <button className="btn-back" onClick={() => navigate(`/jobs/${jobId}`)}>
          ← Back to Job
        </button>
        <h1>🧬 Molecular Evidence</h1>
        <p className="page-subtitle">
          {job.drug_name} for {job.indication}
        </p>
      </div>

      <div className="agent-detail-grid">
        {/* Summary Cards */}
        <div className="summary-cards">
          <div className="summary-card">
            <div className="summary-icon">🎯</div>
            <div className="summary-content">
              <div className="summary-value">{molecular.targets?.length || 0}</div>
              <div className="summary-label">Molecular Targets</div>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-icon">🧪</div>
            <div className="summary-content">
              <div className="summary-value">{molecular.pathways?.length || 0}</div>
              <div className="summary-label">Pathways Involved</div>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-icon">🔬</div>
            <div className="summary-content">
              <div className="summary-value">{molecular.moa_score?.toFixed(2) || 'N/A'}</div>
              <div className="summary-label">MoA Score</div>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-icon">📊</div>
            <div className="summary-content">
              <div className="summary-value">{molecular.confidence?.toFixed(2) || 'N/A'}</div>
              <div className="summary-label">Overall Confidence</div>
            </div>
          </div>
        </div>

        {/* Summary Text */}
        <div className="detail-card">
          <h2>Molecular Summary</h2>
          <p>{molecular.summary || 'No summary available'}</p>
        </div>

        {/* Mechanism of Action */}
        {molecular.mechanism_of_action && (
          <div className="detail-card full-width">
            <h2>Mechanism of Action</h2>
            <p className="moa-text">{molecular.mechanism_of_action}</p>
          </div>
        )}

        {/* Target Confidence Chart */}
        {targetData.length > 0 && (
          <div className="detail-card chart-card full-width">
            <h2>Target Confidence Analysis</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={targetData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                <XAxis dataKey="name" stroke="#a0aec0" angle={-45} textAnchor="end" height={100} />
                <YAxis stroke="#a0aec0" label={{ value: 'Confidence (%)', angle: -90, position: 'insideLeft' }} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a202c', border: '1px solid #4a5568' }}
                  labelStyle={{ color: '#e2e8f0' }}
                />
                <Bar dataKey="confidence" fill="#00C9A7" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Pathway Relevance Chart */}
        {pathwayData.length > 0 && (
          <div className="detail-card chart-card full-width">
            <h2>Pathway Relevance & Gene Involvement</h2>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                <XAxis type="number" dataKey="genes" name="Genes" stroke="#a0aec0" />
                <YAxis type="number" dataKey="relevance" name="Relevance %" stroke="#a0aec0" />
                <ZAxis range={[50, 400]} />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }}
                  contentStyle={{ backgroundColor: '#1a202c', border: '1px solid #4a5568' }}
                  labelStyle={{ color: '#e2e8f0' }}
                  formatter={(value, name) => {
                    if (name === 'Relevance %') return `${value.toFixed(1)}%`;
                    return value;
                  }}
                />
                <Scatter name="Pathways" data={pathwayData} fill="#845EC2" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Molecular Targets Detail */}
        <div className="detail-card full-width">
          <h2>Molecular Targets ({molecular.targets?.length || 0})</h2>
          <div className="targets-grid">
            {molecular.targets?.map((target, idx) => (
              <div key={idx} className="target-card">
                <div className="target-header">
                  <h3>{target.target_name || 'Unknown Target'}</h3>
                  <span className="confidence-badge">{((target.confidence || 0) * 100).toFixed(0)}%</span>
                </div>
                <div className="target-meta">
                  <span className="target-type">{target.target_type || 'Unknown Type'}</span>
                  {target.interaction_type && (
                    <span className="interaction-type">{target.interaction_type}</span>
                  )}
                </div>
                {target.description && (
                  <p className="target-description">{target.description}</p>
                )}
                {target.evidence && target.evidence.length > 0 && (
                  <div className="target-evidence">
                    <strong>Evidence:</strong>
                    <ul>
                      {target.evidence.map((ev, evIdx) => (
                        <li key={evIdx}>{ev}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {target.gene_id && (
                  <div className="target-gene">
                    <strong>Gene ID:</strong> <code>{target.gene_id}</code>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Pathways Detail */}
        <div className="detail-card full-width">
          <h2>Biological Pathways ({molecular.pathways?.length || 0})</h2>
          <div className="pathways-list">
            {molecular.pathways?.map((pathway, idx) => (
              <div key={idx} className="pathway-item">
                <div className="pathway-header">
                  <h3>{pathway.pathway_name || 'Unknown Pathway'}</h3>
                  <div className="pathway-badges">
                    <span className="badge badge-primary">
                      {pathway.genes_involved?.length || 0} genes
                    </span>
                    <span className="badge badge-secondary">
                      {((pathway.relevance_score || 0) * 100).toFixed(0)}% relevant
                    </span>
                  </div>
                </div>
                {pathway.description && (
                  <p className="pathway-description">{pathway.description}</p>
                )}
                {pathway.genes_involved && pathway.genes_involved.length > 0 && (
                  <details className="pathway-details">
                    <summary>View Genes ({pathway.genes_involved.length})</summary>
                    <div className="genes-list">
                      {pathway.genes_involved.map((gene, gIdx) => (
                        <span key={gIdx} className="gene-tag">{gene}</span>
                      ))}
                    </div>
                  </details>
                )}
                {pathway.biological_process && (
                  <div className="pathway-process">
                    <strong>Biological Process:</strong> {pathway.biological_process}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Pharmacokinetics */}
        {molecular.pharmacokinetics && (
          <div className="detail-card">
            <h2>Pharmacokinetics</h2>
            <div className="pk-grid">
              {molecular.pharmacokinetics.absorption && (
                <div className="pk-item">
                  <strong>Absorption:</strong>
                  <p>{molecular.pharmacokinetics.absorption}</p>
                </div>
              )}
              {molecular.pharmacokinetics.distribution && (
                <div className="pk-item">
                  <strong>Distribution:</strong>
                  <p>{molecular.pharmacokinetics.distribution}</p>
                </div>
              )}
              {molecular.pharmacokinetics.metabolism && (
                <div className="pk-item">
                  <strong>Metabolism:</strong>
                  <p>{molecular.pharmacokinetics.metabolism}</p>
                </div>
              )}
              {molecular.pharmacokinetics.elimination && (
                <div className="pk-item">
                  <strong>Elimination:</strong>
                  <p>{molecular.pharmacokinetics.elimination}</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Drug Properties */}
        {molecular.drug_properties && (
          <div className="detail-card">
            <h2>Drug Properties</h2>
            <div className="properties-grid">
              {molecular.drug_properties.molecular_weight && (
                <div className="property-item">
                  <span className="property-label">Molecular Weight:</span>
                  <span className="property-value">{molecular.drug_properties.molecular_weight} g/mol</span>
                </div>
              )}
              {molecular.drug_properties.logp && (
                <div className="property-item">
                  <span className="property-label">LogP:</span>
                  <span className="property-value">{molecular.drug_properties.logp}</span>
                </div>
              )}
              {molecular.drug_properties.solubility && (
                <div className="property-item">
                  <span className="property-label">Solubility:</span>
                  <span className="property-value">{molecular.drug_properties.solubility}</span>
                </div>
              )}
              {molecular.drug_properties.bioavailability && (
                <div className="property-item">
                  <span className="property-label">Bioavailability:</span>
                  <span className="property-value">{molecular.drug_properties.bioavailability}</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default MolecularAgentPage;
