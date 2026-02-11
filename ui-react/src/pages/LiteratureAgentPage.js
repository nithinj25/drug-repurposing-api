import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Button, Empty, Tag, Tabs, Row, Col } from 'antd';
import { ArrowLeftOutlined, FileTextOutlined, LinkOutlined, CalendarOutlined, UserOutlined } from '@ant-design/icons';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import BreadcrumbNav from '../components/BreadcrumbNav';

const COLORS = ['#22d3ee', '#003366', '#10b981', '#f59e0b', '#ef4444'];

function LiteratureAgentPage() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchJobData = async () => {
      try {
        const apiUrl = localStorage.getItem('apiUrl') || 'http://localhost:8000';
        console.log('LiteratureAgentPage - Fetching job:', jobId, 'from:', apiUrl);
        
        const response = await fetch(`${apiUrl}/jobs/${jobId}`);
        
        if (!response.ok) {
          throw new Error(`API returned ${response.status}`);
        }
        
        const jobData = await response.json();
        console.log('LiteratureAgentPage - Job Data:', jobData);
        
        // Find literature agent result in tasks
        const tasks = jobData.tasks || {};
        const literatureTask = Object.values(tasks).find(task => 
          task.agent_name === 'literature_agent' || task.name === 'literature_agent'
        );
        
        console.log('LiteratureAgentPage - Literature task:', literatureTask);
        
        if (!literatureTask) {
          setError('Literature agent results not found. The analysis may still be in progress.');
          setData({ job: jobData, literature: null });
        } else if (!literatureTask.result) {
          setError('Literature agent has no results yet.');
          setData({ job: jobData, literature: null });
        } else {
          setData({
            job: jobData,
            literature: literatureTask.result
          });
        }
      } catch (error) {
        console.error('LiteratureAgentPage - Error:', error);
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };

    fetchJobData();
  }, [jobId]);

  if (loading) {
    return (
      <div className="page-container loading-container">
        <div className="spinner"></div>
        <p>Analyzing scientific literature...</p>
      </div>
    );
  }

  if (error || !data || !data.literature) {
    return (
      <div className="page-container">
        <BreadcrumbNav />
        <div className="page-header">
          <Button 
            type="text" 
            icon={<ArrowLeftOutlined />} 
            onClick={() => navigate(`/jobs/${jobId}`)}
            className="btn-back"
          >
            Back to Analysis
          </Button>
          <h1><FileTextOutlined /> Literature Analysis</h1>
          {data?.job && (
            <p className="page-subtitle">
              {data.job.drug_name} for {data.job.indication}
            </p>
          )}
        </div>
        <div style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto' }}>
          <Empty
            description={
              <div className="error-state">
                <div className="error-icon">📊</div>
                <h2>Literature Data Unavailable</h2>
                <p className="error-message">{error || 'The literature analysis is still being processed.'}</p>
                <p className="error-hint">Please refresh the page in a few moments or check the analysis status.</p>
                <Button type="primary" onClick={() => window.location.reload()}>
                  Refresh Analysis
                </Button>
              </div>
            }
          />
        </div>
      </div>
    );
  }
  const { literature, job } = data;

  // Prepare chart data
  const publicationTrendData = literature.papers?.slice(0, 10).map((paper, idx) => ({
    name: paper.metadata?.publication_date?.substring(0, 7) || `Paper ${idx + 1}`,
    entities: paper.entities?.length || 0,
    relations: paper.relations?.length || 0
  })) || [];

  const entityTypeData = {};
  literature.papers?.forEach(paper => {
    paper.entities?.forEach(entity => {
      const type = entity.entity_type || 'Unknown';
      entityTypeData[type] = (entityTypeData[type] || 0) + 1;
    });
  });
  const entityChartData = Object.keys(entityTypeData).map(type => ({
    name: type,
    value: entityTypeData[type]
  }));

  const relationTypeData = {};
  literature.papers?.forEach(paper => {
    paper.relations?.forEach(relation => {
      const type = relation.relation_type || 'Unknown';
      relationTypeData[type] = (relationTypeData[type] || 0) + 1;
    });
  });
  const relationChartData = Object.keys(relationTypeData).map(type => ({
    name: type,
    count: relationTypeData[type]
  }));

  return (
    <div className="page-container">
      <BreadcrumbNav />
      
      {/* Page Header */}
      <div className="page-header">
        <Button 
          type="text" 
          icon={<ArrowLeftOutlined />} 
          onClick={() => navigate(`/jobs/${jobId}`)}
          className="btn-back"
        >
          Back to Analysis
        </Button>
        <h1><FileTextOutlined /> Literature Analysis</h1>
        <p className="page-subtitle">
          Comprehensive PubMed and scientific database search for {job.drug_name} in {job.indication}
        </p>
      </div>

      {/* Key Metrics */}
      <Row gutter={[24, 24]} style={{ marginBottom: '2rem' }}>
        <Col xs={24} sm={12} lg={6}>
          <div className="metric-pill">
            <div className="metric-value">{literature.papers_found || 0}</div>
            <div className="metric-label">Papers Found</div>
          </div>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <div className="metric-pill">
            <div className="metric-value">{literature.analysis_window_years || 5}</div>
            <div className="metric-label">Year Window</div>
          </div>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <div className="metric-pill">
            <div className="metric-value">{(literature.competition_index_score || 0).toFixed(2)}</div>
            <div className="metric-label">Competition Index</div>
          </div>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <div className="metric-pill">
            <div className="metric-value">{(literature.sentiment_score || 0).toFixed(2)}</div>
            <div className="metric-label">Sentiment Score</div>
          </div>
        </Col>
      </Row>

      {/* Summary Card */}
      <div className="data-card">
        <div className="card-header">
          <span className="card-icon">📋</span>
          <div>
            <h2 className="card-title">Executive Summary</h2>
            <p className="card-subtitle">Key findings from literature analysis</p>
          </div>
        </div>
        <div style={{ lineHeight: '1.8', color: '#334155', fontSize: '1rem' }}>
          {literature.summary || 'No summary available'}
        </div>
        {literature.filtered_out_older_papers > 0 && (
          <div style={{ marginTop: '1.5rem', padding: '1rem', background: '#f0f9ff', borderLeft: '4px solid #22d3ee', borderRadius: '4px' }}>
            <strong style={{ color: '#003366' }}>📌 Note:</strong>
            <p style={{ margin: '0.5rem 0 0 0', color: '#475569' }}>
              {literature.filtered_out_older_papers} older papers were filtered out to focus on recent literature within the selected analysis window.
            </p>
          </div>
        )}
      </div>

      {/* Charts Section */}
      <div style={{ marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#003366', marginBottom: '1.5rem' }}>
          📊 Analysis Results
        </h2>
        
        <Row gutter={[24, 24]}>
          {publicationTrendData.length > 0 && (
            <Col xs={24} lg={12}>
              <div className="chart-container">
                <h3 className="chart-title">
                  <span style={{ fontSize: '1.25rem' }}>📈</span> Publication Depth
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={publicationTrendData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: 'white', border: '1px solid #e2e8f0', borderRadius: '8px' }}
                      labelStyle={{ color: '#475569' }}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="entities" stroke="#22d3ee" strokeWidth={2} name="Entities" />
                    <Line type="monotone" dataKey="relations" stroke="#003366" strokeWidth={2} name="Relations" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Col>
          )}

          {entityChartData.length > 0 && (
            <Col xs={24} lg={12}>
              <div className="chart-container">
                <h3 className="chart-title">
                  <span style={{ fontSize: '1.25rem' }}>🧬</span> Entity Distribution
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={entityChartData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {entityChartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: 'white', border: '1px solid #e2e8f0', borderRadius: '8px' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </Col>
          )}

          {relationChartData.length > 0 && (
            <Col xs={24} lg={24}>
              <div className="chart-container">
                <h3 className="chart-title">
                  <span style={{ fontSize: '1.25rem' }}>🔗</span> Relation Types Found
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={relationChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip contentStyle={{ backgroundColor: 'white', border: '1px solid #e2e8f0', borderRadius: '8px' }} />
                    <Bar dataKey="count" fill="#22d3ee" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Col>
          )}
        </Row>
      </div>

      {/* Research Papers Section */}
      <div className="data-card">
        <div className="card-header">
          <span className="card-icon">📚</span>
          <div>
            <h2 className="card-title">Research Papers</h2>
            <p className="card-subtitle">{literature.papers?.length || 0} papers identified</p>
          </div>
        </div>

        <div className="results-container" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))' }}>
          {literature.papers?.map((paper, idx) => (
            <div key={idx} className="result-item">
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem', marginBottom: '1rem' }}>
                <FileTextOutlined style={{ fontSize: '1.5rem', color: '#22d3ee', flexShrink: 0, marginTop: '0.25rem' }} />
                <div style={{ flex: 1 }}>
                  <h3 className="result-title">{paper.metadata?.title || 'Untitled Study'}</h3>
                  <p className="result-meta">
                    {paper.metadata?.publication_date && (
                      <span className="result-meta-item">
                        <CalendarOutlined /> {paper.metadata.publication_date}
                      </span>
                    )}
                  </p>
                </div>
              </div>

              <p className="result-description">
                {paper.metadata?.abstract ? paper.metadata.abstract.substring(0, 200) + '...' : 'No abstract available'}
              </p>

              <div style={{ marginBottom: '1rem' }}>
                {paper.metadata?.journal && (
                  <Tag color="blue" style={{ marginBottom: '0.5rem' }}>{paper.metadata.journal}</Tag>
                )}
                {paper.metadata?.pmid && (
                  <Tag style={{ marginBottom: '0.5rem' }}>PMID: {paper.metadata.pmid}</Tag>
                )}
              </div>

              <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1rem', fontSize: '0.875rem' }}>
                <span className="custom-tag primary">🧬 {paper.entities?.length || 0} Entities</span>
                <span className="custom-tag success">🔗 {paper.relations?.length || 0} Relations</span>
              </div>

              {paper.metadata?.authors && paper.metadata.authors.length > 0 && (
                <p style={{ fontSize: '0.8125rem', color: '#94a3b8', marginTop: '0.75rem' }}>
                  <UserOutlined /> {paper.metadata.authors.slice(0, 2).join(', ')}{paper.metadata.authors.length > 2 ? ' et al.' : ''}
                </p>
              )}

              {paper.metadata?.doi && (
                <a href={`https://doi.org/${paper.metadata.doi}`} target="_blank" rel="noopener noreferrer" className="result-action">
                  <LinkOutlined /> View Paper
                </a>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Details Section */}
      {literature.papers?.length > 0 && (
        <div className="data-card" style={{ marginTop: '2rem' }}>
          <div className="card-header">
            <span className="card-icon">🔍</span>
            <div>
              <h2 className="card-title">Entity & Relation Details</h2>
              <p className="card-subtitle">Extracted biomedical entities and relationships</p>
            </div>
          </div>

          <Tabs
            items={literature.papers?.slice(0, 5).map((paper, pIdx) => ({
              key: `paper-${pIdx}`,
              label: `Paper ${pIdx + 1}`,
              children: (
                <div>
                  <h4 style={{ color: '#003366', marginBottom: '1rem' }}>{paper.metadata?.title || 'Paper Details'}</h4>
                  
                  {paper.entities && paper.entities.length > 0 && (
                    <div style={{ marginBottom: '2rem' }}>
                      <h5 style={{ color: '#475569', marginBottom: '1rem' }}>Entities ({paper.entities.length})</h5>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                        {paper.entities.map((entity, eIdx) => (
                          <Tag 
                            key={eIdx} 
                            color="cyan"
                            title={`Confidence: ${(entity.confidence * 100).toFixed(0)}%`}
                            style={{ cursor: 'help' }}
                          >
                            {entity.entity_type}: {entity.text.substring(0, 30)}
                          </Tag>
                        ))}
                      </div>
                    </div>
                  )}

                  {paper.relations && paper.relations.length > 0 && (
                    <div>
                      <h5 style={{ color: '#475569', marginBottom: '1rem' }}>Relations ({paper.relations.length})</h5>
                      <table className="data-grid" style={{ width: '100%' }}>
                        <thead>
                          <tr>
                            <th>Source</th>
                            <th>Relationship</th>
                            <th>Target</th>
                            <th>Confidence</th>
                          </tr>
                        </thead>
                        <tbody>
                          {paper.relations.map((relation, rIdx) => {
                            const entity1 = paper.entities?.find(e => e.entity_id === relation.entity1_id);
                            const entity2 = paper.entities?.find(e => e.entity_id === relation.entity2_id);
                            return (
                              <tr key={rIdx}>
                                <td><strong>{entity1?.text || 'Entity 1'}</strong></td>
                                <td>{relation.relation_type}</td>
                                <td><strong>{entity2?.text || 'Entity 2'}</strong></td>
                                <td><span className="status-badge pending">{(relation.confidence * 100).toFixed(0)}%</span></td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )
            })) || []}
          />
        </div>
      )}
    </div>
  );
}
}

export default LiteratureAgentPage;
