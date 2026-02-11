import React from 'react';
import ResultCard from './ResultCard';
import ReasoningBox from './ReasoningBox';
import AgentGrid from './AgentGrid';
import InsightCharts from './InsightCharts';

function ResultsDisplay({ results, onAgentClick }) {
  const agentIcons = {
    'literature_agent': '📚',
    'clinical_agent': '🏥',
    'safety_agent': '⚠️',
    'molecular_agent': '🧬',
    'patent_agent': '📋',
    'market_agent': '💼'
  };

  // Extract the actual data from the nested response structure
  const jobData = results.data || results;

  const tasks = jobData.tasks || {};
  const literature = Object.values(tasks).find(task => task.agent_name === 'literature_agent')?.result;
  const clinical = Object.values(tasks).find(task => task.agent_name === 'clinical_agent')?.result;
  const safety = Object.values(tasks).find(task => task.agent_name === 'safety_agent')?.result;
  const molecular = Object.values(tasks).find(task => task.agent_name === 'molecular_agent')?.result;
  const patent = Object.values(tasks).find(task => task.agent_name === 'patent_agent')?.result;
  const market = Object.values(tasks).find(task => task.agent_name === 'market_agent')?.result;

  const renderList = (items) => {
    if (!items || items.length === 0) return <p className="muted">None reported</p>;
    return (
      <ul className="inline-list">
        {items.map((item, idx) => (
          <li key={idx}>{item}</li>
        ))}
      </ul>
    );
  };

  return (
    <div className="results-display">
      {/* Job Summary */}
      <ResultCard
        title={`${jobData.drug_name} - ${jobData.indication}`}
        metaItems={[
          { label: 'Job ID', value: jobData.job_id },
          { label: 'Status', value: <span className="badge badge-success">{jobData.status}</span> },
          { label: 'Query', value: jobData.query || 'N/A' }
        ]}
      />

      <div className="result-card highlight">
        <h3>Analysis Overview</h3>
        <div className="summary-grid">
          <div>
            <span>Job Status</span>
            <strong>{jobData.status}</strong>
          </div>
          <div>
            <span>Agents Completed</span>
            <strong>{jobData.task_summary?.completed || 0}/{jobData.task_summary?.total || 0}</strong>
          </div>
          <div>
            <span>Evidence Items</span>
            <strong>{jobData.reasoning_result?.total_evidence_count || 'N/A'}</strong>
          </div>
          <div>
            <span>Decision</span>
            <strong>{jobData.reasoning_result?.hypotheses?.[0]?.decision || 'N/A'}</strong>
          </div>
        </div>
      </div>

      {literature && (
        <div className="result-card">
          <h3>Literature Signals</h3>
          <div className="summary-grid">
            <div>
              <span>Papers (5 yrs)</span>
              <strong>{literature.papers_found}</strong>
            </div>
            <div>
              <span>Competition Index</span>
              <strong>{literature.competition_index_score?.toFixed(2)}</strong>
            </div>
            <div>
              <span>Sentiment Score</span>
              <strong>{literature.sentiment_score?.toFixed(2)}</strong>
            </div>
            <div>
              <span>Sentiment Mix</span>
              <strong>
                {literature.sentiment_breakdown?.positive || 0}/
                {literature.sentiment_breakdown?.negative || 0}/
                {literature.sentiment_breakdown?.inconclusive || 0}
              </strong>
            </div>
          </div>
          <p className="muted">{literature.summary}</p>
        </div>
      )}

      {clinical && (
        <div className="result-card">
          <h3>Clinical Trial Signals</h3>
          <div className="summary-grid">
            <div>
              <span>Trials Found</span>
              <strong>{clinical.trials_found}</strong>
            </div>
            <div>
              <span>Latest Status</span>
              <strong>{clinical.trials?.[0]?.status || 'N/A'}</strong>
            </div>
            <div>
              <span>Status Weight</span>
              <strong>{clinical.trials?.[0]?.status_weight ?? 'N/A'}</strong>
            </div>
          </div>
          {clinical.trials?.length > 0 && (
            <div className="detail-block">
              <p><strong>Trial ID:</strong> {clinical.trials[0].trial_id}</p>
              <p><strong>Primary Outcome:</strong> {clinical.trials[0].primary_outcomes?.[0]?.measure || 'N/A'}</p>
              <p><strong>Safety Signals:</strong></p>
              {renderList(clinical.trials[0].safety_signals?.map(signal => `${signal.ae_term} (${signal.frequency || 'N/A'})`))}
            </div>
          )}
        </div>
      )}

      {safety && (
        <div className="result-card">
          <h3>Safety Profile</h3>
          <div className="summary-grid">
            <div>
              <span>Safety Score</span>
              <strong>{safety.safety_score?.toFixed(2)}</strong>
            </div>
            <div>
              <span>Risk Level</span>
              <strong>{safety.risk_level}</strong>
            </div>
            <div>
              <span>Critical Veto</span>
              <strong>{safety.critical_safety_risk ? 'Yes' : 'No'}</strong>
            </div>
            <div>
              <span>Adverse Events</span>
              <strong>{safety.adverse_events?.length || 0}</strong>
            </div>
          </div>
          <div className="detail-block">
            <p><strong>Red Flags:</strong></p>
            {renderList(safety.red_flags)}
            <p><strong>Amber Flags:</strong></p>
            {renderList(safety.amber_flags)}
            <p><strong>Green Flags:</strong></p>
            {renderList(safety.green_flags)}
          </div>
        </div>
      )}

      {market && (
        <div className="result-card">
          <h3>Market Intelligence</h3>
          <div className="summary-grid">
            <div>
              <span>TAM (USD)</span>
              <strong>{market.tam_estimate?.tam_usd?.toLocaleString() || 'N/A'}</strong>
            </div>
            <div>
              <span>Population</span>
              <strong>{market.tam_estimate?.patient_population?.toLocaleString() || 'N/A'}</strong>
            </div>
            <div>
              <span>Opportunity Score</span>
              <strong>{market.market_opportunity_score?.toFixed(2)}</strong>
            </div>
            <div>
              <span>Prevalence Adj.</span>
              <strong>{market.prevalence_adjustment?.toFixed(2)}</strong>
            </div>
          </div>
          <p className="muted">{market.market_snapshot?.market_summary || market.market_summary}</p>
        </div>
      )}

      {molecular && (
        <div className="result-card">
          <h3>Molecular Rationale</h3>
          <div className="summary-grid">
            <div>
              <span>Targets</span>
              <strong>{molecular.predicted_targets?.join(', ') || 'N/A'}</strong>
            </div>
            <div>
              <span>Pathways</span>
              <strong>{molecular.pathways?.join(', ') || 'N/A'}</strong>
            </div>
            <div>
              <span>Plausibility</span>
              <strong>{molecular.mechanistic_plausibility || 'N/A'}</strong>
            </div>
          </div>
        </div>
      )}

      {patent && (
        <div className="result-card">
          <h3>Patent & FTO</h3>
          <div className="summary-grid">
            <div>
              <span>Patents Found</span>
              <strong>{patent.patents_found}</strong>
            </div>
            <div>
              <span>FTO Status</span>
              <strong>{patent.fto_report?.overall_fto_status || 'N/A'}</strong>
            </div>
            <div>
              <span>Blocking Patents</span>
              <strong>{patent.fto_report?.blocking_patents?.length || 0}</strong>
            </div>
          </div>
          <p className="muted">{patent.fto_report?.risk_summary}</p>
        </div>
      )}

      {/* Agent Results */}
      {jobData.tasks && (
        <div className="result-card">
          <h3 style={{ marginBottom: '1.5rem' }}>Agent Analysis</h3>
          <AgentGrid 
            tasks={jobData.tasks}
            agentIcons={agentIcons}
            onAgentClick={onAgentClick}
          />
        </div>
      )}

      {/* Reasoning Section */}
      {jobData.reasoning_result && (
        <>
          <InsightCharts reasoning={jobData.reasoning_result} />
          <ReasoningBox reasoning={jobData.reasoning_result} />
        </>
      )}

      {/* Task Summary */}
      {jobData.task_summary && (
        <div className="result-card" style={{ backgroundColor: '#f0fdf4', borderLeft: '4px solid #10b981' }}>
          <h3 style={{ marginBottom: '1rem', color: '#065f46' }}>Task Summary</h3>
          <div style={{ color: '#1e293b', lineHeight: '1.8' }}>
            {typeof jobData.task_summary === 'string' ? (
              <p>{jobData.task_summary}</p>
            ) : (
              <pre style={{ backgroundColor: 'white', padding: '1rem', borderRadius: '6px', overflow: 'auto', fontSize: '0.9rem' }}>
                {JSON.stringify(jobData.task_summary, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}

      {/* Raw JSON Button */}
      <button 
        className="btn btn-secondary light"
        onClick={() => onAgentClick({ type: 'raw', data: jobData })}
        style={{ marginTop: '1.5rem', width: '100%' }}
      >
        📊 View Raw JSON (Advanced)
      </button>
    </div>
  );
}

export default ResultsDisplay;
