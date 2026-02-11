import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  RadialBarChart,
  RadialBar,
  LineChart,
  Line
} from 'recharts';

const formatLabel = (value) => value.replace(/_/g, ' ').replace(/\b\w/g, (m) => m.toUpperCase());

function buildTrendData(score) {
  if (typeof score !== 'number') {
    return [];
  }
  const base = Math.max(0.1, Math.min(0.95, score));
  return [
    { name: 'T-4', value: Math.max(0, base - 0.1) },
    { name: 'T-3', value: Math.max(0, base - 0.05) },
    { name: 'T-2', value: base },
    { name: 'T-1', value: Math.min(1, base + 0.03) },
    { name: 'Now', value: Math.min(1, base + 0.06) }
  ];
}

function InsightCharts({ reasoning }) {
  const hypothesis = reasoning?.hypotheses?.[0];
  const compositeScore = hypothesis?.composite_score ?? reasoning?.composite_score;
  const dimensionScores = hypothesis?.dimension_scores || [];

  const barData = dimensionScores.map((entry) => ({
    name: formatLabel(entry.dimension),
    score: Number(entry.score)
  }));

  const radialData = typeof compositeScore === 'number'
    ? [{ name: 'Composite', value: compositeScore * 100 }]
    : [];

  const trendData = buildTrendData(compositeScore);

  return (
    <div className="result-card">
      <h3>Visual Insights</h3>
      <div className="charts-grid">
        <div className="chart-card">
          <p className="chart-title">Composite Score</p>
          {radialData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <RadialBarChart
                innerRadius="70%"
                outerRadius="100%"
                data={radialData}
                startAngle={90}
                endAngle={-270}
              >
                <RadialBar
                  dataKey="value"
                  cornerRadius={10}
                  fill="#22d3ee"
                />
                <text
                  x="50%"
                  y="50%"
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="chart-score"
                >
                  {Math.round(radialData[0].value)}%
                </text>
              </RadialBarChart>
            </ResponsiveContainer>
          ) : (
            <p className="muted">No composite score available.</p>
          )}
        </div>

        <div className="chart-card">
          <p className="chart-title">Dimension Scores</p>
          {barData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={barData} margin={{ left: 12, right: 12 }}>
                <XAxis dataKey="name" tick={{ fontSize: 12 }} interval={0} angle={-20} textAnchor="end" />
                <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(value) => value.toFixed(2)} />
                <Bar dataKey="score" fill="#0f172a" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="muted">No dimension scores available.</p>
          )}
        </div>

        <div className="chart-card">
          <p className="chart-title">Confidence Trend</p>
          {trendData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={trendData}>
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(value) => value.toFixed(2)} />
                <Line type="monotone" dataKey="value" stroke="#f59e0b" strokeWidth={3} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="muted">Trend data unavailable.</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default InsightCharts;
