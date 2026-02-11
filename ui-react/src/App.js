import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import './App.css';
import DashboardLayout from './components/DashboardLayout';
import HomePage from './pages/HomePage';
import JobDetailPage from './pages/JobDetailPage';
import LiteratureAgentPage from './pages/LiteratureAgentPage';
import ClinicalAgentPage from './pages/ClinicalAgentPage';
import SafetyAgentPage from './pages/SafetyAgentPage';
import MolecularAgentPage from './pages/MolecularAgentPage';
import PatentAgentPage from './pages/PatentAgentPage';
import MarketAgentPage from './pages/MarketAgentPage';

// Enterprise theme configuration for Ant Design
const enterpriseTheme = {
  token: {
    colorPrimary: '#003366',
    colorSuccess: '#10b981',
    colorWarning: '#f59e0b',
    colorError: '#dc3545',
    colorInfo: '#0ea5e9',
    colorBgBase: '#f5f7fa',
    colorTextBase: '#1f2937',
    borderRadius: 8,
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
  },
  algorithm: []
};

function App() {
  return (
    <ConfigProvider theme={enterpriseTheme}>
      <Routes>
        <Route path="/" element={<DashboardLayout><HomePage /></DashboardLayout>} />
        <Route path="/jobs/:jobId" element={<DashboardLayout><JobDetailPage /></DashboardLayout>} />
        <Route path="/jobs/:jobId/literature" element={<DashboardLayout><LiteratureAgentPage /></DashboardLayout>} />
        <Route path="/jobs/:jobId/clinical" element={<DashboardLayout><ClinicalAgentPage /></DashboardLayout>} />
        <Route path="/jobs/:jobId/safety" element={<DashboardLayout><SafetyAgentPage /></DashboardLayout>} />
        <Route path="/jobs/:jobId/molecular" element={<DashboardLayout><MolecularAgentPage /></DashboardLayout>} />
        <Route path="/jobs/:jobId/patent" element={<DashboardLayout><PatentAgentPage /></DashboardLayout>} />
        <Route path="/jobs/:jobId/market" element={<DashboardLayout><MarketAgentPage /></DashboardLayout>} />
      </Routes>
    </ConfigProvider>
  );
}

export default App;
