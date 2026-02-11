import React, { useState } from 'react';
import { Button, Input, Tooltip, Drawer } from 'antd';
import {
  HomeOutlined,
  FileTextOutlined,
  ThunderboltOutlined,
  ExperimentOutlined,
  SafetyOutlined,
  ApiOutlined,
  BarsOutlined,
  SearchOutlined,
  SettingOutlined,
  LogoutOutlined,
  CloseOutlined
} from '@ant-design/icons';
import { useNavigate, useLocation } from 'react-router-dom';

export default function DashboardLayout({ children }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const agentMenuItems = [
    { icon: <HomeOutlined />, label: 'Dashboard', path: '/', emoji: '🏠' },
    { icon: <FileTextOutlined />, label: 'Literature', path: '/literature', emoji: '📚' },
    { icon: <ThunderboltOutlined />, label: 'Clinical', path: '/clinical', emoji: '🏥' },
    { icon: <SafetyOutlined />, label: 'Safety', path: '/safety', emoji: '🔒' },
    { icon: <ExperimentOutlined />, label: 'Molecular', path: '/molecular', emoji: '🧬' },
    { icon: <FileTextOutlined />, label: 'Patents', path: '/patent', emoji: '📋' },
    { icon: <ApiOutlined />, label: 'Market', path: '/market', emoji: '💼' }
  ];

  const handleNavigation = (path) => {
    navigate(path);
    setSidebarOpen(false);
  };

  const SidebarContent = () => (
    <div className="sidebar-nav">
      <div className="sidebar-header">
        <div className="brand-icon-mini">🧬</div>
        <div className="brand-text-mini">
          <div className="brand-name-mini">PharmAI</div>
        </div>
      </div>
      <nav className="nav-menu">
        {agentMenuItems.map((item) => (
          <button
            key={item.path}
            className={`nav-item ${location.pathname === item.path ? 'active' : ''}`}
            onClick={() => handleNavigation(item.path)}
            title={item.label}
          >
            <span className="nav-emoji">{item.emoji}</span>
            <span className="nav-label">{item.label}</span>
          </button>
        ))}
      </nav>
    </div>
  );

  return (
    <div className="dashboard-layout-container">
      {/* Professional Header */}
      <header className="dashboard-header">
        <div className="header-left">
          <Button
            type="text"
            icon={<BarsOutlined size={20} />}
            className="sidebar-toggle-btn"
            onClick={() => setSidebarOpen(true)}
          />
          <div className="brand-section">
            <div className="brand-icon">🧬</div>
            <div className="brand-text">
              <div className="brand-name">PharmAI Insights</div>
              <div className="brand-subtitle">AI-Powered Drug Discovery</div>
            </div>
          </div>
        </div>

        <div className="header-center">
          <Input
            prefix={<SearchOutlined />}
            placeholder="Search compounds, targets, trials..."
            className="global-search"
            allowClear
          />
        </div>

        <div className="header-right">
          <Tooltip title="Settings">
            <Button type="text" icon={<SettingOutlined />} className="header-icon-btn" />
          </Tooltip>
          <Tooltip title="Logout">
            <Button type="text" icon={<LogoutOutlined />} className="header-icon-btn" />
          </Tooltip>
        </div>
      </header>

      {/* Desktop Sidebar */}
      <div className="sidebar-wrapper">
        <SidebarContent />
      </div>

      {/* Mobile Drawer */}
      <Drawer
        title="Navigation"
        placement="left"
        onClose={() => setSidebarOpen(false)}
        open={sidebarOpen}
        width={280}
        bodyStyle={{ padding: 0 }}
        headerStyle={{ borderBottom: '1px solid #e2e8f0' }}
      >
        <SidebarContent />
      </Drawer>

      {/* Main Content */}
      <main className="dashboard-main">
        <div className="content-wrapper">
          {children}
        </div>
      </main>
    </div>
  );
}
