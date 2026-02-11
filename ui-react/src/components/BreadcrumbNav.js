import React from 'react';
import { Breadcrumb } from 'antd';
import { HomeOutlined, FileTextOutlined } from '@ant-design/icons';
import { useNavigate, useLocation } from 'react-router-dom';

export default function BreadcrumbNav({ items = [] }) {
  const navigate = useNavigate();
  const location = useLocation();

  const agentNames = {
    literature: '📚 Literature',
    clinical: '🏥 Clinical',
    safety: '🔒 Safety',
    molecular: '🧬 Molecular',
    patent: '📋 Patents',
    market: '💼 Market'
  };

  // Auto-generate breadcrumb items based on URL
  const defaultItems = [];
  
  defaultItems.push({
    title: (
      <span onClick={() => navigate('/')} style={{ cursor: 'pointer' }}>
        Dashboard
      </span>
    ),
    onClick: () => navigate('/')
  });

  const pathParts = location.pathname.split('/').filter(Boolean);
  
  if (pathParts.length >= 2 && pathParts[0] === 'jobs') {
    defaultItems.push({
      title: 'Analysis',
      onClick: () => navigate(`/jobs/${pathParts[1]}`)
    });

    if (pathParts.length >= 3) {
      const agentType = pathParts[2];
      defaultItems.push({
        title: agentNames[agentType] || agentType
      });
    }
  }

  const breadcrumbItems = items.length > 0 ? items : defaultItems;

  return (
    <Breadcrumb 
      className="page-breadcrumb"
      items={breadcrumbItems}
      separator="/"
    />
  );
}
