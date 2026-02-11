import React from 'react';

function Header() {
  return (
    <header className="header">
      <div className="header-content">
        <div className="brand">
          <span className="brand-mark">*</span>
          <div>
            <h1>Accelerate Drug Discovery</h1>
            <p className="subtitle">AI-powered repurposing insights across literature, trials, and safety.</p>
          </div>
        </div>
        <nav className="header-nav">
          <a href="#features">Capabilities</a>
          <a href="#workflow">How It Works</a>
          <a href="#analysis">Launch Analysis</a>
        </nav>
      </div>
    </header>
  );
}

export default Header;
