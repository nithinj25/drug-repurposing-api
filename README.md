# Drug Repurposing Assistant

## Overview
The Drug Repurposing Assistant is an Agentic AI tool designed to facilitate the identification of existing drugs that can be repurposed for new therapeutic uses. By leveraging specialized agents, the assistant analyzes literature, molecular data, clinical trials, and safety information to provide comprehensive insights.

## Project Structure
```
drug-repurposing-assistant
├── src
│   ├── main.py                  # Entry point for the application
│   ├── agents                   # Contains specialized agents
│   │   ├── __init__.py
│   │   ├── literature_agent.py   # Queries literature and extracts evidence
│   │   ├── molecular_agent.py    # Analyzes chemical structures and bioactivity
│   │   ├── clinical_agent.py     # Searches and processes clinical trial data
│   │   ├── safety_agent.py       # Aggregates safety and toxicology data
│   │   └── coordinator_agent.py   # Orchestrates the workflow of agents
│   ├── tools                    # Utility functions for agents
│   │   ├── __init__.py
│   │   ├── pubmed_tools.py       # Interacts with PubMed APIs
│   │   ├── molecular_tools.py     # Handles chemical structure data
│   │   ├── clinical_tools.py      # Processes clinical trial data
│   │   └── database_tools.py      # Manages database interactions
│   ├── graphs                   # Defines the workflow for agents
│   │   ├── __init__.py
│   │   └── workflow.py
│   ├── config                   # Configuration settings
│   │   ├── __init__.py
│   │   └── settings.py
│   └── utils                    # Helper functions
│       ├── __init__.py
│       └── helpers.py
├── requirements.txt             # Project dependencies
├── .env.example                  # Example environment variables
└── README.md                    # Project documentation
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd drug-repurposing-assistant
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables by copying `.env.example` to `.env` and filling in the necessary values.

## Usage
To run the Drug Repurposing Assistant, execute the following command:
```
python src/main.py
```

## Agents
- **LiteratureAgent**: Queries PubMed/PMC for relevant literature, extracts claim-level sentences, and returns evidence items with confidence scores.
- **MolecularAgent**: Analyzes chemical structures and bioactivity to identify targets, pathway links, and mechanistic plausibility summaries.
- **ClinicalTrialsAgent**: Searches clinical trial registries, normalizes trial records, and extracts structured trial evidence.
- **SafetyAgent**: Aggregates adverse event data, normalizes terms, and computes safety feasibility scores.
- **CoordinatorAgent**: Orchestrates the workflow, dispatches tasks to agents, and validates outputs.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.