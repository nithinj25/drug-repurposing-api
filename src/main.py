from agents.coordinator_agent import MasterAgent

def main():
    # Initialize the Master Agent
    master_agent = MasterAgent()

    # Start the workflow
    master_agent.run()

if __name__ == "__main__":
    main()