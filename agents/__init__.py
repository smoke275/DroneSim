from agents.sarsa import SARSAAgent
from agents.dijkstra import DijkstraAgent

def get_agent(config, env):
    """
    Returns the appropriate agent based on the configuration.
    :param config: Configuration dictionary.
    :param env: The environment instance.
    :return: An agent instance or a list of agents for multi-agent environments.
    """
    algo = config['simulation']['algo']
    num_agents = config['fleet']['ugv']['count']

    if algo == 'dijkstra':
        return [DijkstraAgent(i, env) for i in range(num_agents)]
    
    if config['simulation']['train'] == 'true':
        if algo == 'sarsa':
            return SARSAAgent(env, config=config)
    
    else:
        policy_name = config['simulation']['policy']
        policy_path = f"runs/{policy_name}/policy.pkl"
        if algo == 'sarsa':
            return [SARSAAgent(env, config=config, policy_path=policy_path) for _ in range(num_agents)]
        
