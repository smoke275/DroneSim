from agent import Agent
from gymnasium import spaces

class UAVAgent(Agent):
    def __init__(self, config, agent_id, base_position):
        super().__init__()
        self.agent_id = agent_id
        self.position = base_position  # e.g., (row, col) or (x, y)
        self.base_point = base_position
        self.max_range = config["range"]        # maximum battery capacity or range
        self.current_range = self.max_range    # start fully charged
        self.current_range_percent = 1.0  # percentage of battery remaining
        self.task_list = []               # tasks assigned to the UGV
        self.path = []                    # planned path (list of positions)
        self.task_timer = 0            # time spent on the current task
        self.active_task = None        # current task being executed

        self.action_space = spaces.Discrete(5)

    def move_to(self, new_position, cost):
        """
        Update the agent's position and reduce the current range by the movement cost.
        """
        self.position = new_position
        self.current_range -= cost
        if self.current_range < 0:
            self.current_range = 0

    def recharge(self):
        """
        Recharge the agent to its maximum range.
        """
        self.current_range = self.max_range

    def assign_task(self, task):
        """
        Add a new task to the agent's task list.
        """
        self.task_list.append(task)

    def update_path(self, new_path):
        """
        Update the planned path for the agent.
        """
        self.path = new_path

    def __str__(self):
        return (f"UGVAgent(id={self.agent_id}, position={self.position}, "
                f"current_range={self.current_range}/{self.max_range}, tasks={self.task_list})")

# Example usage:
if __name__ == "__main__":
    # Create a UGVAgent with ID 1, initial position (5, 5) and a max range of 100.
    ugv = UAVAgent(agent_id=1, initial_position=(5, 5), max_range=100)
    
    # Print out the agent and its action space.
    print(ugv)
    print("Action Space:", ugv.action_space)