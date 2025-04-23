class UGV:
    def __init__(self, ugv_id, base_position, max_range, max_load, drain_rate, G):
        super().__init__()
        self.agent_id = ugv_id
        self.position = base_position  # e.g., (row, col) or (x, y)
        self.base_point = base_position  # Base station position
        self.max_range = max_range       # maximum battery capacity or range
        self.current_range = self.max_range    # start fully charged
        self.current_range_percent = 1.0
        self.task_list = []               # tasks assigned to the UGV
        self.path = []                    # planned path (list of positions)
        self.task_timer = 0            # time spent on the current task
        self.active_task = None        # current task being executed
        self.distance_traveled = 0.0  # distance traveled by the UGV
        self.G = G
        self.max_load = max_load
        self.drain_rate= drain_rate

        self.load = 0

        self.prev_position = None


    def move(self, action, distance=1):
        """
        Move the agent based on action number:
        0: Up
        1: Right
        2: Down
        3: Left
        4: Stay
        Returns True if move was successful, False if out of range
        """
        curr_drain_rate = self.drain_rate * (1 + self.load / self.max_load)
        range_left = self.current_range - curr_drain_rate * distance
        if action !=4 and range_left<=0:
            return 0
        moves = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1),  # Left
            4: (0, 0)    # Stay
        }
            
        dy, dx = moves[action]
        self.prev_position = self.position
        tmp_position = (self.position[0] + dy, self.position[1] + dx)
        if tmp_position in self.G.nodes and (tmp_position, self.position) in self.G.edges:
            self.position = tmp_position
            self.distance_traveled += distance

            self.current_range = range_left
            self.current_range_percent = self.current_range / self.max_range
            return 1
        return 0

    def recharge(self):
        """
        Recharge the agent to its maximum range.
        """
        self.current_range = self.max_range
        self.current_range_percent = 1.0

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
    ugv = UGV(agent_id=1, initial_position=(5, 5), max_range=100)
    
    # Print out the agent and its action space.
    print(ugv)
    print("Action Space:", ugv.action_space)