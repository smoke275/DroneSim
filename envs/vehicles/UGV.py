class UGV:
    def __init__(self, ugv_id, base_position, max_range):
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
        self.move_approved = None  # flag to check if step is approved

        self.tmp_position = None
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
        # if action !=4 and self.current_range < distance:
        #     return False
        moves = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1),  # Left
            4: (0, 0)    # Stay
        }
            
        dx, dy = moves[action]
        self.prev_position = self.position
        self.tmp_position = (self.position[0] + dx, self.position[1] + dy)
        return True

    def update_move(self, distance):
        if self.move_approved:
            d = abs(self.tmp_position[0] - self.position[0]) + abs(self.tmp_position[1] - self.position[1])
            self.current_range -= d*distance
            self.distance_traveled += d*distance
            self.current_range_percent = self.current_range / self.max_range
            self.position = self.tmp_position
        self.move_approved = None
        self.tmp_position = None

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
    ugv = UGV(agent_id=1, initial_position=(5, 5), max_range=100)
    
    # Print out the agent and its action space.
    print(ugv)
    print("Action Space:", ugv.action_space)