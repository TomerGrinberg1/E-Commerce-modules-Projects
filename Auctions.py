class BiddingAgent1:

    def __init__(self):
        self.v = None
        self.dummy_bids = None
        self.position = None

    def get_bid(self, num_of_agents, P, q, v):
        self.v = v
        self.dummy_bids = [self.v / (i + 1) for i in range(3)]  # Adjusted for 3 dummy agents

        expected_profits = [P[i] * q * self.v - self.dummy_bids[i % 3]  for i in range(len(P))]

        max_profit_position = expected_profits.index(max(expected_profits))

        if max_profit_position >= num_of_agents:
            self.position = -1
        else:
            self.position = max_profit_position

        return self.dummy_bids[self.position]

    def notify_outcome(self, reward, outcome, position):
        self.position = position

        if outcome > 0:
            if position == self.position:
                self.v -= reward / (self.position + 1)  # Adjust value based on reward and position
        else:
            pass

    def get_id(self):
        return "id_318155843_302342498"


class BiddingAgent2:
    class BiddingAgent:

        def __init__(self):
            self.v = None
            self.position = None
            self.estimated_dummy_bids = None

        def get_bid(self, num_of_agents, P, q, v):

            self.v = v
            self.estimated_dummy_bids = [(v / 3) + 1 for _ in range(num_of_agents)] # asuming that every one will bid that
            expected_profits = [P[i] * q * self.v - self.estimated_dummy_bids[i % num_of_agents] for i in range(len(P))]

            max_profit_position = expected_profits.index(max(expected_profits))

            if max_profit_position >= num_of_agents:
                self.position = -1
            else:
                self.position = max_profit_position

            return self.v

        def notify_outcome(self, reward, outcome, position):
            self.position = position

            if outcome > 0:
                if position == self.position:
                    self.v -= reward / (self.position + 1)  # Adjust value based on reward and position
            else:
                pass

        def get_id(self):
            return "id_318155843_302342498"
