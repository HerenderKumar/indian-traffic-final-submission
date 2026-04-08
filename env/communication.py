# env/communication.py
class CommunicationLayer:
    def __init__(self, map_config):
        self.topology = map_config.topology

    def get_messages(self, current_obs_dict):
        """
        Extracts N, S, E, W neighbor congestion based on physical topology.
        Returns a dict of neighbor vectors per agent.
        """
        messages = {}
        for jid in self.topology.keys():
            neighbor_vec = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}
            for direction, neighbor_id in self.topology[jid].items():
                if neighbor_id in current_obs_dict:
                    # Mock extracting congestion from neighbor's obs
                    # In reality, this reads obs_dict[neighbor_id][:48].mean()
                    neighbor_vec[direction] = 0.5 
            messages[jid] = neighbor_vec
        return messages