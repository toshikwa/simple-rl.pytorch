from abc import ABC, abstractmethod
import torch


class Algorithm(ABC):

    @abstractmethod
    def is_update(self, steps):
        pass

    @abstractmethod
    def explore(self, state):
        pass

    def exploit(self, state):
        state = torch.tensor(
            state.copy(), dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

    @abstractmethod
    def update(self):
        pass
