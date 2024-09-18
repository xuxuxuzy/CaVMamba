import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):

    def __init__(self, input_dim: int = 512,
                 expansion_factor: int = 4,
                 drop_prob: int = 0.1):
        super().__init__()
        self._layers(input_dim, expansion_factor, drop_prob)

    def _layers(self, input_dim, expansion_factor, drop_prob):
        self.linear1 = nn.Linear(input_dim, expansion_factor)
        self.dropout1 = nn.Dropout(drop_prob)
        self.linear2 = nn.Linear(expansion_factor, input_dim)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x):
        # Swish Activation Function
        # x = self.dropout1(F.silu(self.linear1(x)))
        # x = self.dropout2(self.linear2(x))
        x = F.silu(self.linear1(x))
        x = self.linear2(x)
        return x



#
# class FeedForward(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         hidden_features = int(2 * hidden_features / 3)
#         self.fc1 = nn.Linear(in_features, hidden_features * 2)
#         # self.dwconv = nn.Conv2d(hidden_features,hidden_features,3,1,1,groups=hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x, v = self.fc1(x).chunk(2, dim=-1)
#         # x = self.act(self.dwconv(x)) * v
#         x = self.act(x) * v
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x