"""The SFace loss: what makes the network learn good embeddings.

Paper: Zhong et al., "SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition",
IEEE Transactions on Image Processing, 2021. Official code: github.com/zhongyy/SFace
(SFace_torch/head/metrics.py, class SFaceLoss). This file does the same calculation.

The idea, step by step:

1. Every person in the training set (10,572 of them) gets a "class centre": a learnable direction
   with 128 numbers, stored in self.centres. Embeddings and centres are both scaled to length 1,
   so they all lie on a sphere (the "hypersphere").
2. For each face we measure the angle between its embedding and every centre (via cosine).
   Good embeddings have a SMALL angle to their own person's centre ("intra-class")
   and LARGE angles to everyone else's centre ("inter-class").
3. The loss pulls the face towards its own centre and pushes it away from the others:
       loss = -s * cos(angle to own centre) * pull_weight
              + sum over other people of  s * cos(angle to their centre) * push_weight
4. SFace's trick is in the two weights, which are sigmoid curves of the angle:
       pull_weight = sigmoid(k * (angle_own - a))      close to 1 while the angle is above a,
                                                       fading to 0 once the face is close enough
       push_weight = sigmoid(-k * (angle_other - b))   close to 1 while another centre is nearer than b,
                                                       fading to 0 once it is far enough away
   So the network stops pulling and pushing once a face is "good enough". Older losses keep
   pushing forever, which makes the network overfit to noisy, mislabelled web photos.
   The weights are computed under torch.no_grad(): they only decide HOW HARD to pull or push,
   and no learning flows through them.

Settings (the official ones): s = 64 (scale), k = 80 (how steep the sigmoids are),
a = 0.87 and b = 1.2 radians (about 50 and 69 degrees), from the official CASIA-WebFace command.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class SFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_people, s=64.0, k=80.0, a=0.87, b=1.2):
        super().__init__()
        self.s, self.k, self.a, self.b = s, k, a, b
        self.centres = nn.Parameter(torch.empty(num_people, embedding_size))
        # Same starting values as the official code (normal, standard deviation 2 * sqrt(2 / num_people)).
        nn.init.normal_(self.centres, 0.0, 2.0 * math.sqrt(2.0 / num_people))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Returns (loss, cosines). cosines has one row per face and one column per person."""
        cosines = F.linear(F.normalize(embeddings), F.normalize(self.centres))
        own_cosine = cosines.gather(1, labels[:, None]).squeeze(1)

        with torch.no_grad():
            own_angle = torch.acos(own_cosine.clamp(-1.0, 1.0))
            pull_weight = torch.sigmoid(self.k * (own_angle - self.a))
            other_angles = torch.acos(cosines.clamp(-1.0, 1.0))
            push_weights = torch.sigmoid(-self.k * (other_angles - self.b))
            push_weights.scatter_(1, labels[:, None], 0.0)  # a face is never pushed away from its own person

        pull_loss = -(pull_weight * self.s * own_cosine).mean()
        push_loss = (push_weights * self.s * cosines).sum(dim=1).mean()
        return pull_loss + push_loss, cosines
