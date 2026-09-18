"""
CNN–QNN hybrid classifier (Section 3.4).
ResNet-18 feature extractor → angle encoding → 2-layer variational circuit → classical head.
"""

import torch
import torch.nn as nn
from torchvision import models
# Qiskit / PennyLane or TorchQuantum can be used for the quantum layer.
# Below is a pure-PyTorch placeholder that mimics the quantum feature map.

class HybridCNNQNN(nn.Module):
    def __init__(self, num_classes=2, n_qubits=4):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # global avg pool
        self.proj = nn.Linear(512, n_qubits)
        # Quantum layer would be inserted here (angle encoding + variational ansatz)
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits + 8, 64),  # + IDE features
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, ide_features):
        f = self.features(x).flatten(1)
        z = torch.tanh(self.proj(f))          # normalised features for encoding
        # z_Q = quantum_circuit(z)            # replace with real QNN
        z_Q = z                               # placeholder
        fused = torch.cat([z_Q, ide_features], dim=1)
        return self.classifier(fused)
