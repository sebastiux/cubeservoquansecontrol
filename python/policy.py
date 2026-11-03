"""
Red neuronal del controlador (política determinista).

    g(x) = W3 * tanh( W2 * tanh( W1 * (x ./ x_scale) + b1 ) + b2 ) + b3
    u    = Vmax * tanh( (g(x) - g(-x)) / 2 )

La parte (g(x) - g(-x))/2 hace a la red IMPAR: u(-x) = -u(x), igual que la
física del péndulo (simetría izquierda/derecha). Garantiza u(0) = 0 exacto:
sin esto, un sesgo de solo 0.1 V movería el brazo ~18° en estado estacionario
(porque la ganancia LQR de theta es muy pequeña, K_theta ~ -0.32 V/rad).

Es un MLP pequeño (4-64-64-1) a propósito: se evalúa con multiplicaciones de
matrices puras dentro de un bloque "MATLAB Function" de Simulink, que es
compatible con generación de código (QUARC) y corre sin problema a 500 Hz.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Escalas de normalización de la entrada [theta, alpha, theta_dot, alpha_dot]
X_SCALE = (1.0, 0.35, 5.0, 10.0)


class NNController(nn.Module):
    def __init__(self, hidden=(64, 64), vmax=10.0, x_scale=X_SCALE):
        super().__init__()
        self.vmax = float(vmax)
        self.register_buffer("x_scale", torch.tensor(x_scale, dtype=torch.float32))
        layers, n_in = [], 4
        for h in hidden:
            layers += [nn.Linear(n_in, h), nn.Tanh()]
            n_in = h
        layers += [nn.Linear(n_in, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,4) -> u: (B,) en voltios, saturado suavemente a ±Vmax."""
        xs = x / self.x_scale
        z = 0.5 * (self.net(xs) - self.net(-xs))
        return self.vmax * torch.tanh(z.squeeze(-1))

    def linear_layers(self):
        return [m for m in self.net if isinstance(m, nn.Linear)]
