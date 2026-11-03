"""
Modelo NO LINEAL del péndulo rotatorio (Furuta) Quanser Qube-Servo 3.

Convenciones (idénticas a ProyectoControlFinal.m y al Simulink qs3_lqr_ctrl):
    x = [theta, alpha, theta_dot, alpha_dot]
    theta : ángulo del brazo rotatorio (rad)
    alpha : ángulo del péndulo, alpha = 0 -> péndulo ARRIBA (invertido)
    u     : voltaje del motor Vm (V)

Ecuaciones de Euler-Lagrange (inercias Jr, Jp medidas respecto al pivote,
igual que en tu script .m). Linealizando en x = 0 se obtienen EXACTAMENTE
las matrices A y B de ProyectoControlFinal.m (se verifica en
`linearize_numeric`).

    (Jr + mp r^2 + Jp sin^2 a) th'' - mp l r cos a a''
        + 2 Jp sin a cos a th' a' + mp l r sin a a'^2 = tau - br th'
    -mp l r cos a th'' + Jp a'' - Jp sin a cos a th'^2 - mp g l sin a = -bp a'

    tau = km (Vm - km th') / Rm      (inductancia despreciada: L/R = 0.15 ms)

Todo está escrito en PyTorch y vectorizado por lotes (batch) para poder:
  * simular miles de trayectorias en paralelo,
  * aleatorizar parámetros por trayectoria (domain randomization),
  * diferenciar a través de la simulación (BPTT) durante el entrenamiento.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import scipy.linalg as sla
import torch


# --------------------------------------------------------------------------
# Parámetros
# --------------------------------------------------------------------------
@dataclass
class QubeParams:
    # Motor (hoja de datos AM-CL2642MAE-2405 / tu script)
    Rm: float = 7.5          # Resistencia de armadura (Ohm)
    kt: float = 0.0422       # Constante de par (N·m/A)
    km: float = 0.0422       # Constante contra-electromotriz (V·s/rad)
    # Brazo rotatorio
    mr: float = 0.095        # Masa (kg)
    r: float = 0.085         # Longitud total (m)
    br: float = 1e-3         # Amortiguamiento viscoso (N·m·s/rad)
    # Péndulo
    mp: float = 0.024        # Masa (kg)
    Lp: float = 0.129        # Longitud total (m)
    bp: float = 5e-5         # Amortiguamiento viscoso (N·m·s/rad)
    g: float = 9.81
    # Actuación / muestreo
    Vmax: float = 10.0       # Saturación de voltaje usada en entrenamiento (V)
    Ts: float = 0.002        # Paso fijo del Simulink (500 Hz)
    enc_counts: int = 2048   # Cuentas/rev en cuadratura (512 CPR x 4)

    @property
    def l(self) -> float:    # Centro de masa del péndulo
        return self.Lp / 2

    @property
    def Jr(self) -> float:   # Inercia del brazo respecto al pivote
        return self.mr * self.r ** 2 / 3

    @property
    def Jp(self) -> float:   # Inercia del péndulo respecto a su pivote
        return self.mp * self.Lp ** 2 / 3

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(l=self.l, Jr=self.Jr, Jp=self.Jp)
        return d


# Nombres de los parámetros físicos que se aleatorizan
PHYS_KEYS = ("Rm", "km", "mr", "r", "br", "mp", "Lp", "bp", "g")


def linear_model(p: QubeParams):
    """Matrices A, B linealizadas (mismas fórmulas que ProyectoControlFinal.m)."""
    Rm, km, r, br = p.Rm, p.km, p.r, p.br
    mp, l, bp, g = p.mp, p.l, p.bp, p.g
    Jr, Jp = p.Jr, p.Jp
    Jt = (Jr + mp * r ** 2) * Jp - mp ** 2 * l ** 2 * r ** 2
    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, mp ** 2 * l ** 2 * r * g / Jt,
         -(br * Jp) / Jt - km ** 2 * Jp / (Rm * Jt),
         -(mp * l * r * bp) / Jt],
        [0, mp * g * l * (Jr + mp * r ** 2) / Jt,
         -(mp * l * r * br) / Jt - km ** 2 * mp * l * r / (Rm * Jt),
         -(Jr + mp * r ** 2) * bp / Jt],
    ])
    B = np.array([[0.0], [0.0], [km * Jp / (Rm * Jt)], [km * mp * l * r / (Rm * Jt)]])
    return A, B


def lqr(A, B, Q, R):
    """LQR continuo. Devuelve K (1x4) y P (solución de Riccati)."""
    P = sla.solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, P


# Pesos LQR de tu proyecto
Q_LQR = np.diag([1.0, 20.0, 0.1, 0.1])
R_LQR = np.array([[10.0]])


# --------------------------------------------------------------------------
# Parámetros en forma de tensores (uno por trayectoria del batch)
# --------------------------------------------------------------------------
class BatchParams:
    """Parámetros físicos como tensores (B,) -> permite domain randomization."""

    def __init__(self, base: QubeParams, batch: int, rel_spread: float = 0.0,
                 device="cpu", dtype=torch.float32, generator=None):
        self.base = base
        for k in PHYS_KEYS:
            v = torch.full((batch,), getattr(base, k), device=device, dtype=dtype)
            if rel_spread > 0 and k != "g":
                noise = (torch.rand(batch, generator=generator, device=device,
                                    dtype=dtype) * 2 - 1) * rel_spread
                v = v * (1 + noise)
            setattr(self, k, v)
        self.l = self.Lp / 2
        self.Jr = self.mr * self.r ** 2 / 3
        self.Jp = self.mp * self.Lp ** 2 / 3


def dynamics(x: torch.Tensor, u: torch.Tensor, p: BatchParams) -> torch.Tensor:
    """dx/dt = f(x, u) no lineal. x: (B,4), u: (B,) -> (B,4)."""
    th_d, al, al_d = x[:, 2], x[:, 1], x[:, 3]
    s, c = torch.sin(al), torch.cos(al)

    tau = p.km * (u - p.km * th_d) / p.Rm

    m11 = p.Jr + p.mp * p.r ** 2 + p.Jp * s ** 2
    m12 = -p.mp * p.l * p.r * c
    m22 = p.Jp

    c1 = tau - p.br * th_d - 2 * p.Jp * s * c * th_d * al_d - p.mp * p.l * p.r * s * al_d ** 2
    c2 = -p.bp * al_d + p.Jp * s * c * th_d ** 2 + p.mp * p.g * p.l * s

    det = m11 * m22 - m12 ** 2
    th_dd = (m22 * c1 - m12 * c2) / det
    al_dd = (-m12 * c1 + m11 * c2) / det
    return torch.stack([th_d, al_d, th_dd, al_dd], dim=1)


def rk4_step(x, u, p, dt):
    """Un paso RK4 con u retenido (ZOH) durante dt."""
    k1 = dynamics(x, u, p)
    k2 = dynamics(x + 0.5 * dt * k1, u, p)
    k3 = dynamics(x + 0.5 * dt * k2, u, p)
    k4 = dynamics(x + dt * k3, u, p)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def linearize_numeric(p: QubeParams, eps=1e-6):
    """Jacobiano numérico del modelo no lineal en x=0 (para verificación)."""
    bp_ = BatchParams(p, 1, dtype=torch.float64)
    x0 = torch.zeros(1, 4, dtype=torch.float64)
    u0 = torch.zeros(1, dtype=torch.float64)
    A = np.zeros((4, 4))
    for i in range(4):
        dx = torch.zeros(1, 4, dtype=torch.float64)
        dx[0, i] = eps
        A[:, i] = ((dynamics(x0 + dx, u0, bp_) - dynamics(x0 - dx, u0, bp_)) / (2 * eps))[0].numpy()
    du = torch.full((1,), eps, dtype=torch.float64)
    B = ((dynamics(x0, u0 + du, bp_) - dynamics(x0, u0 - du, bp_)) / (2 * eps))[0].numpy()
    return A, B.reshape(4, 1)
