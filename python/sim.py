"""
Simulación en lazo cerrado (por lotes) del Qube-Servo 3 no lineal.
Se usa para entrenamiento (con gradiente) y para evaluación (sin gradiente).
"""

from __future__ import annotations

import numpy as np
import torch

from qube_model import BatchParams, QubeParams, SensorModel, rk4_step


def lqr_controller(K, vmax):
    """u = sat(-K x). K: array (1,4)."""
    Kt = torch.tensor(np.asarray(K, dtype=np.float32).reshape(4))

    def ctrl(y):
        return torch.clamp(-(y * Kt.to(y)).sum(dim=1), -vmax, vmax)
    return ctrl


def sample_initial_states(n, ranges, generator=None, device="cpu"):
    """Estados iniciales uniformes en ±ranges = (theta, alpha, theta_d, alpha_d)."""
    r = torch.tensor(ranges, dtype=torch.float32, device=device)
    return (torch.rand(n, 4, generator=generator, device=device) * 2 - 1) * r


def rollout(controller, x0, p: BatchParams, steps, dt, sensor: SensorModel | None = None,
            record=True, u_noise_std=0.0, mix_ctrl=None, beta=0.0):
    """Simula `steps` pasos de control (ZOH, RK4 a dt).

    controller : callable y(B,4) -> u(B,)
    sensor     : modelo de medición; None => estado perfecto
    mix_ctrl   : controlador experto para DAgger; u = beta*u_exp + (1-beta)*u
    Devuelve dict con X (T+1,B,4), Y (T,B,4), U (T,B).
    """
    x = x0
    y = sensor.reset(x) if sensor is not None else x
    X, Y, U = [x], [], []
    for _ in range(steps):
        u = controller(y)
        if mix_ctrl is not None and beta > 0:
            u = beta * mix_ctrl(x) + (1 - beta) * u
        if u_noise_std > 0:
            u = u + u_noise_std * torch.randn_like(u)
        vmax = p.base.Vmax
        u = torch.clamp(u, -vmax, vmax)
        if record:
            Y.append(y)
            U.append(u)
        x = rk4_step(x, u, p, dt)
        y = sensor.measure(x, dt) if sensor is not None else x
        if record:
            X.append(x)
    if not record:
        return {"x_final": x}
    return {"X": torch.stack(X), "Y": torch.stack(Y), "U": torch.stack(U)}


def success_mask(X, alpha_tol_deg=3.0, theta_tol_deg=15.0, fall_deg=60.0):
    """Éxito: nunca |alpha|>fall_deg y al final |alpha|<tol, |theta|<tol."""
    a = torch.rad2deg(X[..., 1].abs())
    th = torch.rad2deg(X[..., 0].abs())
    never_fell = (a.max(dim=0).values < fall_deg)
    tail = X.shape[0] // 10
    settled = (a[-tail:].max(dim=0).values < alpha_tol_deg) & \
              (th[-tail:].max(dim=0).values < theta_tol_deg)
    return never_fell & settled


def default_params(**kw):
    return QubeParams(**kw)
