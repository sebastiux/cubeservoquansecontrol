"""
Entorno RL (vectorizado) del Qube-Servo 3 NO LINEAL para SWING-UP + BALANCE:
el péndulo empieza colgando y la red debe levantarlo y mantenerlo arriba.

Diferencias con rl_env.py (solo balance):
  * alpha puede dar la vuelta completa. Observación de 5 entradas:
        [theta, sin(alpha), cos(alpha), theta_dot, alpha_dot]
    (sin/cos evita el salto de +-180° del ángulo envuelto del Simulink).
  * La velocidad del péndulo se estima con TU observador de 3er orden, pero
    alimentado con alpha DESENROLLADO (sin el salto de 2*pi), integrado con
    Euler a 2 ms. Es exactamente lo que hace el bloque dqn_swingup_block.m.
  * El episodio termina (con castigo) si el brazo pasa de theta_lim.
  * Recompensa densa para subir + bono al mantenerse arriba y centrado.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from qube_model import BatchParams, QubeParams, rk4_step

DEG = math.pi / 180

# Normalización de [theta, sin(alpha), cos(alpha), theta_dot, alpha_dot]
X_SCALE_SU = np.array([1.0, 1.0, 1.0, 5.0, 10.0])
# Para reflejar un estado: (theta, sin, cos, th_d, al_d) -> (-, -, +, -, -)
ESPEJO = np.array([-1.0, -1.0, 1.0, -1.0, -1.0])


def envolver(a):
    """Ángulo a (-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))


class QubeSwingUpEnv:
    def __init__(self, n_envs: int, acciones, Ts_agente: float = 0.01,
                 params: QubeParams | None = None, rel_spread: float = 0.15,
                 vel_noise: float = 0.05, theta_lim_deg: float = 120.0,
                 t_max: float = 6.0, p_abajo: float = 0.7, seed: int = 0,
                 r_terminal: float = -20.0):
        self.p = params or QubeParams()
        self.n = n_envs
        self.acciones = torch.as_tensor(np.asarray(acciones, dtype=np.float32))
        self.dt = self.p.Ts
        self.substeps = max(1, int(round(Ts_agente / self.dt)))
        self.Ts_agente = self.substeps * self.dt
        self.rel_spread = rel_spread
        self.vel_noise = vel_noise
        self.theta_lim = theta_lim_deg * DEG
        self.max_steps = int(round(t_max / self.Ts_agente))
        self.p_abajo = p_abajo
        self.r_terminal = r_terminal
        self.gen = torch.Generator().manual_seed(seed)
        self.q_enc = 2 * math.pi / self.p.enc_counts
        a = 100.0                                   # ProyectoControlFinal.m
        self.obs_beta, self.obs_L = 3 * a, 3 * a ** 2
        self.obs_m = a ** 3 - self.obs_L * self.obs_beta

        self.x = torch.zeros(n_envs, 4)             # alpha SIN envolver, 0 = arriba
        self.y = torch.zeros(n_envs, 2)             # ángulos medidos (cuantizados)
        self.vel = torch.zeros(n_envs, 2)           # velocidades estimadas
        self.p_o = torch.zeros(n_envs, 2)
        self.v_o = torch.zeros(n_envs, 2)
        self.z_o = torch.zeros(n_envs, 2)
        self.t = torch.zeros(n_envs, dtype=torch.long)
        self.bp = BatchParams(self.p, n_envs, rel_spread, generator=self.gen)
        self.reset(torch.ones(n_envs, dtype=torch.bool))

    # ------------------------------------------------------------------
    def _u(self, k, lo, hi):
        return lo + (hi - lo) * torch.rand(k, generator=self.gen)

    def estados_iniciales(self, k):
        """70 %: colgando casi quieto. 30 %: cualquier ángulo y velocidad."""
        x0 = torch.zeros(k, 4)
        abajo = torch.rand(k, generator=self.gen) < self.p_abajo
        signo = torch.where(torch.rand(k, generator=self.gen) < 0.5, -1.0, 1.0)
        x0[:, 0] = torch.where(abajo, self._u(k, -0.3, 0.3), self._u(k, -1.2, 1.2))
        x0[:, 1] = torch.where(abajo, signo * math.pi + self._u(k, -0.2, 0.2),
                               self._u(k, -math.pi, math.pi))
        x0[:, 2] = torch.where(abajo, self._u(k, -0.5, 0.5), self._u(k, -3.0, 3.0))
        x0[:, 3] = torch.where(abajo, self._u(k, -1.0, 1.0), self._u(k, -8.0, 8.0))
        return x0

    def reset(self, mask: torch.Tensor, x0: torch.Tensor | None = None):
        k = int(mask.sum())
        if k == 0:
            return self.observe()
        if x0 is None:
            x0 = self.estados_iniciales(k)
        self.x[mask] = x0
        self.t[mask] = 0
        nb = BatchParams(self.p, k, self.rel_spread, generator=self.gen)
        for name in ("Rm", "km", "mr", "r", "br", "mp", "Lp", "bp", "g", "l", "Jr", "Jp"):
            getattr(self.bp, name)[mask] = getattr(nb, name)
        y0 = torch.round(x0[:, 0:2] / self.q_enc) * self.q_enc
        self.y[mask] = y0
        self.vel[mask] = x0[:, 2:4]
        self.p_o[mask] = y0                         # observador convergido
        self.v_o[mask] = x0[:, 2:4]
        self.z_o[mask] = 0.0
        return self.observe()

    def _update_sensor(self):
        y = torch.round(self.x[:, 0:2] / self.q_enc) * self.q_enc
        self.vel = self.v_o.clone()                 # salida en k; luego Euler con y_k
        e = y - self.p_o
        dv = self.obs_L * e + self.obs_m * self.z_o
        dz = e - self.obs_beta * self.z_o
        self.p_o = self.p_o + self.dt * self.v_o
        self.v_o = self.v_o + self.dt * dv
        self.z_o = self.z_o + self.dt * dz
        self.y = y

    def observe(self):
        v = self.vel
        if self.vel_noise > 0:
            v = v + self.vel_noise * torch.randn(v.shape, generator=self.gen)
        al = self.y[:, 1]
        return torch.stack([self.y[:, 0], torch.sin(al), torch.cos(al), v[:, 0], v[:, 1]], dim=1)

    # ------------------------------------------------------------------
    def recompensa(self, x, u):
        th, th_d, al_d = x[:, 0], x[:, 2], x[:, 3]
        aw = envolver(x[:, 1])
        subir = 0.5 * (1 - aw.abs() / math.pi)                       # 0 abajo .. 0.5 arriba
        cerca = (aw.abs() < 0.3).float()
        bono = cerca * torch.clamp(1 - 20 * aw ** 2 - 0.05 * al_d ** 2 - 0.05 * th_d ** 2
                                   - 0.002 * u ** 2, min=0.0)        # hasta +1 arriba y quieto
        return (subir + bono - 0.2 * th ** 2 - 0.0005 * u ** 2) / 1.5

    def step(self, a_idx: torch.Tensor):
        u = self.acciones[a_idx]
        for _ in range(self.substeps):
            self.x = rk4_step(self.x, u, self.bp, self.dt)
            self._update_sensor()
        self.t += 1

        terminated = (self.x[:, 0].abs() > self.theta_lim) | ~torch.isfinite(self.x).all(dim=1)
        truncated = (self.t >= self.max_steps) & ~terminated
        r = self.recompensa(self.x, u)
        r = torch.where(terminated, torch.full_like(r, self.r_terminal), r)
        r = torch.nan_to_num(r, nan=self.r_terminal)

        obs_final = torch.nan_to_num(self.observe())
        done = terminated | truncated
        obs = obs_final.clone()
        if done.any():
            self.x = torch.nan_to_num(self.x)
            obs[done] = self.reset(done)[done]
        return obs, r, terminated, truncated, obs_final
