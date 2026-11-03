"""
Entorno de Aprendizaje por Refuerzo (vectorizado) del Qube-Servo 3 NO LINEAL
para la tarea de BALANCE del péndulo invertido.

    * Planta: qube_model.dynamics integrada con RK4 a Ts = 2 ms (como el Simulink).
    * El agente decide cada Ts_agente (por defecto 10 ms = 5 pasos de 2 ms) y la
      acción se mantiene (ZOH) entre decisiones.
    * Acciones DISCRETAS: voltajes del motor (Q-learning y DQN necesitan un
      conjunto finito de acciones).
    * Observación = lo que ve el controlador en el hardware:
      encoders cuantizados (2048 cuentas/rev) y velocidades estimadas con
      derivada filtrada 50s/(s+50) + ruido. Parámetros físicos aleatorizados.
    * Episodio termina si |alpha| > 25° o |theta| > 120°, o por tiempo (5 s).

N entornos corren en paralelo (tensores de PyTorch), con reinicio automático.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from qube_model import BatchParams, QubeParams, rk4_step

DEG = math.pi / 180

# Normalización de la observación [theta, alpha, theta_dot, alpha_dot]
X_SCALE = np.array([1.0, 0.3, 4.0, 8.0])


class QubeBalanceEnv:
    def __init__(self, n_envs: int, acciones, Ts_agente: float = 0.01,
                 params: QubeParams | None = None, rel_spread: float = 0.10,
                 sensor_real: bool = True, vel_noise: float = 0.05,
                 alpha_lim_deg: float = 25.0, theta_lim_deg: float = 120.0,
                 t_max: float = 5.0,
                 init_ranges=(30 * DEG, 15 * DEG, 1.0, 2.0), seed: int = 0,
                 pesos_costo=(1.0, 20.0, 0.05, 0.05, 0.002)):
        self.p = params or QubeParams()
        self.n = n_envs
        self.acciones = torch.as_tensor(np.asarray(acciones, dtype=np.float32))
        self.n_acciones = len(acciones)
        self.dt = self.p.Ts
        self.substeps = max(1, int(round(Ts_agente / self.dt)))
        self.Ts_agente = self.substeps * self.dt
        self.rel_spread = rel_spread
        self.sensor_real = sensor_real
        self.vel_noise = vel_noise
        self.alpha_lim = alpha_lim_deg * DEG
        self.theta_lim = theta_lim_deg * DEG
        self.max_steps = int(round(t_max / self.Ts_agente))
        self.init_ranges = torch.tensor(init_ranges, dtype=torch.float32)
        self.gen = torch.Generator().manual_seed(seed)
        self.q_enc = 2 * math.pi / self.p.enc_counts
        self.wc = 50.0
        self.w = pesos_costo

        self.x = torch.zeros(n_envs, 4)
        self.y_prev = torch.zeros(n_envs, 2)
        self.v_f = torch.zeros(n_envs, 2)
        self.t = torch.zeros(n_envs, dtype=torch.long)
        self.bp = BatchParams(self.p, n_envs, rel_spread, generator=self.gen)
        self.reset(torch.ones(n_envs, dtype=torch.bool))

    # ------------------------------------------------------------------
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen)

    def reset(self, mask: torch.Tensor, x0: torch.Tensor | None = None):
        """Reinicia los entornos indicados por `mask` (bool, (N,))."""
        k = int(mask.sum())
        if k == 0:
            return self.observe()
        if x0 is None:
            x0 = (self._rand(k, 4) * 2 - 1) * self.init_ranges
        self.x[mask] = x0
        self.t[mask] = 0
        # nuevos parámetros físicos para los entornos reiniciados
        nb = BatchParams(self.p, k, self.rel_spread, generator=self.gen)
        for name in ("Rm", "km", "mr", "r", "br", "mp", "Lp", "bp", "g", "l", "Jr", "Jp"):
            getattr(self.bp, name)[mask] = getattr(nb, name)
        self.y_prev[mask] = self._angles(x0)
        self.v_f[mask] = x0[:, 2:4]
        return self.observe()

    def _angles(self, x):
        y = x[:, 0:2]
        if self.sensor_real:
            y = torch.round(y / self.q_enc) * self.q_enc
        return y

    def _update_sensor(self):
        y = self._angles(self.x)
        if self.sensor_real:
            self.v_f = (self.v_f + self.wc * (y - self.y_prev)) / (1 + self.wc * self.dt)
        else:
            self.v_f = self.x[:, 2:4].clone()
        self.y_prev = y

    def observe(self):
        v = self.v_f
        if self.vel_noise > 0:
            v = v + self.vel_noise * torch.randn(v.shape, generator=self.gen)
        return torch.cat([self.y_prev, v], dim=1)

    # ------------------------------------------------------------------
    def costo(self, x, u):
        """Costo cuadrático tipo LQR: w_th th^2 + w_al al^2 + w_thd thd^2 + w_ald ald^2 + w_u u^2."""
        w = self.w
        return (w[0] * x[:, 0] ** 2 + w[1] * x[:, 1] ** 2 + w[2] * x[:, 2] ** 2
                + w[3] * x[:, 3] ** 2 + w[4] * u ** 2)

    def step(self, a_idx: torch.Tensor):
        """Aplica las acciones (índices) y avanza Ts_agente.

        Devuelve obs_siguiente, recompensa, terminado, truncado, obs_final
        (obs_final = observación antes del reinicio automático; es la que
        se guarda en el replay buffer)."""
        u = self.acciones[a_idx]
        for _ in range(self.substeps):
            self.x = rk4_step(self.x, u, self.bp, self.dt)
            self._update_sensor()
        self.t += 1

        terminated = (self.x[:, 1].abs() > self.alpha_lim) | (self.x[:, 0].abs() > self.theta_lim) \
            | ~torch.isfinite(self.x).all(dim=1)
        truncated = (self.t >= self.max_steps) & ~terminated

        r = 1.0 - torch.clamp(self.costo(self.x, u), 0.0, 2.0)
        r = torch.where(terminated, torch.full_like(r, -10.0), r)

        obs_final = self.observe()
        done = terminated | truncated
        obs = obs_final.clone()
        if done.any():
            obs[done] = self.reset(done)[done]
        return obs, r, terminated, truncated, obs_final
