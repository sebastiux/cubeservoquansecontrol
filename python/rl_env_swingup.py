"""
Entorno RL (vectorizado) del Qube-Servo 3 NO LINEAL para SWING-UP + BALANCE:
el péndulo empieza colgando y la red debe levantarlo y mantenerlo arriba.

Versión 2 (robusta para el Qube real). Además del modelo no lineal y los
encoders cuantizados, cada episodio aleatoriza lo que el modelo ideal no tiene
y que en el hardware producía temblor y varios intentos de subida:
  * inercia del brazo entre 0.2 y 1.1 veces mr*r^2/3. En el Qube real el
    temblor (alpha +-3..8° a 13 Hz) solo se reproduce si el brazo tiene bastante
    menos inercia que la del modelo (masa concentrada cerca del eje),
  * ganancia voltaje->par del motor (x0.8 a x1.25) y parámetros ±15 %,
  * retardo de actuación de 0 a 4 muestras (0 a 8 ms),
  * fricción seca del brazo (0 a 3 mN·m) y amortiguamientos (x0.5 a x3),
  * error de cero del encoder del péndulo (±1°).

Observación (8 entradas), calculada igual que en dqn_swingup_block.m:
    [theta, sin(alpha), cos(alpha),
     theta_dot, alpha_dot       (observador lento, a = 100, poco ruido)
     theta_dot, alpha_dot       (observador rápido, a = 250, poco retraso)
     u_anterior]
Los observadores son los de 3er orden del proyecto, con alpha DESENROLLADO e
integrados con Euler a 2 ms. u_anterior permite a la red suavizar el voltaje.

Recompensa: subir + bono al mantenerse arriba y quieto, con castigo a los
cambios bruscos de voltaje. El episodio termina si el brazo pasa de theta_lim.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from qube_model import BatchParams, QubeParams, rk4_step

DEG = math.pi / 180

A_OBS = (100.0, 250.0)                      # polos de los observadores (lento, rápido)
# Normalización de la observación
X_SCALE_SU = np.array([1.0, 1.0, 1.0, 5.0, 10.0, 5.0, 10.0, 10.0])
# Para reflejar un estado (x -> -x, u -> -u); cos(alpha) no cambia
ESPEJO = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
N_OBS = len(X_SCALE_SU)
MAX_RETARDO = 4                             # muestras de 2 ms


def envolver(a):
    """Ángulo a (-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))


class QubeSwingUpEnv:
    def __init__(self, n_envs: int, acciones, Ts_agente: float = 0.006,
                 params: QubeParams | None = None, aleatorio: bool = True,
                 theta_lim_deg: float = 120.0, t_max: float = 6.0,
                 p_abajo: float = 0.7, seed: int = 0, r_terminal: float = -20.0):
        self.p = params or QubeParams()
        self.n = n_envs
        self.acciones = torch.as_tensor(np.asarray(acciones, dtype=np.float32))
        self.dt = self.p.Ts
        self.substeps = max(1, int(round(Ts_agente / self.dt)))
        self.Ts_agente = self.substeps * self.dt
        self.aleatorio = aleatorio
        self.theta_lim = theta_lim_deg * DEG
        self.max_steps = int(round(t_max / self.Ts_agente))
        self.p_abajo = p_abajo
        self.r_terminal = r_terminal
        self.gen = torch.Generator().manual_seed(seed)
        self.q_enc = 2 * math.pi / self.p.enc_counts
        a = torch.tensor(A_OBS)
        self.obs_beta = 3 * a                        # (2,)
        self.obs_L = 3 * a ** 2
        self.obs_m = a ** 3 - self.obs_L * self.obs_beta

        n = n_envs
        self.x = torch.zeros(n, 4)                   # alpha SIN envolver, 0 = arriba
        self.y = torch.zeros(n, 2)                   # ángulos medidos
        self.vel = torch.zeros(n, 2, 2)              # [ángulo, observador]
        self.p_o = torch.zeros(n, 2, 2)
        self.v_o = torch.zeros(n, 2, 2)
        self.z_o = torch.zeros(n, 2, 2)
        self.u_prev = torch.zeros(n)                 # última acción ordenada
        self.u_fifo = torch.zeros(n, MAX_RETARDO + 1)
        self.retardo = torch.zeros(n, dtype=torch.long)
        self.k_u = torch.ones(n)                     # ganancia de voltaje
        self.offset = torch.zeros(n)                 # error de cero de alpha
        self.t = torch.zeros(n, dtype=torch.long)
        self.bp = BatchParams(self.p, n, 0.0)
        self.bp.tau_c = torch.zeros(n)
        self.reset(torch.ones(n, dtype=torch.bool))

    # ------------------------------------------------------------------
    def _u(self, k, lo, hi):
        return lo + (hi - lo) * torch.rand(k, generator=self.gen)

    def estados_iniciales(self, k):
        """p_abajo: colgando casi quieto. El resto: cualquier ángulo y velocidad."""
        x0 = torch.zeros(k, 4)
        abajo = torch.rand(k, generator=self.gen) < self.p_abajo
        signo = torch.where(torch.rand(k, generator=self.gen) < 0.5, -1.0, 1.0)
        x0[:, 0] = torch.where(abajo, self._u(k, -0.3, 0.3), self._u(k, -1.2, 1.2))
        x0[:, 1] = torch.where(abajo, signo * math.pi + self._u(k, -0.2, 0.2),
                               self._u(k, -math.pi, math.pi))
        x0[:, 2] = torch.where(abajo, self._u(k, -0.5, 0.5), self._u(k, -3.0, 3.0))
        x0[:, 3] = torch.where(abajo, self._u(k, -1.0, 1.0), self._u(k, -8.0, 8.0))
        return x0

    def _medir(self, x, offset):
        y = x[:, 0:2].clone()
        y[:, 1] = y[:, 1] + offset
        return torch.round(y / self.q_enc) * self.q_enc

    def reset(self, mask: torch.Tensor, x0: torch.Tensor | None = None):
        k = int(mask.sum())
        if k == 0:
            return self.observe()
        if x0 is None:
            x0 = self.estados_iniciales(k)
        self.x[mask] = x0
        self.t[mask] = 0
        # ---- aleatorización del "Qube" de este episodio
        nb = BatchParams(self.p, k, 0.15 if self.aleatorio else 0.0, generator=self.gen)
        if self.aleatorio:
            nb.mr = self.p.mr * self._u(k, 0.2, 1.1)        # inercia del brazo
            nb.Jr = nb.mr * nb.r ** 2 / 3
            nb.br = self.p.br * self._u(k, 0.5, 3.0)
            nb.bp = self.p.bp * self._u(k, 0.5, 3.0)
            self.bp.tau_c[mask] = self._u(k, 0.0, 3e-3)
            self.k_u[mask] = self._u(k, 0.8, 1.25)
            self.retardo[mask] = torch.randint(0, MAX_RETARDO + 1, (k,), generator=self.gen)
            self.offset[mask] = self._u(k, -1.0, 1.0) * DEG
        for name in ("Rm", "km", "mr", "r", "br", "mp", "Lp", "bp", "g", "l", "Jr", "Jp"):
            getattr(self.bp, name)[mask] = getattr(nb, name)
        y0 = self._medir(x0, self.offset[mask])
        self.y[mask] = y0
        v0 = x0[:, 2:4, None].expand(k, 2, 2)
        self.vel[mask] = v0
        self.p_o[mask] = y0[:, :, None].expand(k, 2, 2)      # observadores convergidos
        self.v_o[mask] = v0
        self.z_o[mask] = 0.0
        self.u_prev[mask] = 0.0
        self.u_fifo[mask] = 0.0
        return self.observe()

    def _update_sensor(self):
        y = self._medir(self.x, self.offset)
        self.vel = self.v_o.clone()                 # salida en k; luego Euler con y_k
        e = y[:, :, None] - self.p_o
        dv = self.obs_L * e + self.obs_m * self.z_o
        dz = e - self.obs_beta * self.z_o
        self.p_o = self.p_o + self.dt * self.v_o
        self.v_o = self.v_o + self.dt * dv
        self.z_o = self.z_o + self.dt * dz
        self.y = y

    def observe(self):
        al = self.y[:, 1]
        return torch.stack([self.y[:, 0], torch.sin(al), torch.cos(al),
                            self.vel[:, 0, 0], self.vel[:, 1, 0],
                            self.vel[:, 0, 1], self.vel[:, 1, 1], self.u_prev], dim=1)

    # ------------------------------------------------------------------
    def recompensa(self, x, u, du):
        th, th_d, al_d = x[:, 0], x[:, 2], x[:, 3]
        aw = envolver(x[:, 1])
        subir = 0.5 * (1 - aw.abs() / math.pi)                       # 0 abajo .. 0.5 arriba
        cerca = (aw.abs() < 0.3).float()
        bono = cerca * torch.clamp(1 - 20 * aw ** 2 - 0.05 * al_d ** 2 - 0.05 * th_d ** 2
                                   - 0.01 * u ** 2 - 0.03 * du ** 2, min=0.0)
        suave = torch.clamp(0.002 * du ** 2, max=0.1)
        return (subir + bono - 0.2 * th ** 2 - 0.0005 * u ** 2 - suave) / 1.5

    def step(self, a_idx: torch.Tensor):
        u = self.acciones[a_idx]
        du = u - self.u_prev
        filas = torch.arange(self.n)
        for _ in range(self.substeps):
            self.u_fifo = torch.roll(self.u_fifo, 1, dims=1)
            self.u_fifo[:, 0] = u
            u_real = self.k_u * self.u_fifo[filas, self.retardo]
            self.x = rk4_step(self.x, u_real, self.bp, self.dt)
            self._update_sensor()
        self.u_prev = u.clone()
        self.t += 1

        terminated = (self.x[:, 0].abs() > self.theta_lim) | ~torch.isfinite(self.x).all(dim=1)
        truncated = (self.t >= self.max_steps) & ~terminated
        r = self.recompensa(self.x, u, du)
        r = torch.where(terminated, torch.full_like(r, self.r_terminal), r)
        r = torch.nan_to_num(r, nan=self.r_terminal)

        obs_final = torch.nan_to_num(self.observe())
        done = terminated | truncated
        obs = obs_final.clone()
        if done.any():
            self.x = torch.nan_to_num(self.x)
            obs[done] = self.reset(done)[done]
        return obs, r, terminated, truncated, obs_final
