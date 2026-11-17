"""
Entorno RL (vectorizado) del Qube-Servo 3 para SWING-UP + BALANCE (versión 4).

Planta: MODELO NO LINEAL OFICIAL de Quanser (qube_model_oficial.py), integrado
con RK4 a 2 ms. No se usa ningún modelo linealizado.

Lazo completo, igual que en el Simulink del Qube:

  red DQN -> voltaje ordenado -> FILTRO PASA BAJAS de 2.º orden
          (w = 300 rad/s, zeta = 1, Euler a 2 ms como ode1) -> motor
  encoders (2048 cuentas/rev) -> observadores de velocidad -> red DQN

El filtro va DENTRO del entrenamiento: la red ve su estado y aprende a
compensar su retraso (~7 ms).

Cada episodio usa un Qube distinto (lo que el modelo ideal no garantiza):
  * inercia del brazo x0.7 a x1.9 de la oficial (mr*r^2/3), resto ±15 %
    (x1.76 equivale a sumar mp*r^2, como en el modelo anterior del proyecto),
  * ganancia voltaje->par x0.8 a x1.25, retardo extra de 0 a 7 muestras
    (0 a 14 ms: en el Qube real la v3, entrenada con 0 a 6 ms, subía y se caía),
  * fricción seca del brazo (0 a 3 mN·m), amortiguamientos x0.5 a x3,
  * error de cero del encoder del péndulo (±1°).

Observación (10 entradas), calculada igual que en dqn_swingup_block.m:
    [theta, sin(alpha), cos(alpha),
     theta_dot, alpha_dot      (observador lento, a = 100)
     theta_dot, alpha_dot      (observador rápido, a = 250)
     voltaje ordenado anterior, voltaje filtrado, derivada del filtrado]

Acciones: voltajes (modo absoluto) o incrementos de voltaje (modo incremental).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from qube_model_oficial import LoteParametros, ParametrosQube, paso_rk4

DEG = math.pi / 180

A_OBS = (100.0, 250.0)                      # polos de los observadores (lento, rápido)
FILTRO_W, FILTRO_ZETA = 300.0, 1.0          # pasa bajas óptimo (ver analisis_filtro.py)
X_SCALE_SU = np.array([1.0, 1.0, 1.0, 5.0, 10.0, 5.0, 10.0, 10.0, 10.0, 2000.0])
# Para reflejar un estado (x -> -x, u -> -u); cos(alpha) no cambia
ESPEJO = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
N_OBS = len(X_SCALE_SU)
MAX_RETARDO = 7                             # muestras de 2 ms (hasta 14 ms)


def envolver(a):
    """Ángulo a (-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))


class QubeSwingUpEnv:
    def __init__(self, n_envs: int, acciones, incremental: bool = False,
                 Ts_agente: float = 0.01, params: ParametrosQube | None = None,
                 aleatorio: bool = True, theta_lim_deg: float = 120.0, t_max: float = 6.0,
                 p_abajo: float = 0.7, seed: int = 0, r_terminal: float = -20.0,
                 filtro_w: float = FILTRO_W, filtro_zeta: float = FILTRO_ZETA):
        self.p = params or ParametrosQube()
        self.n = n_envs
        self.acciones = torch.as_tensor(np.asarray(acciones, dtype=np.float32))
        self.incremental = incremental
        self.dt = self.p.Ts
        self.substeps = max(1, int(round(Ts_agente / self.dt)))
        self.Ts_agente = self.substeps * self.dt
        self.aleatorio = aleatorio
        self.theta_lim = theta_lim_deg * DEG
        self.max_steps = int(round(t_max / self.Ts_agente))
        self.p_abajo = p_abajo
        self.r_terminal = r_terminal
        self.gen = torch.Generator().manual_seed(seed)
        self.filtro_w, self.filtro_zeta = filtro_w, filtro_zeta      # w <= 0: sin filtro
        self.registro = None        # lista para registrar (voltaje al motor, estado) cada 2 ms
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
        self.u_cmd = torch.zeros(n)                  # voltaje ordenado por la red
        self.f1 = torch.zeros(n)                     # filtro: voltaje filtrado
        self.f2 = torch.zeros(n)                     # filtro: su derivada
        self.u_fifo = torch.zeros(n, MAX_RETARDO + 1)
        self.retardo = torch.zeros(n, dtype=torch.long)
        self.k_u = torch.ones(n)                     # ganancia de voltaje
        self.zona_muerta = torch.zeros(n)            # voltaje mínimo que mueve el motor (V)
        self.filtro2_w = 0.0                         # 2.º filtro en serie (solo para pruebas)
        self.g1 = torch.zeros(n)
        self.g2 = torch.zeros(n)
        self.offset = torch.zeros(n)                 # error de cero de alpha
        self.t = torch.zeros(n, dtype=torch.long)
        self.bp = LoteParametros(self.p, n)
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
        # ---- el "Qube" de este episodio
        b = self.p
        if self.aleatorio:
            for nombre in ("Rm", "km", "r", "mp", "Lp"):
                getattr(self.bp, nombre)[mask] = getattr(b, nombre) * self._u(k, 0.85, 1.15)
            self.bp.mr[mask] = b.mr * self._u(k, 0.7, 1.9)          # inercia del brazo
            self.bp.br[mask] = b.br * self._u(k, 0.5, 3.0)
            self.bp.bp[mask] = b.bp * self._u(k, 0.5, 3.0)
            self.bp.tau_c[mask] = self._u(k, 0.0, 3e-3)
            self.k_u[mask] = self._u(k, 0.8, 1.25)
            self.retardo[mask] = torch.randint(0, MAX_RETARDO + 1, (k,), generator=self.gen)
            self.offset[mask] = self._u(k, -1.0, 1.0) * DEG
        y0 = self._medir(x0, self.offset[mask])
        self.y[mask] = y0
        v0 = x0[:, 2:4, None].expand(k, 2, 2)
        self.vel[mask] = v0
        self.p_o[mask] = y0[:, :, None].expand(k, 2, 2)      # observadores convergidos
        self.v_o[mask] = v0
        self.z_o[mask] = 0.0
        self.u_cmd[mask] = 0.0
        self.f1[mask] = 0.0
        self.f2[mask] = 0.0
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
                            self.vel[:, 0, 1], self.vel[:, 1, 1],
                            self.u_cmd, self.f1, self.f2], dim=1)

    # ------------------------------------------------------------------
    def recompensa(self, x, u, du):
        """u: voltaje ordenado, du: su cambio en esta decisión."""
        th, th_d, al_d = x[:, 0], x[:, 2], x[:, 3]
        aw = envolver(x[:, 1])
        subir = 0.5 * (1 - aw.abs() / math.pi)                       # 0 abajo .. 0.5 arriba
        cerca = (aw.abs() < 0.3).float()
        bono = cerca * torch.clamp(1 - 20 * aw ** 2 - 0.05 * al_d ** 2 - 0.05 * th_d ** 2
                                   - 0.02 * u ** 2 - 0.10 * du ** 2, min=0.0)
        suave = torch.clamp(0.002 * du ** 2, max=0.1)
        return (subir + bono - 0.2 * th ** 2 - 0.0005 * u ** 2 - suave) / 1.5

    def step(self, a_idx: torch.Tensor):
        vmax = self.p.Vmax
        if self.incremental:
            u = torch.clamp(self.u_cmd + self.acciones[a_idx], -vmax, vmax)
        else:
            u = self.acciones[a_idx]
        du = u - self.u_cmd
        filas = torch.arange(self.n)
        w, z = self.filtro_w, self.filtro_zeta
        for _ in range(self.substeps):
            if w <= 0:                              # sin filtro: sale lo que ordena la red
                self.f1, self.f2 = u.clone(), torch.zeros_like(u)
            # voltaje que sale del filtro en este paso (estado, como un integrador)
            self.u_fifo = torch.roll(self.u_fifo, 1, dims=1)
            self.u_fifo[:, 0] = self.f1
            u_real = self.k_u * self.u_fifo[filas, self.retardo]
            if self.filtro2_w > 0:                  # filtro externo adicional (Euler)
                w2 = self.filtro2_w
                salida2 = self.g1
                g1n = self.g1 + self.dt * self.g2
                self.g2 = self.g2 + self.dt * (w2 * w2 * (u_real - self.g1) - 2 * w2 * self.g2)
                self.g1 = g1n
                u_real = salida2
            # zona muerta: por debajo de ese voltaje el motor no entrega par
            u_real = torch.sign(u_real) * torch.clamp(u_real.abs() - self.zona_muerta, min=0.0)
            if self.registro is not None:
                self.registro.append((u_real.clone(), self.x.clone()))
            self.x = paso_rk4(self.x, u_real, self.bp, self.dt)
            if w > 0:
                # filtro de 2.º orden, Euler a 2 ms (igual que ode1 en Simulink)
                f1n = self.f1 + self.dt * self.f2
                f2n = self.f2 + self.dt * (w * w * (u - self.f1) - 2 * z * w * self.f2)
                self.f1, self.f2 = f1n, f2n
            self._update_sensor()
        self.u_cmd = u.clone()
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
