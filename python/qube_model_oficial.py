"""
Modelo NO LINEAL OFICIAL de Quanser para el péndulo rotatorio del Qube-Servo 3.

Fuente (repositorio quanser/Quanser_Academic_Resources, rama dev-windows):
  * 4_concept_reviews/Modeling/007_pendulum_state_space_model.pdf, ecs. 1.1-1.3
  * 6_teaching/1_Controls/Qube_Servo_3/supplementary_material/Pendulum Equations.pdf
    (K. J. Astrom), ec. 8
  * .../sp6_pendulum_control/1b_lqr_control/hardware/matlab/qube3_rotpen_param.m

Ecuaciones oficiales (péndulo colgando, alpha_d = 0 abajo):

  (Jr + Jp sin^2 a) th'' + mp l r cos a a'' + 2 Jp sin a cos a th' a'
                                    - mp l r sin a a'^2 = tau - br th'       (1.1)
  Jp a'' + mp l r cos a th'' - Jp sin a cos a th'^2 + mp g l sin a = -bp a'  (1.2)
  tau = km (vm - km th') / Rm                                                (1.3)
  Jr = mr r^2 / 3,  Jp = mp Lp^2 / 3,  l = Lp / 2

El hardware mide el ángulo invertido alpha = alpha_d - pi (0 = arriba). Con
sin(alpha_d) = -sin(alpha) y cos(alpha_d) = -cos(alpha) las mismas ecuaciones son:

  (Jr + Jp sin^2 a) th'' - mp l r cos a a'' + 2 Jp sin a cos a th' a'
                                    + mp l r sin a a'^2 = tau - br th'
  Jp a'' - mp l r cos a th'' - Jp sin a cos a th'^2 - mp g l sin a = -bp a'

Este archivo NO usa ningún modelo linealizado. La implementación se comprueba
con la conservación de la energía (prueba_energia): sin voltaje ni fricción,
E = 1/2 (Jr + Jp sin^2 a) th'^2 - mp l r cos a th' a' + 1/2 Jp a'^2 + mp g l cos a
debe mantenerse constante.

Estado: x = [theta, alpha, theta_dot, alpha_dot] con alpha = 0 ARRIBA.
Todo vectorizado por lotes en PyTorch (un Qube distinto por fila).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ParametrosQube:
    """qube3_rotpen_param.m (valores oficiales)."""
    Rm: float = 7.5          # Resistencia (Ohm)
    kt: float = 0.0422       # Constante de par (N-m/A)
    km: float = 0.0422       # Constante contra-electromotriz (V-s/rad)
    mr: float = 0.095        # Masa del brazo (kg)
    r: float = 0.085         # Longitud del brazo (m)
    br: float = 1e-3         # Amortiguamiento del brazo (N-m-s/rad)
    mp: float = 0.024        # Masa del péndulo (kg)
    Lp: float = 0.129        # Longitud del péndulo (m)
    bp: float = 5e-5         # Amortiguamiento del péndulo (N-m-s/rad)
    g: float = 9.81
    # Interfaz
    Ts: float = 0.002        # paso del Simulink (500 Hz)
    enc_counts: int = 2048   # cuentas por vuelta de los encoders
    Vmax: float = 10.0       # voltaje máximo usado


FISICOS = ("Rm", "km", "mr", "r", "br", "mp", "Lp", "bp", "g")


class LoteParametros:
    """Parámetros físicos como tensores (N,), uno por Qube simulado."""

    def __init__(self, base: ParametrosQube, n: int, dtype=torch.float32):
        self.base = base
        for k in FISICOS:
            setattr(self, k, torch.full((n,), float(getattr(base, k)), dtype=dtype))
        self.tau_c = torch.zeros(n, dtype=dtype)     # fricción seca del brazo (N-m)

    @property
    def Jr(self):
        return self.mr * self.r ** 2 / 3

    @property
    def Jp(self):
        return self.mp * self.Lp ** 2 / 3

    @property
    def l(self):  # noqa: E743
        return self.Lp / 2


def dinamica(x: torch.Tensor, vm: torch.Tensor, p: LoteParametros) -> torch.Tensor:
    """dx/dt del modelo oficial (alpha = 0 arriba). x: (N,4), vm: (N,)."""
    th_d, al, al_d = x[:, 2], x[:, 1], x[:, 3]
    s, c = torch.sin(al), torch.cos(al)
    Jr, Jp, mlr = p.Jr, p.Jp, p.mp * p.l * p.r

    tau = p.km * (vm - p.km * th_d) / p.Rm                                   # (1.3)

    # [m11 m12; m12 m22] [th''; a''] = [c1; c2]
    m11 = Jr + Jp * s ** 2
    m12 = -mlr * c
    m22 = Jp
    c1 = tau - p.br * th_d - 2 * Jp * s * c * th_d * al_d - mlr * s * al_d ** 2
    c1 = c1 - p.tau_c * torch.tanh(th_d / 0.1)        # fricción seca (no está en el oficial)
    c2 = -p.bp * al_d + Jp * s * c * th_d ** 2 + p.mp * p.g * p.l * s

    det = m11 * m22 - m12 ** 2
    th_dd = (m22 * c1 - m12 * c2) / det
    al_dd = (-m12 * c1 + m11 * c2) / det
    return torch.stack([th_d, al_d, th_dd, al_dd], dim=1)


def paso_rk4(x, vm, p, dt):
    """Un paso RK4 con el voltaje constante durante dt."""
    k1 = dinamica(x, vm, p)
    k2 = dinamica(x + 0.5 * dt * k1, vm, p)
    k3 = dinamica(x + 0.5 * dt * k2, vm, p)
    k4 = dinamica(x + dt * k3, vm, p)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def energia(x, p):
    th_d, al, al_d = x[:, 2], x[:, 1], x[:, 3]
    s, c = torch.sin(al), torch.cos(al)
    return (0.5 * (p.Jr + p.Jp * s ** 2) * th_d ** 2 - p.mp * p.l * p.r * c * th_d * al_d
            + 0.5 * p.Jp * al_d ** 2 + p.mp * p.g * p.l * c)


def prueba_energia(segundos=2.0):
    """Sin motor ni fricción la energía debe conservarse. Devuelve la deriva relativa."""
    base = ParametrosQube(km=0.0, br=0.0, bp=0.0)
    p = LoteParametros(base, 4, dtype=torch.float64)
    x = torch.tensor([[0.3, 2.5, 1.0, -2.0], [0.0, 0.5, -3.0, 4.0],
                      [1.0, -3.0, 2.0, 6.0], [-0.5, 1.2, 0.5, -8.0]], dtype=torch.float64)
    e0 = energia(x, p)
    for _ in range(int(segundos / base.Ts)):
        x = paso_rk4(x, torch.zeros(4, dtype=torch.float64), p, base.Ts)
    escala = (p.mp * p.g * p.l)                      # energía potencial de referencia
    return ((energia(x, p) - e0).abs() / escala).max().item()


if __name__ == "__main__":
    print(f"Deriva de energía en 2 s (relativa a mp*g*l): {prueba_energia():.2e}")
