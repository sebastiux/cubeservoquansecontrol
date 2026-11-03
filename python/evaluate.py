"""
Evaluación en Python del controlador neuronal vs LQR sobre el modelo NO lineal.
(La validación "oficial" se hace en MATLAB: matlab/main_simulacion_nn.m)

Uso:
    python evaluate.py [../models/nn_qube_policy.pt]
Genera ../models/figuras/python_*.png
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from policy import NNController
from qube_model import BatchParams, QubeParams, SensorModel
from sim import lqr_controller, rollout, sample_initial_states, success_mask

ROOT = Path(__file__).resolve().parent.parent
DEG = math.pi / 180


def load(path):
    ck = torch.load(path, weights_only=False)
    meta = ck["meta"]
    net = NNController(hidden=tuple(meta["hidden"]), vmax=meta["params"]["Vmax"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    p = QubeParams(**{k: meta["params"][k] for k in QubeParams.__dataclass_fields__})
    return net, p, np.array(meta["K_lqr"])


@torch.no_grad()
def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "nn_qube_policy.pt"
    net, p, K = load(path)
    lqr = lqr_controller(K, p.Vmax)
    figs = ROOT / "models" / "figuras"
    figs.mkdir(parents=True, exist_ok=True)

    # --- Monte Carlo: parámetros ±15 %, encoders + filtro + ruido
    g = torch.Generator().manual_seed(42)
    n = 2000
    x0 = sample_initial_states(n, (45 * DEG, 25 * DEG, 2.0, 4.0), g)
    bp = BatchParams(p, n, rel_spread=0.15, generator=g)
    for name, c in (("NN ", net), ("LQR", lqr)):
        torch.manual_seed(0)
        s = SensorModel(quantize=True, filtered_velocity=True, vel_noise_std=0.05)
        out = rollout(c, x0, bp, int(4 / p.Ts), p.Ts, sensor=s)
        ok = success_mask(out["X"])
        U = out["U"]
        print(f"Monte Carlo {name}: éxito = {100 * ok.float().mean():5.1f} %   "
              f"|u| medio = {U.abs().mean():.2f} V   "
              f"rms du = {torch.diff(U, dim=0).pow(2).mean().sqrt():.3f} V/paso")

    # --- Región de atracción alpha0 vs alpha_dot0
    a0 = torch.linspace(-40, 40, 81) * DEG
    ad0 = torch.linspace(-10, 10, 81)
    AA, AD = torch.meshgrid(a0, ad0, indexing="xy")
    x0 = torch.stack([torch.zeros_like(AA), AA, torch.zeros_like(AA), AD], -1).reshape(-1, 4)
    bp = BatchParams(p, x0.shape[0])
    roa = {}
    for name, c in (("LQR", lqr), ("Red neuronal", net)):
        out = rollout(c, x0, bp, int(4 / p.Ts), p.Ts, sensor=None)
        roa[name] = success_mask(out["X"]).reshape(AA.shape).float().numpy()
        print(f"Región de atracción {name}: {100 * roa[name].mean():.1f} % del área")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a, (name, r) in zip(ax, roa.items()):
        a.imshow(r, origin="lower", extent=[-40, 40, -10, 10], aspect="auto", cmap="gray")
        a.set_title(f"{name} ({100 * r.mean():.0f} %)")
        a.set_xlabel(r"$\alpha_0$ (deg)")
    ax[0].set_ylabel(r"$\dot\alpha_0$ (rad/s)")
    fig.tight_layout()
    fig.savefig(figs / "python_region_atraccion.png", dpi=120)

    # --- Respuesta temporal alpha0 = 20°
    x0 = torch.tensor([[0.0, 20 * DEG, 0.0, 0.0]])
    bp = BatchParams(p, 1)
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for name, c, st in (("LQR", lqr, "r--"), ("NN", net, "b-")):
        s = SensorModel(quantize=True, filtered_velocity=True)
        out = rollout(c, x0, bp, int(5 / p.Ts), p.Ts, sensor=s)
        t = np.arange(out["X"].shape[0]) * p.Ts
        ax[0].plot(t, out["X"][:, 0, 1].numpy() / DEG, st, label=name)
        ax[1].plot(t, out["X"][:, 0, 0].numpy() / DEG, st)
        ax[2].plot(t[:-1], out["U"][:, 0].numpy(), st)
    ax[0].set_ylabel(r"$\alpha$ (deg)"); ax[1].set_ylabel(r"$\theta$ (deg)")
    ax[2].set_ylabel("Vm (V)"); ax[2].set_xlabel("t (s)")
    ax[0].legend(); [a.grid(True) for a in ax]
    ax[0].set_title(r"Modelo no lineal, $\alpha_0 = 20°$, encoders + filtro")
    fig.tight_layout()
    fig.savefig(figs / "python_respuesta_alpha20.png", dpi=120)
    print(f"Figuras en {figs}")


if __name__ == "__main__":
    main()
