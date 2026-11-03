"""
Q-LEARNING TABULAR para balancear el péndulo del Qube-Servo 3 usando el
MODELO NO LINEAL (rl_env.QubeBalanceEnv).

El estado continuo [theta, alpha, theta_dot, alpha_dot] se discretiza en
celdas (más finas cerca del equilibrio) y se aprende una tabla Q(s, a):

    Q(s,a) <- Q(s,a) + lr * ( r + gamma * max_a' Q(s',a') - Q(s,a) )

Se simulan cientos de entornos en paralelo para llenar la tabla rápido y se
aprovecha la SIMETRÍA del péndulo: la transición (x, u, r, x') es tan válida
como (-x, -u, r, -x'), así que cada paso actualiza ambas celdas.

Índice de celda (0-based), idéntico en MATLAB:
    i_k = número de bordes_k menores que x_k          (k = 1..4)
    s   = i1 + n1*(i2 + n2*(i3 + n3*i4))               -> fila s+1 en MATLAB

Uso:
    python train_qlearning.py            (~5-10 min CPU)
    python train_qlearning.py --quick
Salida:
    ../models/qlearning_qube.mat   (tabla Q + bordes + acciones)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from rl_env import DEG, QubeBalanceEnv
from train_dqn import evaluar

ROOT = Path(__file__).resolve().parent.parent

ACCIONES_Q = [-10.0, -5.0, -2.0, -0.7, 0.0, 0.7, 2.0, 5.0, 10.0]

# Bordes INTERIORES de cada dimensión (más finos cerca de 0)
BORDES = [
    np.array([-0.7, -0.3, -0.1, 0.1, 0.3, 0.7]),                                      # theta (rad)
    np.array([-12, -8, -5, -3, -1.5, -0.5, 0.5, 1.5, 3, 5, 8, 12]) * DEG,             # alpha (rad)
    np.array([-2.5, -1, -0.3, 0.3, 1, 2.5]),                                          # theta_dot
    np.array([-7, -4, -2, -1, -0.3, 0.3, 1, 2, 4, 7]),                                # alpha_dot
]
N_BINS = [len(b) + 1 for b in BORDES]
N_ESTADOS = int(np.prod(N_BINS))


def estado_discreto(obs: torch.Tensor) -> torch.Tensor:
    """obs (N,4) -> índice de celda (N,) en [0, N_ESTADOS)."""
    s = torch.zeros(obs.shape[0], dtype=torch.long)
    mult = 1
    for k in range(4):
        b = torch.as_tensor(BORDES[k], dtype=obs.dtype)
        i = (obs[:, k:k + 1] > b).sum(1)
        s += i * mult
        mult *= N_BINS[k]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=512)
    ap.add_argument("--pasos", type=int, default=80_000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr-ini", type=float, default=0.2)
    ap.add_argument("--lr-fin", type=float, default=0.01)
    ap.add_argument("--eps-ini", type=float, default=0.3)
    ap.add_argument("--eps-fin", type=float, default=0.01)
    ap.add_argument("--ts-agente", type=float, default=0.01)
    ap.add_argument("--eval-cada", type=int, default=4_000)
    ap.add_argument("--out", default=str(ROOT / "models" / "qlearning_qube"))
    args = ap.parse_args()
    if args.quick:
        args.pasos, args.eval_cada = 3_000, 1_000

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    acciones = ACCIONES_Q
    nA = len(acciones)
    env = QubeBalanceEnv(args.n_envs, acciones, Ts_agente=args.ts_agente, seed=args.seed)

    Q = torch.zeros(N_ESTADOS, nA, dtype=torch.float64)
    # Desempate: en celdas nunca visitadas argmax elige 0 V (no -10 V)
    Q[:, acciones.index(0.0)] = 1e-6
    visitas = torch.zeros(N_ESTADOS, nA, dtype=torch.float64)
    print(f"Tabla Q: {N_ESTADOS} estados x {nA} acciones = {Q.numel()} valores")

    obs = env.observe()
    s = estado_discreto(obs)
    mejor, mejor_Q, t0 = -1.0, Q.clone(), time.time()

    for paso in range(1, args.pasos + 1):
        frac = paso / args.pasos
        eps = args.eps_ini + (args.eps_fin - args.eps_ini) * min(1.0, frac / 0.7)
        lr = args.lr_ini * (args.lr_fin / args.lr_ini) ** frac

        a = Q[s].argmax(1)
        rnd = torch.rand(args.n_envs, generator=gen) < eps
        a[rnd] = torch.randint(0, nA, (int(rnd.sum()),), generator=gen)

        obs2, r, term, trunc, obs_fin = env.step(a)
        s_fin = estado_discreto(obs_fin)
        # transición real + su espejo (-x, -u): acciones simétricas -> índice nA-1-a
        s_all = torch.cat([s, estado_discreto(-obs)])
        a_all = torch.cat([a, nA - 1 - a])
        sf_all = torch.cat([s_fin, estado_discreto(-obs_fin)])
        r_all = torch.cat([r, r]).double()
        vivo = torch.cat([~term, ~term]).double()
        objetivo = r_all + args.gamma * vivo * Q[sf_all].max(1).values
        # actualización TD (promedia si varios entornos caen en la misma celda)
        flat = s_all * nA + a_all
        delta = torch.zeros(N_ESTADOS * nA, dtype=torch.float64)
        cuenta = torch.zeros(N_ESTADOS * nA, dtype=torch.float64)
        delta.index_add_(0, flat, objetivo - Q.view(-1)[flat])
        cuenta.index_add_(0, flat, torch.ones_like(objetivo))
        m = cuenta > 0
        Q.view(-1)[m] += lr * delta[m] / cuenta[m]
        visitas.view(-1).add_(cuenta)

        obs = obs2
        s = estado_discreto(obs)

        if paso % args.eval_cada == 0 or paso == args.pasos:
            Qe = Q.clone()
            exito, sobrevive = evaluar(lambda y: Qe[estado_discreto(y)].argmax(1), acciones,
                                       Ts_agente=args.ts_agente)
            if exito >= mejor:
                mejor, mejor_Q = exito, Qe
            cob = 100 * (visitas.sum(1) > 0).double().mean().item()
            print(f"paso {paso:6d}/{args.pasos}  eps={eps:.2f}  lr={lr:.3f}  "
                  f"éxito={100 * exito:5.1f}%  sobrevive={100 * sobrevive:5.1f}%  "
                  f"celdas visitadas={cob:4.1f}%  [{time.time() - t0:4.0f}s]", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    d = {
        "tipo": "qlearning",
        "Q": mejor_Q.numpy(),                        # (N_ESTADOS, nA)
        "acciones": np.asarray(acciones, float).reshape(-1, 1),
        "bordes_theta": BORDES[0].reshape(-1, 1), "bordes_alpha": BORDES[1].reshape(-1, 1),
        "bordes_theta_d": BORDES[2].reshape(-1, 1), "bordes_alpha_d": BORDES[3].reshape(-1, 1),
        "n_bins": np.asarray(N_BINS, float).reshape(-1, 1),
        "Ts_agente": float(env.Ts_agente), "Vmax": float(max(map(abs, acciones))),
        "exito_val": mejor,
    }
    g = torch.Generator().manual_seed(7)
    xt = (torch.rand(200, 4, generator=g) * 2 - 1) * torch.tensor([1.0, 0.3, 4.0, 8.0], dtype=torch.float64)
    d["x_test"] = xt.numpy()
    d["s_test"] = estado_discreto(xt).numpy().reshape(-1, 1).astype(float) + 1   # 1-based
    d["u_test"] = d["acciones"][mejor_Q[estado_discreto(xt)].argmax(1).numpy(), 0].reshape(-1, 1)
    sio.savemat(str(out.with_suffix(".mat")), d, oned_as="column")
    out.with_suffix(".json").write_text(json.dumps(
        {"acciones": acciones, "n_bins": N_BINS, "exito_val": mejor, "args": vars(args)},
        indent=2, ensure_ascii=False))
    print(f"\nMejor éxito en validación: {100 * mejor:.1f}%   ->  {out.with_suffix('.mat')}")


if __name__ == "__main__":
    main()
