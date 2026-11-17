"""
Deep Q-Learning para SWING-UP + BALANCE del péndulo del Qube-Servo 3 sobre el
MODELO NO LINEAL (rl_env_swingup.QubeSwingUpEnv): la red levanta el péndulo
desde abajo y lo mantiene arriba.

Misma variante estable de DQN que train_dqn.py (Double DQN, n-step, red
objetivo con copia dura, targets acotados, Huber) más:
  * simetría del péndulo como aumento de datos (estado y acción reflejados),
  * red 5 -> 256 -> 256 -> 11 (entradas: theta, sin a, cos a, theta_dot, alpha_dot).

Uso:
    python train_dqn_swingup.py
Salida:
    ../models/dqn_swingup_qube.mat   contiene el struct `red_swingup` listo para
                                     el bloque de Simulink (solo hace falta MATLAB)
    ../models/dqn_swingup_qube.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_env_swingup import DEG, ESPEJO, X_SCALE_SU, QubeSwingUpEnv, envolver
from train_dqn import NStep

ROOT = Path(__file__).resolve().parent.parent
ACCIONES = [-10.0, -6.0, -3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 6.0, 10.0]


class QNetSU(nn.Module):
    def __init__(self, n_acciones, hidden=(256, 256)):
        super().__init__()
        self.register_buffer("x_scale", torch.tensor(X_SCALE_SU, dtype=torch.float32))
        layers, n_in = [], 5
        for h in hidden:
            layers += [nn.Linear(n_in, h), nn.ReLU()]
            n_in = h
        layers += [nn.Linear(n_in, n_acciones)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x / self.x_scale)


class Replay:
    def __init__(self, cap, dim=5):
        self.cap, self.i, self.n = cap, 0, 0
        self.o = torch.zeros(cap, dim)
        self.a = torch.zeros(cap, dtype=torch.long)
        self.g = torch.zeros(cap)
        self.o2 = torch.zeros(cap, dim)
        self.d = torch.zeros(cap)
        self.disc = torch.zeros(cap)

    def add(self, o, a, g, o2, d, disc):
        k = o.shape[0]
        idx = (torch.arange(k) + self.i) % self.cap
        self.o[idx], self.a[idx], self.g[idx] = o, a, g
        self.o2[idx], self.d[idx], self.disc[idx] = o2, d.float(), disc
        self.i = (self.i + k) % self.cap
        self.n = min(self.cap, self.n + k)

    def sample(self, b, gen):
        idx = torch.randint(0, self.n, (b,), generator=gen)
        return self.o[idx], self.a[idx], self.g[idx], self.o2[idx], self.d[idx], self.disc[idx]


# --------------------------------------------------------------------------
@torch.no_grad()
def evaluar(policy_fn, acciones, Ts_agente, n=256, t_max=8.0, seed=123, theta_lim_deg=120.0,
            rel_spread=0.15, vel_noise=0.05):
    """Desde colgando: éxito = el brazo nunca pasa del límite y en los últimos
    2 s |alpha| < 5° y |theta| < 45°. Devuelve (éxito, tiempo medio de subida)."""
    env = QubeSwingUpEnv(n, acciones, Ts_agente=Ts_agente, rel_spread=rel_spread,
                         vel_noise=vel_noise, t_max=t_max + 5, p_abajo=1.0, seed=seed,
                         theta_lim_deg=theta_lim_deg)
    obs = env.observe()
    steps = int(round(t_max / env.Ts_agente))
    A = torch.zeros(steps, n)
    TH = torch.zeros(steps, n)
    vivo = torch.ones(n, dtype=torch.bool)
    for k in range(steps):
        obs, _, term, _, _ = env.step(policy_fn(obs))
        vivo &= ~term
        A[k] = envolver(env.x[:, 1]).abs()
        TH[k] = env.x[:, 0].abs()
    cola = int(round(2.0 / env.Ts_agente))
    ok = vivo & (A[-cola:].max(0).values < 5 * DEG) & (TH[-cola:].max(0).values < 45 * DEG)
    # tiempo de subida: último instante con |alpha| > 10°
    fuera = (A > 10 * DEG).float()
    idx = torch.arange(1, steps + 1, dtype=torch.float32)[:, None]
    t_sub = (fuera * idx).max(0).values * env.Ts_agente
    t_medio = t_sub[ok].mean().item() if ok.any() else float("nan")
    return ok.float().mean().item(), t_medio, TH.max().item() / DEG


@torch.no_grad()
def trayectoria_prueba(q, acciones, Ts_agente, segundos=6.0):
    """Trayectoria determinista (sin ruido, parámetros nominales) registrada a
    2 ms: lo que ve el bloque de Simulink y el voltaje que debe dar."""
    env = QubeSwingUpEnv(1, acciones, Ts_agente=Ts_agente, rel_spread=0.0, vel_noise=0.0,
                         t_max=segundos + 5, p_abajo=1.0)
    x0 = torch.tensor([[0.0, -math.pi, 0.0, 0.0]])
    obs = env.reset(torch.ones(1, dtype=torch.bool), x0)
    X, U, XR = [], [], []
    from qube_model import rk4_step
    for _ in range(int(round(segundos / env.Ts_agente))):
        a = q(obs).argmax(1)
        u = env.acciones[a]
        for _ in range(env.substeps):
            aw = torch.remainder(env.y[0, 1] + math.pi, 2 * math.pi) - math.pi
            X.append([env.y[0, 0].item(), aw.item(), env.vel[0, 0].item(), 0.0])
            U.append(u.item())
            XR.append(env.x[0].tolist())
            env.x = rk4_step(env.x, u, env.bp, env.dt)
            env._update_sensor()
        obs = env.observe()
    return np.array(X), np.array(U).reshape(-1, 1), np.array(XR)


def exportar(q, acciones, env, path, theta_max_deg, extra):
    lin = [m for m in q.net if isinstance(m, nn.Linear)]
    red = {}
    for i, m in enumerate(lin, start=1):
        red[f"W{i}"] = m.weight.detach().double().numpy()
        red[f"b{i}"] = m.bias.detach().double().numpy().reshape(-1, 1)
    red["x_scale"] = q.x_scale.double().numpy().reshape(-1, 1)
    red["acciones"] = np.asarray(acciones, dtype=float).reshape(-1, 1)
    red["Ts"] = float(env.dt)
    red["n_hold"] = float(env.substeps)
    red["L"], red["beta"], red["m"] = float(env.obs_L), float(env.obs_beta), float(env.obs_m)
    red["theta_max"] = float(theta_max_deg * DEG)
    Xt, Ut, XRt = trayectoria_prueba(q, acciones, env.Ts_agente)
    d = {"red_swingup": red, "x_test": Xt, "u_test": Ut, "x_real_test": XRt}
    bloque = ROOT / "matlab" / "dqn_swingup_block.m"
    if bloque.exists():
        d["codigo_bloque"] = bloque.read_text(encoding="utf-8")
    d.update(extra)
    sio.savemat(str(path), d, oned_as="column")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--pasos", type=int, default=150_000)
    ap.add_argument("--updates", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--nstep", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--target-cada", type=int, default=1000)
    ap.add_argument("--buffer", type=int, default=1_000_000)
    ap.add_argument("--warmup", type=int, default=20_000)
    ap.add_argument("--eps-pasos", type=int, default=30_000)
    ap.add_argument("--eps-final", type=float, default=0.05)
    ap.add_argument("--ts-agente", type=float, default=0.01)
    ap.add_argument("--theta-lim", type=float, default=120.0)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--eval-cada", type=int, default=5_000)
    ap.add_argument("--out", default=str(ROOT / "models" / "dqn_swingup_qube"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    acciones = ACCIONES
    nA = len(acciones)
    env = QubeSwingUpEnv(args.n_envs, acciones, Ts_agente=args.ts_agente, seed=args.seed,
                         theta_lim_deg=args.theta_lim)
    q = QNetSU(nA, tuple(args.hidden))
    q_obj = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.pasos, eta_min=args.lr / 20)
    buf = Replay(args.buffer)
    nstep = NStep(args.nstep, args.n_envs, args.gamma)
    espejo = torch.tensor(ESPEJO, dtype=torch.float32)
    q_max = 1.0 / (1 - args.gamma)
    q_min = env.r_terminal - 1.5 / (1 - args.gamma)

    obs = env.observe()
    ret_ep = torch.zeros(args.n_envs)
    retornos, mejor, mejor_estado = [], (-1.0, 0.0), copy.deepcopy(q.state_dict())
    t0, n_grad = time.time(), 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for paso in range(1, args.pasos + 1):
        eps = max(args.eps_final, 1.0 - (1.0 - args.eps_final) * paso / args.eps_pasos)
        with torch.no_grad():
            a = q(obs).argmax(1)
        rnd = torch.rand(args.n_envs, generator=gen) < eps
        a[rnd] = torch.randint(0, nA, (int(rnd.sum()),), generator=gen)

        obs2, r, term, trunc, obs_fin = env.step(a)
        done = term | trunc
        tr = nstep.push(obs, a, r, term, done, obs_fin)
        if tr is not None:
            buf.add(*tr)
        ret_ep += r
        if done.any():
            retornos += ret_ep[done].tolist()
            ret_ep[done] = 0
        obs = obs2

        if buf.n >= args.warmup:
            for _ in range(args.updates):
                o, aa, g, o2, dd, disc = buf.sample(args.batch, gen)
                # simetría: la mitad del lote se refleja (x -> -x, u -> -u)
                fl = torch.rand(args.batch, generator=gen) < 0.5
                o = torch.where(fl[:, None], o * espejo, o)
                o2 = torch.where(fl[:, None], o2 * espejo, o2)
                aa = torch.where(fl, nA - 1 - aa, aa)
                with torch.no_grad():
                    a2 = q(o2).argmax(1, keepdim=True)                      # Double DQN
                    y = g + disc * (1 - dd) * q_obj(o2).gather(1, a2).squeeze(1)
                    y = y.clamp(q_min, q_max)
                loss = F.smooth_l1_loss(q(o).gather(1, aa[:, None]).squeeze(1), y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
                n_grad += 1
                if n_grad % args.target_cada == 0:
                    q_obj.load_state_dict(q.state_dict())
            sched.step()

        if paso % args.eval_cada == 0 or paso == args.pasos:
            exito, t_sub, th_max = evaluar(lambda y: q(y).argmax(1), acciones, args.ts_agente,
                                           theta_lim_deg=args.theta_lim)
            clave = (exito, -t_sub if t_sub == t_sub else -99.0)
            if clave >= mejor:
                mejor, mejor_estado = clave, copy.deepcopy(q.state_dict())
                torch.save({"state_dict": mejor_estado, "args": vars(args), "exito": exito},
                           out.with_suffix(".pt"))
            rm = np.mean(retornos[-100:]) if retornos else float("nan")
            print(f"paso {paso:6d}/{args.pasos}  eps={eps:.2f}  retorno={rm:7.1f}  "
                  f"éxito={100 * exito:5.1f}%  t_subida={t_sub:4.2f}s  theta_max={th_max:5.0f}°  "
                  f"[{time.time() - t0:5.0f}s]", flush=True)

    q.load_state_dict(mejor_estado)
    exito, t_sub, th_max = evaluar(lambda y: q(y).argmax(1), acciones, args.ts_agente,
                                   n=1000, seed=999, theta_lim_deg=args.theta_lim)
    meta = {"acciones": acciones, "Ts_agente": env.Ts_agente, "exito_val": exito,
            "t_subida": t_sub, "theta_max_deg": th_max, "args": vars(args)}
    torch.save({"state_dict": q.state_dict(), "meta": meta}, out.with_suffix(".pt"))
    exportar(q, acciones, env, out.with_suffix(".mat"), theta_max_deg=150.0,
             extra={"exito_val": exito})
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nValidación final (1000 casos): éxito={100 * exito:.1f}%  t_subida={t_sub:.2f}s  "
          f"theta_max={th_max:.0f}°  ->  {out.with_suffix('.mat')}")


if __name__ == "__main__":
    main()
