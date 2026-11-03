"""
Deep Q-Learning (DQN) para balancear el péndulo del Qube-Servo 3 usando el
MODELO NO LINEAL (rl_env.QubeBalanceEnv).

Variante estable de DQN:
  * Double DQN (la red online elige la acción, la red objetivo la evalúa)
  * Retornos n-step (la recompensa se propaga n decisiones de golpe)
  * Red objetivo con copia dura cada `--target-cada` gradientes
  * Targets acotados al rango físico posible [r_min/(1-g), r_max/(1-g)]
  * Pérdida de Huber, recorte de gradiente, entornos en paralelo
  (Con actualización suave y muestreo fino los valores Q divergían.)

Red Q:  x (4) -> 128 ReLU -> 128 ReLU -> Q(x, a) para cada voltaje a.
Política en MATLAB/Simulink:  u = acciones( argmax_a Q(x, a) )

Uso:
    python train_dqn.py            # completo
    python train_dqn.py --quick    # prueba
Salida:
    ../models/dqn_qube.mat   (pesos + acciones + Ts_agente para Simulink)
    ../models/dqn_qube.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_env import DEG, X_SCALE, QubeBalanceEnv

ROOT = Path(__file__).resolve().parent.parent
ACCIONES_DQN = [-10.0, -6.0, -3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 6.0, 10.0]
R_TERMINAL = -10.0   # debe coincidir con rl_env.step


class QNet(nn.Module):
    def __init__(self, n_acciones, hidden=(128, 128)):
        super().__init__()
        self.register_buffer("x_scale", torch.tensor(X_SCALE, dtype=torch.float32))
        layers, n_in = [], 4
        for h in hidden:
            layers += [nn.Linear(n_in, h), nn.ReLU()]
            n_in = h
        layers += [nn.Linear(n_in, n_acciones)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x / self.x_scale)


class Replay:
    """Guarda transiciones n-step: (o, a, G, o_n, terminal, gamma^m)."""

    def __init__(self, cap):
        self.cap, self.i, self.n = cap, 0, 0
        self.o = torch.zeros(cap, 4)
        self.a = torch.zeros(cap, dtype=torch.long)
        self.g = torch.zeros(cap)
        self.o2 = torch.zeros(cap, 4)
        self.d = torch.zeros(cap)
        self.disc = torch.zeros(cap)

    def add(self, o, a, g, o2, d, disc):
        k = o.shape[0]
        idx = (torch.arange(k) + self.i) % self.cap
        self.o[idx], self.a[idx], self.g[idx] = o, a, g
        self.o2[idx], self.d[idx], self.disc[idx] = o2, d.float(), disc
        self.i = (self.i + k) % self.cap
        self.n = min(self.cap, self.n + k)

    def __len__(self):
        return self.n

    def sample(self, b, gen):
        idx = torch.randint(0, self.n, (b,), generator=gen)
        return self.o[idx], self.a[idx], self.g[idx], self.o2[idx], self.d[idx], self.disc[idx]


class NStep:
    """Ventana deslizante (n, N_envs) que arma transiciones n-step."""

    def __init__(self, n, n_envs, gamma):
        self.n, self.gamma = n, gamma
        self.o, self.a, self.r = [], [], []
        self.term, self.done, self.of = [], [], []

    def push(self, o, a, r, term, done, of):
        for lst, v in ((self.o, o), (self.a, a), (self.r, r), (self.term, term),
                       (self.done, done), (self.of, of)):
            lst.append(v)
        if len(self.o) < self.n:
            return None
        R = torch.stack(self.r)                 # (n, N)
        D = torch.stack(self.done)
        T = torch.stack(self.term)
        OF = torch.stack(self.of)               # (n, N, 4)
        # vivo_k = no hubo fin de episodio antes del paso k
        vivo = torch.ones_like(R)
        vivo[1:] = torch.cumprod(1 - D[:-1].float(), 0)
        pot = self.gamma ** torch.arange(self.n, dtype=R.dtype)[:, None]
        G = (pot * R * vivo).sum(0)
        m = vivo.sum(0).long()                  # nº de pasos usados (1..n)
        idx = m - 1
        cols = torch.arange(R.shape[1])
        out = (self.o[0], self.a[0], G, OF[idx, cols], T[idx, cols], self.gamma ** m.float())
        for lst in (self.o, self.a, self.r, self.term, self.done, self.of):
            lst.pop(0)
        return out


# --------------------------------------------------------------------------
@torch.no_grad()
def evaluar(policy_fn, acciones, n=256, t_max=5.0, seed=123, Ts_agente=0.01):
    """Éxito = sobrevive t_max sin caer y termina con |alpha|<3°, |theta|<15°."""
    env = QubeBalanceEnv(n, acciones, Ts_agente=Ts_agente, rel_spread=0.10,
                         t_max=t_max + 1, seed=seed)
    obs = env.observe()
    vivo = torch.ones(n, dtype=torch.bool)
    steps = int(round(t_max / env.Ts_agente))
    hist_a, hist_th = [], []
    for _ in range(steps):
        obs, _, term, _, _ = env.step(policy_fn(obs))
        vivo &= ~term
        hist_a.append(env.x[:, 1].abs())
        hist_th.append(env.x[:, 0].abs())
    cola = max(1, steps // 10)
    a_fin = torch.stack(hist_a[-cola:]).max(0).values
    th_fin = torch.stack(hist_th[-cola:]).max(0).values
    ok = vivo & (a_fin < 3 * DEG) & (th_fin < 15 * DEG)
    return ok.float().mean().item(), vivo.float().mean().item()


def exportar(qnet, acciones, Ts_agente, path, extra):
    lin = [m for m in qnet.net if isinstance(m, nn.Linear)]
    d = {}
    for i, m in enumerate(lin, start=1):
        d[f"W{i}"] = m.weight.detach().double().numpy()
        d[f"b{i}"] = m.bias.detach().double().numpy().reshape(-1, 1)
    d["x_scale"] = qnet.x_scale.double().numpy().reshape(-1, 1)
    d["acciones"] = np.asarray(acciones, dtype=float).reshape(-1, 1)
    d["Ts_agente"] = float(Ts_agente)
    d["Vmax"] = float(max(abs(a) for a in acciones))
    # vectores de prueba para verificar el forward en MATLAB
    g = torch.Generator().manual_seed(7)
    xt = (torch.rand(200, 4, generator=g) * 2 - 1) * torch.tensor([1.0, 0.4, 5.0, 10.0])
    with torch.no_grad():
        q = qnet(xt)
    d["x_test"] = xt.double().numpy()
    d["q_test"] = q.double().numpy()
    d["u_test"] = d["acciones"][q.argmax(1).numpy(), 0].reshape(-1, 1)
    d.update(extra)
    sio.savemat(str(path), d, oned_as="column")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--pasos", type=int, default=60_000, help="pasos vectorizados")
    ap.add_argument("--updates", type=int, default=2, help="gradientes por paso")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--nstep", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--target-cada", type=int, default=500)
    ap.add_argument("--buffer", type=int, default=400_000)
    ap.add_argument("--warmup", type=int, default=5_000)
    ap.add_argument("--eps-pasos", type=int, default=8_000)
    ap.add_argument("--eps-final", type=float, default=0.05)
    ap.add_argument("--ts-agente", type=float, default=0.01)
    ap.add_argument("--peso-theta", type=float, default=1.0, help="peso de theta^2 en el costo")
    ap.add_argument("--eval-cada", type=int, default=2_000)
    ap.add_argument("--out", default=str(ROOT / "models" / "dqn_qube"))
    args = ap.parse_args()
    if args.quick:
        args.pasos, args.eval_cada, args.eps_pasos = 4_000, 1_000, 2_000

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    acciones = ACCIONES_DQN
    env = QubeBalanceEnv(args.n_envs, acciones, Ts_agente=args.ts_agente, seed=args.seed,
                         pesos_costo=(args.peso_theta, 20.0, 0.05, 0.05, 0.002))
    q = QNet(len(acciones))
    q_obj = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.pasos, eta_min=args.lr / 20)
    buf = Replay(args.buffer)
    nstep = NStep(args.nstep, args.n_envs, args.gamma)
    q_max = 1.0 / (1 - args.gamma)
    q_min = R_TERMINAL - q_max

    obs = env.observe()
    ret_ep = torch.zeros(args.n_envs)
    retornos, mejor, mejor_estado = [], -1.0, copy.deepcopy(q.state_dict())
    t0, n_grad = time.time(), 0

    for paso in range(1, args.pasos + 1):
        eps = max(args.eps_final, 1.0 - (1.0 - args.eps_final) * paso / args.eps_pasos)
        with torch.no_grad():
            a = q(obs).argmax(1)
        rnd = torch.rand(args.n_envs, generator=gen) < eps
        a[rnd] = torch.randint(0, len(acciones), (int(rnd.sum()),), generator=gen)

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

        if len(buf) >= args.warmup:
            for _ in range(args.updates):
                o, aa, g, o2, dd, disc = buf.sample(args.batch, gen)
                with torch.no_grad():
                    a2 = q(o2).argmax(1, keepdim=True)                       # Double DQN
                    y = g + disc * (1 - dd) * q_obj(o2).gather(1, a2).squeeze(1)
                    y = y.clamp(q_min, q_max)
                qa = q(o).gather(1, aa[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(qa, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
                n_grad += 1
                if n_grad % args.target_cada == 0:
                    q_obj.load_state_dict(q.state_dict())

        sched.step()
        if paso % args.eval_cada == 0 or paso == args.pasos:
            exito, sobrevive = evaluar(lambda y: q(y).argmax(1), acciones,
                                       Ts_agente=args.ts_agente)
            if exito >= mejor:
                mejor, mejor_estado = exito, copy.deepcopy(q.state_dict())
            rm = np.mean(retornos[-100:]) if retornos else float("nan")
            print(f"paso {paso:6d}/{args.pasos}  eps={eps:.2f}  retorno(100 ep)={rm:7.1f}  "
                  f"éxito={100 * exito:5.1f}%  sobrevive={100 * sobrevive:5.1f}%  "
                  f"[{time.time() - t0:5.0f}s]", flush=True)

    q.load_state_dict(mejor_estado)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"acciones": acciones, "Ts_agente": env.Ts_agente, "exito_val": mejor,
            "args": vars(args)}
    torch.save({"state_dict": q.state_dict(), "meta": meta}, out.with_suffix(".pt"))
    exportar(q, acciones, env.Ts_agente, out.with_suffix(".mat"),
             {"tipo": "dqn", "exito_val": mejor})
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nMejor éxito en validación: {100 * mejor:.1f}%   ->  {out.with_suffix('.mat')}")


if __name__ == "__main__":
    main()
