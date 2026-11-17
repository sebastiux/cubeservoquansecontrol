"""
Deep Q-Learning para SWING-UP + BALANCE del péndulo del Qube-Servo 3 (versión 4).

Planta: modelo NO LINEAL OFICIAL de Quanser (qube_model_oficial.py) con el
filtro pasa bajas de la salida dentro del lazo (rl_env_swingup.py). No se usa
ningún modelo linealizado.

DQN estable: Double DQN, retornos n-step, red objetivo con copia dura, targets
acotados, Huber, y la simetría del péndulo como aumento de datos.
Red: 10 entradas -> 256 -> 256 -> una salida por acción.

Dos modos de acción:
  absoluto     la red elige el voltaje (15 niveles)
  incremental  la red elige cuánto sube o baja el voltaje (--incremental)

Uso:
    python train_dqn_swingup.py [--incremental]
Salida:
    ../models/dqn_swingup_v4_qube.mat   struct `red_swingup` listo para el bloque
                                        de Simulink (solo hace falta MATLAB)
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

from qube_model_oficial import paso_rk4, prueba_energia
from rl_env_swingup import (A_OBS, DEG, ESPEJO, FILTRO_W, FILTRO_ZETA, N_OBS, X_SCALE_SU,
                            QubeSwingUpEnv, envolver)

ROOT = Path(__file__).resolve().parent.parent
NOMBRE = "dqn_swingup_v4_qube"
ACCIONES_ABS = [-10.0, -6.0, -3.5, -2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0, 3.5, 6.0, 10.0]
ACCIONES_INC = [-4.0, -1.5, -0.5, -0.15, -0.05, 0.0, 0.05, 0.15, 0.5, 1.5, 4.0]


def lista_acciones(incremental):
    return ACCIONES_INC if incremental else ACCIONES_ABS


class QNetSU(nn.Module):
    def __init__(self, n_acciones, hidden=(256, 256)):
        super().__init__()
        self.register_buffer("x_scale", torch.tensor(X_SCALE_SU, dtype=torch.float32))
        layers, n_in = [], N_OBS
        for h in hidden:
            layers += [nn.Linear(n_in, h), nn.ReLU()]
            n_in = h
        layers += [nn.Linear(n_in, n_acciones)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x / self.x_scale)


class Replay:
    def __init__(self, cap, dim=N_OBS):
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
        OF = torch.stack(self.of)               # (n, N, dim)
        vivo = torch.ones_like(R)               # vivo_k = sin fin de episodio antes del paso k
        vivo[1:] = torch.cumprod(1 - D[:-1].float(), 0)
        pot = self.gamma ** torch.arange(self.n, dtype=R.dtype)[:, None]
        G = (pot * R * vivo).sum(0)
        m = vivo.sum(0).long()                  # pasos usados (1..n)
        idx = m - 1
        cols = torch.arange(R.shape[1])
        out = (self.o[0], self.a[0], G, OF[idx, cols], T[idx, cols], self.gamma ** m.float())
        for lst in (self.o, self.a, self.r, self.term, self.done, self.of):
            lst.pop(0)
        return out


# --------------------------------------------------------------------------
@torch.no_grad()
def evaluar(policy_fn, acciones, incremental, Ts_agente, n=256, t_max=8.0, seed=123,
            theta_lim_deg=120.0, aleatorio=True, preparar=None,
            filtro_w=FILTRO_W, filtro_zeta=FILTRO_ZETA):
    """Desde colgando. Éxito = el brazo nunca pasa del límite y en los últimos
    2 s |alpha| < 5° y |theta| < 45°. Devuelve éxito, subida, temblor de alpha y
    rizo del voltaje que llega al motor."""
    env = QubeSwingUpEnv(n, acciones, incremental=incremental, Ts_agente=Ts_agente,
                         aleatorio=aleatorio, t_max=t_max + 5, p_abajo=1.0, seed=seed,
                         theta_lim_deg=theta_lim_deg, filtro_w=filtro_w, filtro_zeta=filtro_zeta)
    if preparar is not None:
        preparar(env)
    obs = env.observe()
    steps = int(round(t_max / env.Ts_agente))
    A = torch.zeros(steps, n)
    TH = torch.zeros(steps, n)
    UC = torch.zeros(steps, n)
    UF = torch.zeros(steps, n)
    vivo = torch.ones(n, dtype=torch.bool)
    for k in range(steps):
        obs, _, term, _, _ = env.step(policy_fn(obs))
        vivo &= ~term
        A[k] = envolver(env.x[:, 1])
        TH[k] = env.x[:, 0].abs()
        UC[k] = env.u_cmd
        UF[k] = env.f1
    cola = int(round(2.0 / env.Ts_agente))
    ok = vivo & (A[-cola:].abs().max(0).values < 5 * DEG) & (TH[-cola:].max(0).values < 45 * DEG)
    fuera = (A.abs() > 10 * DEG).float()
    idx = torch.arange(1, steps + 1, dtype=torch.float32)[:, None]
    t_sub = (fuera * idx).max(0).values * env.Ts_agente
    res = {"exito": ok.float().mean().item(), "theta_max": TH.max().item() / DEG,
           "t_subida": float("nan"), "t_subida_max": float("nan"),
           "temblor_alpha": float("nan"), "du_rms": float("nan"), "rizo_motor": float("nan")}
    if ok.any():
        res["t_subida"] = t_sub[ok].mean().item()
        res["t_subida_max"] = t_sub[ok].max().item()
        Ac = A[-cola:, ok]
        res["temblor_alpha"] = (Ac - Ac.mean(0)).pow(2).mean().sqrt().item() / DEG   # ° RMS
        res["du_rms"] = torch.diff(UC[-cola:, ok], dim=0).pow(2).mean().sqrt().item()
        Uf = UF[-cola:, ok]
        res["rizo_motor"] = (Uf - Uf.mean(0)).pow(2).mean().sqrt().item()            # V RMS
    return res


def brazo_pesado(factor=1.76):
    """Inercia del brazo como en el modelo anterior del proyecto (Jr + mp r^2)."""
    def preparar(env):
        env.bp.mr[:] = env.p.mr * factor
    return preparar


def resumen(r):
    return (f"éxito={100 * r['exito']:5.1f}%  subida={r['t_subida']:4.2f}s (máx {r['t_subida_max']:4.2f})  "
            f"temblor={r['temblor_alpha']:4.2f}°  rizo={r['rizo_motor']:4.2f}V  du={r['du_rms']:4.2f}V  "
            f"theta_max={r['theta_max']:4.0f}°")


def clave(r):
    """Primero éxito (redondeado), después menos temblor y menos rizo de voltaje."""
    t, z = r["temblor_alpha"], r["rizo_motor"]
    if t != t:
        return (round(r["exito"], 2), -99.0)
    return (round(r["exito"], 2), -(t + z))


# --------------------------------------------------------------------------
@torch.no_grad()
def trayectoria_prueba(q, acciones, incremental, Ts_agente, filtro_w, filtro_zeta, segundos=6.0):
    """Trayectoria determinista (Qube oficial nominal) registrada a 2 ms: los
    ángulos que ve el bloque de Simulink y el voltaje que debe sacar en cada paso."""
    env = QubeSwingUpEnv(1, acciones, incremental=incremental, Ts_agente=Ts_agente,
                         aleatorio=False, t_max=segundos + 5, p_abajo=1.0,
                         filtro_w=filtro_w, filtro_zeta=filtro_zeta)
    x0 = torch.tensor([[0.0, -math.pi, 0.0, 0.0]])
    obs = env.reset(torch.ones(1, dtype=torch.bool), x0)
    X, U, XR = [], [], []
    w, z = filtro_w, filtro_zeta
    vmax = env.p.Vmax
    for _ in range(int(round(segundos / env.Ts_agente))):
        a = q(obs).argmax(1)
        u = env.acciones[a]
        if incremental:
            u = torch.clamp(env.u_cmd + u, -vmax, vmax)
        for _ in range(env.substeps):
            aw = torch.remainder(env.y[0, 1] + math.pi, 2 * math.pi) - math.pi
            X.append([env.y[0, 0].item(), aw.item(), 0.0, 0.0])
            U.append(env.f1.item())                       # salida del bloque (filtrada)
            XR.append(env.x[0].tolist())
            env.x = paso_rk4(env.x, env.f1, env.bp, env.dt)
            f1n = env.f1 + env.dt * env.f2
            f2n = env.f2 + env.dt * (w * w * (u - env.f1) - 2 * z * w * env.f2)
            env.f1, env.f2 = f1n, f2n
            env._update_sensor()
        env.u_cmd = u.clone()
        obs = env.observe()
    return np.array(X), np.array(U).reshape(-1, 1), np.array(XR)


def exportar(q, acciones, incremental, env, path, theta_max_deg, extra):
    lin = [m for m in q.net if isinstance(m, nn.Linear)]
    red = {}
    for i, m in enumerate(lin, start=1):
        red[f"W{i}"] = m.weight.detach().double().numpy()
        red[f"b{i}"] = m.bias.detach().double().numpy().reshape(-1, 1)
    red["x_scale"] = q.x_scale.double().numpy().reshape(-1, 1)
    red["acciones"] = np.asarray(acciones, dtype=float).reshape(-1, 1)
    red["incremental"] = float(incremental)
    red["Vmax"] = float(env.p.Vmax)
    red["Ts"] = float(env.dt)
    red["n_hold"] = float(env.substeps)
    a_obs = np.asarray(A_OBS, dtype=float).reshape(1, -1)           # [lento, rápido]
    red["obs_beta"] = 3 * a_obs
    red["obs_L"] = 3 * a_obs ** 2
    red["obs_m"] = a_obs ** 3 - red["obs_L"] * red["obs_beta"]
    red["filtro_w"] = float(env.filtro_w)
    red["filtro_zeta"] = float(env.filtro_zeta)
    red["theta_max"] = float(theta_max_deg * DEG)
    Xt, Ut, XRt = trayectoria_prueba(q, acciones, incremental, env.Ts_agente,
                                     env.filtro_w, env.filtro_zeta)
    d = {"red_swingup": red, "x_test": Xt, "u_test": Ut, "x_real_test": XRt}
    bloque = ROOT / "matlab" / "dqn_swingup_block.m"
    if bloque.exists():
        d["codigo_bloque"] = bloque.read_text(encoding="utf-8")
    d.update(extra)
    sio.savemat(str(path), d, oned_as="column")


def cargar_red(pt):
    ck = torch.load(pt, weights_only=False)
    a = ck["meta"]["args"] if "meta" in ck else ck["args"]
    acc = lista_acciones(a["incremental"])
    q = QNetSU(len(acc), tuple(a["hidden"]))
    q.load_state_dict(ck["state_dict"])
    return q, a, acc


def exportar_desde_pt(pt, out):
    """Valida y exporta a .mat un checkpoint guardado durante el entrenamiento."""
    q, a, acc = cargar_red(pt)
    inc = a["incremental"]
    fw, fz = a.get("filtro_w", FILTRO_W), a.get("filtro_zeta", FILTRO_ZETA)
    env = QubeSwingUpEnv(1, acc, incremental=inc, Ts_agente=a["ts_agente"],
                         theta_lim_deg=a["theta_lim"], filtro_w=fw, filtro_zeta=fz)
    pol = lambda y: q(y).argmax(1)  # noqa: E731
    kw = dict(theta_lim_deg=a["theta_lim"], filtro_w=fw, filtro_zeta=fz)
    r = evaluar(pol, acc, inc, a["ts_agente"], n=1000, seed=999, **kw)
    r_nom = evaluar(pol, acc, inc, a["ts_agente"], n=64, seed=5, aleatorio=False, **kw)
    r_pes = evaluar(pol, acc, inc, a["ts_agente"], n=64, seed=5, aleatorio=False,
                    preparar=brazo_pesado(), **kw)
    meta = {"acciones": acc, "incremental": inc, "Ts_agente": env.Ts_agente, "validacion": r,
            "validacion_oficial_nominal": r_nom, "validacion_brazo_pesado": r_pes, "args": a}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": q.state_dict(), "meta": meta}, out.with_suffix(".pt"))
    exportar(q, acc, inc, env, out.with_suffix(".mat"), theta_max_deg=a["theta_lim"] + 30,
             extra={"exito_val": r["exito"]})
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Qube aleatorio      (1000): {resumen(r)}")
    print(f"Modelo oficial nominal (64): {resumen(r_nom)}")
    print(f"Brazo pesado x1.76     (64): {resumen(r_pes)}")
    print(f"->  {out.with_suffix('.mat')}")


def iniciar_desde(q, pt):
    """Copia los pesos de otro checkpoint; si tiene menos entradas (versión
    anterior), las columnas nuevas de la primera capa empiezan en cero."""
    sd = torch.load(pt, weights_only=False)["state_dict"]
    propio = q.state_dict()
    for k, v in sd.items():
        if k not in propio or k == "x_scale":
            continue
        if v.shape == propio[k].shape:
            propio[k] = v.clone()
        elif v.dim() == 2 and v.shape[0] == propio[k].shape[0] and v.shape[1] < propio[k].shape[1]:
            nuevo = torch.zeros_like(propio[k])
            nuevo[:, :v.shape[1]] = v
            propio[k] = nuevo
    q.load_state_dict(propio)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incremental", action="store_true", help="acciones = incrementos de voltaje")
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
    ap.add_argument("--eps-inicial", type=float, default=1.0)
    ap.add_argument("--eps-pasos", type=int, default=30_000)
    ap.add_argument("--eps-final", type=float, default=0.05)
    ap.add_argument("--ts-agente", type=float, default=0.01)
    ap.add_argument("--theta-lim", type=float, default=120.0)
    ap.add_argument("--filtro-w", type=float, default=FILTRO_W, help="corte del filtro (rad/s)")
    ap.add_argument("--filtro-zeta", type=float, default=FILTRO_ZETA)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--eval-cada", type=int, default=5_000)
    ap.add_argument("--out", default=str(ROOT / "models" / NOMBRE))
    ap.add_argument("--exportar-pt", default="", help="solo exporta este .pt a --out (.mat)")
    ap.add_argument("--init-pt", default="", help="continúa el entrenamiento desde este .pt")
    args = ap.parse_args()
    if args.exportar_pt:
        return exportar_desde_pt(args.exportar_pt, Path(args.out))

    print(f"[modelo oficial] deriva de energía sin motor ni fricción: {prueba_energia():.1e}")
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    acciones = lista_acciones(args.incremental)
    nA = len(acciones)
    env = QubeSwingUpEnv(args.n_envs, acciones, incremental=args.incremental,
                         Ts_agente=args.ts_agente, seed=args.seed, theta_lim_deg=args.theta_lim,
                         filtro_w=args.filtro_w, filtro_zeta=args.filtro_zeta)
    q = QNetSU(nA, tuple(args.hidden))
    if args.init_pt:
        iniciar_desde(q, args.init_pt)
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
    retornos, mejor = [], (-1.0, 0.0)
    t0, n_grad = time.time(), 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pol = lambda y: q(y).argmax(1)  # noqa: E731

    for paso in range(1, args.pasos + 1):
        eps = max(args.eps_final,
                  args.eps_inicial - (args.eps_inicial - args.eps_final) * paso / args.eps_pasos)
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
            kw = dict(theta_lim_deg=args.theta_lim, filtro_w=args.filtro_w,
                      filtro_zeta=args.filtro_zeta)
            res = evaluar(pol, acciones, args.incremental, args.ts_agente, **kw)
            nom = evaluar(pol, acciones, args.incremental, args.ts_agente, n=64, seed=5,
                          aleatorio=False, **kw)
            if clave(res) >= mejor:
                mejor = clave(res)
                torch.save({"state_dict": copy.deepcopy(q.state_dict()), "args": vars(args),
                            "validacion": res, "paso": paso}, out.with_suffix(".pt"))
            rm = np.mean(retornos[-100:]) if retornos else float("nan")
            print(f"paso {paso:6d}/{args.pasos}  eps={eps:.2f}  retorno={rm:7.1f}  {resumen(res)}  "
                  f"| oficial: {100 * nom['exito']:3.0f}% {nom['temblor_alpha']:4.2f}° "
                  f"{nom['rizo_motor']:4.2f}V  [{time.time() - t0:5.0f}s]", flush=True)

    exportar_desde_pt(out.with_suffix(".pt"), out)


if __name__ == "__main__":
    main()
