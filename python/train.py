"""
Entrenamiento del controlador neuronal para BALANCEAR el péndulo del
Qube-Servo 3 usando el MODELO NO LINEAL.

Etapa 1 - Imitación + DAgger (rápida, da una buena inicialización)
    La red aprende a imitar al LQR saturado (u = sat(-K x)) sobre estados
    visitados por la propia red en la planta NO lineal, con parámetros
    aleatorizados.

Etapa 2 - Optimización directa de la política (BPTT por la simulación)
    Se simula en paralelo la planta no lineal (RK4, Ts = 2 ms) con:
      * parámetros físicos aleatorizados (±15 %),
      * encoders cuantizados (2048 cuentas/rev),
      * velocidades por derivada filtrada 50s/(s+50),
      * ruido en las velocidades,
    y se minimiza directamente el costo
        J = sum_k (x'Qx + R u^2 + Rd du^2) Ts + x_T' P x_T
    con los mismos Q, R de tu LQR (P = Riccati -> costo terminal).
    Así la red supera al LQR lejos del origen (no linealidades + saturación).

Uso:
    python train.py                 # entrenamiento completo (~10-20 min CPU)
    python train.py --quick         # prueba rápida (~1-2 min)
Salida:
    ../models/nn_qube_policy.pt     (PyTorch)
    ../models/nn_qube_policy.mat    (pesos para MATLAB/Simulink)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from export_to_matlab import export_mat, export_onnx
from policy import NNController
from qube_model import (Q_LQR, R_LQR, BatchParams, QubeParams, SensorModel,
                        linear_model, linearize_numeric, lqr)
from sim import lqr_controller, rollout, sample_initial_states, success_mask

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

DEG = math.pi / 180


# --------------------------------------------------------------------------
def check_model(p: QubeParams):
    """Verifica que el Jacobiano del modelo no lineal == A, B de tu .m."""
    A, B = linear_model(p)
    An, Bn = linearize_numeric(p)
    err = max(np.abs(A - An).max(), np.abs(B - Bn).max())
    print(f"[modelo] |A_nl - A_lin|max, |B_nl - B_lin|max = {err:.2e}")
    assert err < 1e-4, "El modelo no lineal no coincide con la linealización"


# --------------------------------------------------------------------------
def stage1_dagger(net, p, K, args, gen):
    """Imitación del LQR saturado con agregación de datos (DAgger)."""
    print("\n=== Etapa 1: imitación del LQR + DAgger (modelo no lineal) ===")
    expert = lqr_controller(K, p.Vmax)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    ranges0 = (60 * DEG, 25 * DEG, 3.0, 4.0)
    box = torch.tensor((90 * DEG, 35 * DEG, 8.0, 15.0))

    data = []
    # Cobertura uniforme del espacio de estados alrededor del equilibrio
    data.append((torch.rand(args.bc_uniform, 4, generator=gen) * 2 - 1) * box)

    for it in range(args.dagger_iters):
        beta = max(0.0, 1.0 - it / max(1, args.dagger_iters - 1))
        with torch.no_grad():
            x0 = sample_initial_states(args.dagger_batch, ranges0, gen)
            bp = BatchParams(p, x0.shape[0], rel_spread=0.15, generator=gen)
            out = rollout(net, x0, bp, args.dagger_steps, p.Ts, sensor=None,
                          mix_ctrl=expert, beta=beta, u_noise_std=0.5)
            Xs = out["X"][::5].reshape(-1, 4)
            keep = (Xs[:, 1].abs() < 45 * DEG) & (Xs[:, 0].abs() < 2.5)
            data.append(Xs[keep])
        D = torch.cat(data)
        for _ in range(args.bc_steps):
            idx = torch.randint(0, D.shape[0], (1024,), generator=gen)
            xb = D[idx]
            xb_noisy = xb + torch.randn_like(xb) * torch.tensor([0.003, 0.003, 0.05, 0.1])
            target = expert(xb)
            loss = (((net(xb_noisy) - target) / p.Vmax) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"  DAgger {it + 1}/{args.dagger_iters}  beta={beta:.2f}  "
              f"datos={D.shape[0]:7d}  mse={loss.item():.2e}")


# --------------------------------------------------------------------------
def trajectory_cost(X, U, Q, R, Rd, P, dt):
    """Costo LQR no lineal por trayectoria. X (T+1,B,4), U (T,B)."""
    Xc = X.clone()
    # Usar 2 sin(a/2) en lugar de a: idéntico cerca de 0, acotado lejos
    Xc[..., 1] = 2 * torch.sin(X[..., 1] / 2)
    stage = (Xc[:-1] @ Q * Xc[:-1]).sum(-1) + R * U ** 2
    du = torch.diff(U, dim=0, prepend=U[:1])
    stage = stage + Rd * du ** 2
    stage = 50.0 * torch.tanh(stage / 50.0)           # acota trayectorias perdidas
    xT = Xc[-1]
    term = (xT @ P * xT).sum(-1)
    term = 50.0 * torch.tanh(term / 50.0)
    return stage.sum(0) * dt + term


def stage2_bptt(net, p, P, args, gen):
    print("\n=== Etapa 2: optimización directa por BPTT (no lineal + sensores) ===")
    Q = torch.tensor(Q_LQR, dtype=torch.float32)
    P = torch.tensor(P, dtype=torch.float32)
    R = float(R_LQR[0, 0])
    Rd = args.rd
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=args.lr / 10)

    best, best_state = -1.0, None
    t0 = time.time()
    for it in range(args.iters):
        # Currículo: los rangos y el horizonte crecen con el entrenamiento
        prog = min(1.0, (it + 1) / (0.6 * args.iters))
        ranges = (
            (20 + 40 * prog) * DEG,      # theta0
            (8 + 22 * prog) * DEG,       # alpha0 (hasta ±30°, el switch es ±15°)
            0.5 + 2.5 * prog,            # theta_dot0
            1.0 + 4.0 * prog,            # alpha_dot0
        )
        steps = int(args.horizon * (0.5 + 0.5 * prog) / p.Ts)

        x0 = sample_initial_states(args.batch, ranges, gen)
        bp = BatchParams(p, args.batch, rel_spread=0.15, generator=gen)
        sensor = SensorModel(quantize=True, filtered_velocity=True,
                             vel_noise_std=args.vel_noise)
        out = rollout(net, x0, bp, steps, p.Ts, sensor=sensor)
        J = trajectory_cost(out["X"], out["U"], Q, R, Rd, P, p.Ts)
        loss = J.mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if (it + 1) % args.eval_every == 0 or it == args.iters - 1:
            sr = evaluate_success(net, p, gen)
            if sr >= best:
                best = sr
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            print(f"  it {it + 1:4d}/{args.iters}  J={loss.item():8.4f}  "
                  f"T={steps * p.Ts:.2f}s  alpha0<=±{ranges[1] / DEG:4.1f}°  "
                  f"éxito(val)={100 * sr:5.1f}%  [{time.time() - t0:5.0f}s]")
    if best_state is not None:
        net.load_state_dict(best_state)
    return best


@torch.no_grad()
def evaluate_success(ctrl, p, gen, n=512, seconds=4.0,
                     ranges=(45 * DEG, 25 * DEG, 2.0, 4.0)):
    g = torch.Generator().manual_seed(1234)   # validación fija
    x0 = sample_initial_states(n, ranges, g)
    bp = BatchParams(p, n, rel_spread=0.15, generator=g)
    sensor = SensorModel(quantize=True, filtered_velocity=True, vel_noise_std=0.0)
    out = rollout(ctrl, x0, bp, int(seconds / p.Ts), p.Ts, sensor=sensor)
    return success_mask(out["X"]).float().mean().item()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="entrenamiento corto de prueba")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vmax", type=float, default=10.0)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--dagger-iters", type=int, default=8)
    ap.add_argument("--dagger-batch", type=int, default=1024)
    ap.add_argument("--dagger-steps", type=int, default=1000)
    ap.add_argument("--bc-steps", type=int, default=1500)
    ap.add_argument("--bc-uniform", type=int, default=200_000)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--horizon", type=float, default=1.2, help="s de BPTT")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--rd", type=float, default=0.05, help="peso de du^2")
    ap.add_argument("--vel-noise", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--out", type=str, default=str(MODELS / "nn_qube_policy"))
    args = ap.parse_args()

    if args.quick:
        args.dagger_iters, args.dagger_batch, args.dagger_steps = 3, 256, 500
        args.bc_steps, args.bc_uniform = 400, 50_000
        args.iters, args.batch, args.horizon, args.eval_every = 40, 128, 0.6, 10

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    p = QubeParams(Vmax=args.vmax)
    check_model(p)
    A, B = linear_model(p)
    K, P = lqr(A, B, Q_LQR, R_LQR)
    print(f"[LQR] K = {np.array2string(K.ravel(), precision=4)}")

    net = NNController(hidden=tuple(args.hidden), vmax=p.Vmax)

    stage1_dagger(net, p, K, args, gen)
    sr_bc = evaluate_success(net, p, gen)
    sr_lqr = evaluate_success(lqr_controller(K, p.Vmax), p, gen)
    print(f"  éxito(val): NN-imitación={100 * sr_bc:.1f}%   LQR={100 * sr_lqr:.1f}%")

    best = stage2_bptt(net, p, P, args, gen)
    print(f"\nÉxito final en validación: NN={100 * best:.1f}%  vs  LQR={100 * sr_lqr:.1f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "params": p.to_dict(), "K_lqr": K.ravel().tolist(),
        "Q": Q_LQR.diagonal().tolist(), "R": float(R_LQR[0, 0]),
        "hidden": args.hidden, "x_scale": net.x_scale.tolist(),
        "success_val_nn": best, "success_val_lqr": sr_lqr, "args": vars(args),
    }
    torch.save({"state_dict": net.state_dict(), "meta": meta}, out.with_suffix(".pt"))
    export_mat(net, p, K, out.with_suffix(".mat"))
    export_onnx(net, out.with_suffix(".onnx"))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Guardado: {out.with_suffix('.pt')}, {out.with_suffix('.mat')}")


if __name__ == "__main__":
    main()
