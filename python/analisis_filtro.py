"""
¿Qué filtro pasa bajas conviene a la salida de la red para proteger el motor?

Barre la frecuencia de corte w y el amortiguamiento zeta del filtro de 2.º orden
    H(s) = w^2 / (s^2 + 2 zeta w s + w^2)
y simula el lazo completo (modelo no lineal oficial de Quanser + encoders +
observadores + red DQN) en los mismos Qubes aleatorios para cada filtro.

Para cada filtro mide, con el péndulo ya arriba (últimos 3 s):
  CONTROL    éxito, tiempo de subida, temblor de alpha
  ACTUADOR   rizo del voltaje del motor, dV/dt, inversiones de voltaje por
             segundo, corriente RMS y potencia disipada en el devanado (Rm i^2),
             y el mayor salto de corriente en 2 ms de toda la maniobra
y lo compara con la teoría: retraso del filtro (2 zeta / w) y atenuación a la
frecuencia de conmutación de la red.

Dos análisis:
  1) Barrido con UNA red fija (que no conoce el filtro): muestra cuánto protege
     el filtro y cuánto control cuesta su retraso.
         python analisis_filtro.py [red.pt]            (por defecto la red v2)
  2) Co-diseño: una red entrenada CON cada filtro candidato; cada una se mide
     con su propio filtro y se elige el óptimo.
         python analisis_filtro.py --codiseno a.pt b.pt c.pt ...
Salida:
    ../models/analisis_filtro*.json  y  ../models/figuras/analisis_filtro*.png
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import rl_env_swingup as E

ROOT = Path(__file__).resolve().parent.parent
DT = 0.002
W_LISTA = [40, 60, 80, 100, 150, 200, 300, 400, 600]
ZETAS = [1.0, 0.707]


def cargar_politica(pt):
    """Devuelve (política, acciones, incremental, Ts_agente). Acepta redes v2 (8
    entradas, sin estado del filtro) y v3 (10 entradas)."""
    ck = torch.load(pt, weights_only=False)
    sd = ck["state_dict"]
    a = ck["meta"]["args"] if "meta" in ck else ck["args"]
    n_in, n_out = sd["net.0.weight"].shape[1], sd["net.4.weight"].shape[0]
    h = sd["net.0.weight"].shape[0]
    net = nn.Sequential(nn.Linear(n_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_out))
    net.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("net.")})
    escala = sd["x_scale"]
    incremental = bool(a.get("incremental", False))
    from train_dqn_swingup import lista_acciones
    acciones = lista_acciones(incremental)
    assert len(acciones) == n_out
    pol = (lambda y: net(y[:, :n_in] / escala).argmax(1))
    pol.filtro = (a.get("filtro_w", 0.0), a.get("filtro_zeta", 1.0))   # filtro con que se entrenó
    return pol, acciones, incremental, a["ts_agente"]


@torch.no_grad()
def medir(pol, acciones, incremental, ts, w, zeta, n=192, segundos=8.0, seed=11):
    env = E.QubeSwingUpEnv(n, acciones, incremental=incremental, Ts_agente=ts, aleatorio=True,
                           t_max=segundos + 5, p_abajo=1.0, seed=seed, filtro_w=w, filtro_zeta=zeta)
    env.registro = []
    obs = env.observe()
    for _ in range(int(round(segundos / env.Ts_agente))):
        obs, _, _, _, _ = env.step(pol(obs))
    V = torch.stack([r[0] for r in env.registro])            # (T, n) voltaje al motor
    X = torch.stack([r[1] for r in env.registro])            # (T, n, 4)
    A = E.envolver(X[:, :, 1])
    TH = X[:, :, 0]
    Icorr = (V - env.bp.km * X[:, :, 2]) / env.bp.Rm         # corriente del motor (A)
    T = V.shape[0]
    cola = slice(T - int(3.0 / DT), T)
    ok = (TH.abs().max(0).values < env.theta_lim) & (A[cola].abs().max(0).values < 5 * E.DEG) \
        & (TH[cola].abs().max(0).values < 45 * E.DEG)
    idx = torch.arange(1, T + 1, dtype=torch.float32)[:, None]
    t_sub = ((A.abs() > 10 * E.DEG).float() * idx).max(0).values * DT
    r = {"w": w, "zeta": zeta, "exito": ok.float().mean().item()}
    if not ok.any():
        return r
    v, i, a = V[cola][:, ok], Icorr[cola][:, ok], A[cola][:, ok]
    signo = torch.sign(torch.where(v.abs() < 0.05, torch.zeros_like(v), v))
    cambios = ((signo[1:] * signo[:-1]) < 0).float().sum(0) / 3.0
    r.update({
        "t_subida": t_sub[ok].mean().item(),
        "temblor_alpha": (a - a.mean(0)).pow(2).mean().sqrt().item() / E.DEG,
        "rizo_v": (v - v.mean(0)).pow(2).mean().sqrt().item(),
        "dvdt": (torch.diff(v, dim=0) / DT).pow(2).mean().sqrt().item(),
        "inversiones": cambios.mean().item(),
        "i_rms": i.pow(2).mean().sqrt().item(),
        "potencia": (env.bp.Rm[ok] * i.pow(2)).mean().item(),
        "salto_i_max": torch.diff(Icorr[:, ok], dim=0).abs().max().item(),
    })
    return r


def teoria(w, zeta, f_hz):
    """Retraso de grupo en baja frecuencia (ms) y ganancia del filtro a f_hz."""
    s = 1j * 2 * math.pi * f_hz
    return 2 * zeta / w * 1e3, abs(w * w / (s * s + 2 * zeta * w * s + w * w))


CABECERA = (" zeta    w   retraso  |H(50Hz)|  éxito  subida  temblor  rizo_V   dV/dt  inver/s  I_rms   "
            "P_Joule  salto_I  esfuerzo")


def linea(r, nombre):
    if "rizo_v" not in r:
        return f"{nombre} {'':17s} {100 * r['exito']:5.1f}%   (no se sostiene)"
    return (f"{nombre} {r['retraso_ms']:5.1f}ms   {r['gan_50hz']:5.2f}    {100 * r['exito']:5.1f}%  "
            f"{r['t_subida']:5.2f}s  {r['temblor_alpha']:5.2f}°  {r['rizo_v']:5.2f}V  {r['dvdt']:6.0f}  "
            f"{r['inversiones']:6.1f}  {r['i_rms']:5.3f}A  {r['potencia']:6.3f}W  {r['salto_i_max']:5.2f}A  "
            f"{r['esfuerzo']:6.2f}")


def normalizar(filas, base):
    for r in filas:
        r["retraso_ms"], r["gan_50hz"] = (0.0, 1.0) if r["w"] <= 0 else teoria(r["w"], r["zeta"], 50.0)
        if "rizo_v" not in r:
            r["esfuerzo"] = r["control"] = float("nan")
            continue
        r["esfuerzo"] = float(np.mean([r["dvdt"] / base["dvdt"], r["potencia"] / base["potencia"],
                                       r["inversiones"] / max(base["inversiones"], 1e-9)]))
        r["control"] = r["temblor_alpha"] / base["temblor_alpha"]


def codiseno(pts):
    """Una red entrenada con cada filtro: cada una se mide con SU filtro."""
    torch.manual_seed(0)
    v2 = str(ROOT / "models" / "dqn_swingup_v2_qube.pt")
    pol, acc, inc, ts = cargar_politica(v2)
    base = medir(pol, acc, inc, ts, 0.0, 1.0)
    base["red"] = "v2 (sin filtro)"
    filas = [base]
    for pt in pts:
        pol, acc, inc, ts = cargar_politica(pt)
        w, z = pol.filtro
        r = medir(pol, acc, inc, ts, w, z)
        r["red"] = f"entrenada con w={w:.0f}"
        r["pt"] = str(pt)
        filas.append(r)
        # la red v2, que no conoce el filtro, con ese mismo filtro
        pol2, acc2, inc2, ts2 = cargar_politica(v2)
        r2 = medir(pol2, acc2, inc2, ts2, w, z)
        r2["red"] = "v2 + ese filtro"
        filas.append(r2)
    normalizar(filas, base)
    print(f"{'red':22s}" + CABECERA)
    for r in filas:
        nombre = "   sin filtro " if r["w"] <= 0 else f"{r['zeta']:5.3f} {r['w']:4.0f} "
        print(f"{r['red']:22s}" + linea(r, nombre))
    cand = [r for r in filas if r["red"].startswith("entrenada") and "rizo_v" in r]
    mejor_exito = max(r["exito"] for r in cand)
    mejor_temblor = min(r["temblor_alpha"] for r in cand)
    validos = [r for r in cand if r["exito"] >= mejor_exito - 0.015
               and r["temblor_alpha"] <= 1.25 * mejor_temblor]
    optimo = min(validos, key=lambda r: r["esfuerzo"])
    print(f"\nÓptimo: w = {optimo['w']:.0f} rad/s, zeta = {optimo['zeta']}  "
          f"(menor esfuerzo del actuador entre los candidatos con éxito >= {100 * (mejor_exito - 0.015):.1f} % "
          f"y temblor <= {1.25 * mejor_temblor:.2f}°)")
    print(f"   frente a no filtrar: dV/dt x{optimo['dvdt'] / base['dvdt']:.2f}, "
          f"inversiones x{optimo['inversiones'] / max(base['inversiones'], 1e-9):.2f}, "
          f"potencia x{optimo['potencia'] / base['potencia']:.2f}, "
          f"salto de corriente {base['salto_i_max']:.2f} A -> {optimo['salto_i_max']:.2f} A; "
          f"temblor {base['temblor_alpha']:.2f}° -> {optimo['temblor_alpha']:.2f}°, "
          f"éxito {100 * base['exito']:.1f} % -> {100 * optimo['exito']:.1f} %")
    (ROOT / "models" / "analisis_filtro_codiseno.json").write_text(
        json.dumps({"filas": filas, "optimo": optimo}, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return optimo
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    grupos = [("entrenada", "tab:green", "red entrenada con el filtro"),
              ("v2 +", "tab:red", "red que no conoce el filtro")]
    paneles = [("temblor_alpha", "Temblor de α (° RMS)", 1), ("exito", "Éxito (%)", 100),
               ("esfuerzo", "Esfuerzo del actuador (1 = sin filtro)", 1)]
    for a_, (k, titulo, esc) in zip(ax, paneles):
        for pref, color, etiqueta in grupos:
            fs = sorted([r for r in filas if r["red"].startswith(pref) and k in r and r[k] == r[k]],
                        key=lambda r: r["w"])
            a_.plot([r["w"] for r in fs], [esc * r[k] for r in fs], "o-", color=color, label=etiqueta)
        a_.axhline(esc * base[k], color="gray", ls="--", label="sin filtro")
        a_.axvline(optimo["w"], color="k", ls=":", label="óptimo")
        a_.set_title(titulo); a_.set_xlabel("corte del filtro w (rad/s)"); a_.grid(True, alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("Co-diseño filtro + red: qué corte protege al motor sin perder control")
    fig.tight_layout()
    figs = ROOT / "models" / "figuras"
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / "analisis_filtro_codiseno.png", dpi=110)
    print(f"Figura: {figs / 'analisis_filtro_codiseno.png'}")
    return optimo


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--codiseno":
        return codiseno(sys.argv[2:])
    pt = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "models" / "dqn_swingup_v2_qube.pt")
    pol, acciones, incremental, ts = cargar_politica(pt)
    torch.manual_seed(0)
    filas = [medir(pol, acciones, incremental, ts, 0.0, 1.0)]
    for zeta in ZETAS:
        for w in W_LISTA:
            filas.append(medir(pol, acciones, incremental, ts, float(w), zeta))
            print(".", end="", flush=True)
    print()
    base = filas[0]

    normalizar(filas, base)           # índices respecto a "sin filtro"

    print(f"Red: {Path(pt).name}   decisiones cada {ts * 1e3:.0f} ms   acciones: "
          f"{'incrementos' if incremental else 'voltajes'}")
    print(CABECERA)
    for r in filas:
        print(linea(r, "   sin filtro " if r["w"] <= 0 else f"{r['zeta']:5.3f} {r['w']:4.0f} "))

    # ---- filtro óptimo: el de menor esfuerzo del actuador que no degrada el control
    validos = [r for r in filas[1:] if "rizo_v" in r and r["exito"] >= base["exito"] - 0.01
               and r["temblor_alpha"] <= 1.10 * base["temblor_alpha"]]
    optimo = min(validos, key=lambda r: r["esfuerzo"]) if validos else None
    if optimo:
        print(f"\nÓptimo (menor esfuerzo del actuador sin perder éxito ni aumentar el temblor >10 %): "
              f"w = {optimo['w']:.0f} rad/s, zeta = {optimo['zeta']}")
        print(f"   esfuerzo del actuador x{optimo['esfuerzo']:.2f} respecto a sin filtro "
              f"(dV/dt x{optimo['dvdt'] / base['dvdt']:.2f}, potencia x{optimo['potencia'] / base['potencia']:.2f}, "
              f"inversiones x{optimo['inversiones'] / max(base['inversiones'], 1e-9):.2f}); "
              f"temblor x{optimo['control']:.2f}")
    out = ROOT / "models" / "analisis_filtro.json"
    out.write_text(json.dumps({"red": Path(pt).name, "filas": filas, "optimo": optimo}, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    colores = {1.0: "tab:blue", 0.707: "tab:orange"}
    paneles = [("temblor_alpha", "Temblor de α (° RMS)"), ("exito", "Éxito (%)"),
               ("dvdt", "dV/dt del motor (V/s RMS)"), ("potencia", "Potencia disipada Rm·i² (W)")]
    for a_, (clave, titulo) in zip(ax.ravel(), paneles):
        for zeta in ZETAS:
            fs = [r for r in filas[1:] if r["zeta"] == zeta and clave in r]
            esc = 100 if clave == "exito" else 1
            a_.plot([r["w"] for r in fs], [esc * r[clave] for r in fs], "o-", color=colores[zeta],
                    label=f"ζ = {zeta}")
        if clave in base:
            a_.axhline((100 if clave == "exito" else 1) * base[clave], color="gray", ls="--",
                       label="sin filtro")
        if optimo:
            a_.axvline(optimo["w"], color="green", ls=":", label="óptimo")
        a_.set_xscale("log"); a_.set_title(titulo); a_.grid(True, which="both", alpha=0.3)
    ax[0, 0].legend()
    for a_ in ax[1]:
        a_.set_xlabel("frecuencia de corte del filtro w (rad/s)")
    fig.suptitle("Filtro pasa bajas a la salida de la red: control vs. esfuerzo del actuador")
    fig.tight_layout()
    figs = ROOT / "models" / "figuras"
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / "analisis_filtro.png", dpi=110)
    print(f"Figura: {figs / 'analisis_filtro.png'}")


if __name__ == "__main__":
    main()
