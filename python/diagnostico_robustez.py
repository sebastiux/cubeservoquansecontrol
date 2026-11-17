"""
¿Qué diferencia entre el modelo y el Qube real hace fallar a una red?

Simula cada red con UNA diferencia a la vez respecto al modelo oficial nominal
(inercia del brazo, ganancia del motor, retardo extra, zona muerta, fricción
seca, un segundo filtro en serie) y mide si sube el péndulo, si lo sostiene y
cuánto tiembla. Comparando con lo que se observa en el hardware se identifica
la causa más probable.

Uso:
    python diagnostico_robustez.py red1.pt[:w_filtro] red2.pt[:w_filtro] ...
    (w_filtro: corte del filtro con que se usa esa red; por defecto, el de su
     entrenamiento. Para la v2, que no trae filtro: v2.pt:200 = con un filtro
     externo de 200 rad/s.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import rl_env_swingup as E
from analisis_filtro import cargar_politica

CASOS = [
    ("modelo oficial nominal", {}),
    ("inercia del brazo x0.5", dict(mrf=0.5)),
    ("inercia del brazo x1.76", dict(mrf=1.76)),
    ("inercia del brazo x2.5", dict(mrf=2.5)),
    ("motor x0.6", dict(ku=0.6)),
    ("motor x1.6", dict(ku=1.6)),
    ("retardo extra 6 ms", dict(ret=3)),
    ("retardo extra 10 ms", dict(ret=5)),
    ("retardo extra 14 ms", dict(ret=7)),
    ("zona muerta 0.6 V", dict(zm=0.6)),
    ("zona muerta 1.0 V", dict(zm=1.0)),
    ("fricción seca 6 mN·m", dict(tc=6e-3)),
    ("2.º filtro de 200 rad/s en serie", dict(w2=200.0)),
]


@torch.no_grad()
def prueba(pt, w, mrf=1.0, ku=1.0, ret=0, tc=0.0, zm=0.0, w2=0.0, n=48, seg=10.0):
    pol, acc, inc, ts = cargar_politica(pt)
    if w is None:
        w = pol.filtro[0]
    env = E.QubeSwingUpEnv(n, acc, incremental=inc, Ts_agente=ts, aleatorio=False, t_max=seg + 5,
                           p_abajo=1.0, seed=4, filtro_w=w, filtro_zeta=1.0)
    env.bp.mr[:] = env.p.mr * mrf
    env.k_u[:] = ku
    env.bp.tau_c[:] = tc
    env.zona_muerta[:] = zm
    env.filtro2_w = w2
    env.u_fifo = torch.zeros(n, max(ret, E.MAX_RETARDO) + 1)
    env.retardo[:] = ret
    obs = env.observe()
    A = []
    for _ in range(int(seg / env.Ts_agente)):
        obs, _, _, _, _ = env.step(pol(obs))
        A.append(E.envolver(env.x[:, 1]).abs())
    A = torch.stack(A)
    c = int(3 / env.Ts_agente)
    subio = A.min(0).values < 10 * E.DEG
    sostiene = A[-c:].max(0).values < 10 * E.DEG
    temblor = A[-c:][:, sostiene].std().item() / E.DEG if sostiene.any() else float("nan")
    return (f"sube {100 * subio.float().mean():3.0f}%  sostiene {100 * sostiene.float().mean():3.0f}%  "
            f"temblor {temblor:4.2f}°")


def main():
    redes = []
    for arg in sys.argv[1:]:
        pt, _, w = arg.partition(":")
        if len(pt) < 3 and w:                       # ruta de Windows tipo C:\...
            pt, w = arg, ""
        redes.append((Path(pt).stem + (f" + filtro {w}" if w else ""), pt, float(w) if w else None))
    for nombre, cfg in CASOS:
        print(nombre)
        for etiqueta, pt, w in redes:
            print(f"    {etiqueta:34s} {prueba(pt, w, **cfg)}", flush=True)


if __name__ == "__main__":
    main()
