"""
Exporta la red entrenada a MATLAB.

  * .mat  -> W1, b1, W2, b2, W3, b3, x_scale, Vmax, Ts, K_lqr, params,
             x_test/u_test (para verificar el forward en MATLAB)
             (lo usan matlab/nn_controller.m y el bloque MATLAB Function)
  * .onnx -> opcional, para importNetworkFromONNX (Deep Learning Toolbox)

Uso independiente:
    python export_to_matlab.py ../models/nn_qube_policy.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


def export_mat(net, p, K, path):
    layers = net.linear_layers()
    d = {}
    for i, lin in enumerate(layers, start=1):
        d[f"W{i}"] = lin.weight.detach().double().numpy()             # (n_out, n_in)
        d[f"b{i}"] = lin.bias.detach().double().numpy().reshape(-1, 1)  # columna
    d["n_layers"] = float(len(layers))
    d["x_scale"] = net.x_scale.double().numpy().reshape(-1, 1)
    d["Vmax"] = float(net.vmax)
    d["Ts"] = float(p.Ts)
    d["K_lqr"] = np.asarray(K, dtype=float).reshape(1, 4)
    d["params"] = {k: float(v) for k, v in p.to_dict().items()}
    # Vectores de prueba: MATLAB verifica que su forward da lo mismo que PyTorch
    g = torch.Generator().manual_seed(7)
    xt = (torch.rand(200, 4, generator=g) * 2 - 1) * torch.tensor([1.0, 0.6, 6.0, 12.0])
    with torch.no_grad():
        ut = net(xt)
    d["x_test"] = xt.double().numpy()                 # (200, 4)
    d["u_test"] = ut.double().numpy().reshape(-1, 1)  # (200, 1)
    sio.savemat(str(path), d, oned_as="column")


def export_onnx(net, path):
    try:
        net.eval()
        torch.onnx.export(net, torch.zeros(1, 4), str(path), input_names=["x"],
                          output_names=["u"], dynamic_axes={"x": {0: "batch"}, "u": {0: "batch"}},
                          dynamo=False)
    except Exception as e:  # ONNX es opcional
        print(f"[aviso] no se exportó ONNX ({type(e).__name__}: {e})")


if __name__ == "__main__":
    from policy import NNController
    from qube_model import QubeParams

    ck = torch.load(sys.argv[1], weights_only=False)
    meta = ck["meta"]
    net = NNController(hidden=tuple(meta["hidden"]), vmax=meta["params"]["Vmax"])
    net.load_state_dict(ck["state_dict"])
    pp = {k: meta["params"][k] for k in QubeParams.__dataclass_fields__}
    p = QubeParams(**pp)
    out = Path(sys.argv[1]).with_suffix(".mat")
    export_mat(net, p, np.array(meta["K_lqr"]), out)
    export_onnx(net, out.with_suffix(".onnx"))
    print(f"Exportado {out}")
