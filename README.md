# Control neuronal del péndulo invertido – Quanser Qube-Servo 3

Entrenamiento en **Python (PyTorch)** de una red neuronal que balancea el
péndulo del Qube-Servo 3 usando el **modelo no lineal** de la planta, y
simulación/validación en **MATLAB/Simulink**.

```
python/                         ENTRENAMIENTO
  qube_model.py                 modelo no lineal (PyTorch, por lotes) + sensores
  policy.py                     red neuronal (MLP 4-64-64-1, impar)
  sim.py                        simulación en lazo cerrado, criterio de éxito
  train.py                      etapa 1 (imitación+DAgger) + etapa 2 (BPTT)
  evaluate.py                   métricas y figuras en Python
  export_to_matlab.py           .pt -> .mat (+ .onnx opcional)
matlab/                         SIMULACIÓN / PROCESAMIENTO
  qube_params.m                 parámetros (mismos de ProyectoControlFinal.m) + LQR
  qube_nonlinear_dynamics.m     modelo no lineal (vectorizado)
  nn_controller.m               forward de la red (scripts)
  nn_controller_block.m         código del bloque MATLAB Function (codegen/QUARC)
  simular_lazo_cerrado.m        RK4 @ 2 ms, sensor ideal o encoder+filtro
  main_simulacion_nn.m          NN vs LQR: escenarios, Monte Carlo, región de atracción
  construir_modelo_simulink.m   genera qube_nn_nonlinear_sim.slx y lo simula
  cargar_variables_nn.m         variables del workspace para Simulink
  planta_block.m, tasa_exito.m  auxiliares
models/
  nn_qube_policy.pt / .mat / .onnx / .json   red ya entrenada
  figuras/                      resultados
```

## 1. Modelo no lineal

Estado `x = [θ, α, θ̇, α̇]` (α = 0 péndulo arriba), entrada `u = Vm` (V).
Mismos parámetros y convenciones que `ProyectoControlFinal.m` y que tu Simulink
(`qs3_lqr_ctrl_simulinkFInal.slx`), por lo que la red reemplaza directamente
a la ganancia `u = -K*x`.

```
(Jr + mp r² + Jp sin²α) θ̈ − mp l r cosα α̈ + 2 Jp sinα cosα θ̇ α̇ + mp l r sinα α̇² = τ − br θ̇
−mp l r cosα θ̈ + Jp α̈ − Jp sinα cosα θ̇² − mp g l sinα = −bp α̇
τ = km (Vm − km θ̇) / Rm
```

`Jr = mr r²/3`, `Jp = mp Lp²/3` (respecto al pivote), `l = Lp/2`. Se desprecia la
inductancia (L/R = 1.15 mH / 7.5 Ω = 0.15 ms ≪ Ts = 2 ms).
**Verificación:** el Jacobiano numérico del modelo no lineal en x = 0 coincide
con tus matrices A y B con error 1e-10, y Python y MATLAB producen la misma
trayectoria hasta 8 cifras.

## 2. Entrenamiento (Python)

| Etapa | Qué hace | Por qué |
|---|---|---|
| 1. Imitación + DAgger | La red imita al LQR saturado `sat(−Kx)` en estados que la propia red visita en la planta no lineal | Inicialización rápida y estable |
| 2. BPTT | Se simula en paralelo la planta no lineal y se minimiza directamente `Σ(xᵀQx + Ru² + R_d Δu²)Ts + x_Tᵀ P x_T` derivando a través de la simulación | La red aprende a manejar no linealidades y saturación, donde el LQR ya no es óptimo |

Robustez para el paso a hardware (domain randomization): parámetros físicos
±15 %, encoder cuantizado (2048 cuentas/rev), velocidades por derivada filtrada
`50s/(s+50)` y ruido. `Q = diag(1, 20, 0.1, 0.1)`, `R = 10` (los de tu LQR);
`P` es la solución de Riccati y sirve como costo terminal.

La red es **impar** por construcción, `u(−x) = −u(x)` (simetría física del
péndulo), lo que garantiza `u(0) = 0`. Sin esto, un sesgo de solo 0.1 V movía
el brazo unos 18° en estado estacionario, porque `K_θ ≈ −0.32 V/rad`.

```bash
cd python
pip install -r requirements.txt
python train.py            # completo (~20-40 min en CPU); --quick para prueba
python evaluate.py         # métricas + figuras en models/figuras
```

## 3. Simulación (MATLAB)

```matlab
cd matlab
main_simulacion_nn          % NN vs LQR en el modelo no lineal (sin toolboxes)
construir_modelo_simulink   % crea y simula qube_nn_nonlinear_sim.slx
```

`main_simulacion_nn` primero comprueba que el forward en MATLAB da lo mismo que
PyTorch, y luego compara NN y LQR en: α₀ = 10°, α₀ = 25° con encoders,
perturbación, Monte Carlo (±15 % de parámetros) y región de atracción.

En el Simulink generado: `usar_nn = 1/0` (NN/LQR), `usar_sensor_real = 1/0`,
`x0_sim = [θ; α; θ̇; α̇]`.

## 4. Siguiente paso: hardware

El bloque **"Controlador NN"** (MATLAB Function, `nn_controller_block.m`) usa
solo operaciones matriciales, así que es compatible con QUARC. Cópialo a
`qs3_lqr_ctrl_simulinkFInal.slx` en lugar de la ganancia `u = -K*x`, ejecuta
`cargar_variables_nn` antes de compilar y conserva tu switch de habilitación
(|α| ≤ 15°).

> `Vmax = 10 V` en el entrenamiento. Si tu amplificador o tu modelo usa otro
> límite, reentrena con `python train.py --vmax <V>`.
