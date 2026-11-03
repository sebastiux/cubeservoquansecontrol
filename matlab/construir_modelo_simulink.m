function construir_modelo_simulink()
%CONSTRUIR_MODELO_SIMULINK Crea qube_nn_nonlinear_sim.slx y lo simula.
%
%   Diagrama:
%     Planta no lineal (MATLAB Function + Integrator, ode4 @ 2 ms)
%        -> Sensor ideal (ZOH)  ó  encoder cuantizado + derivada filtrada
%        -> Controlador NN (MATLAB Function, parámetros W1..b3)  ó  LQR -K*x
%        -> Saturación ±Vmax -> Planta
%
%   Selectores en el workspace (antes de simular):
%     usar_nn = 1 (NN) | 0 (LQR)
%     usar_sensor_real = 1 (encoders + filtro) | 0 (estado ideal)
%     x0_sim = [theta; alpha; theta_dot; alpha_dot] inicial
%
%   El bloque "Controlador NN" se puede copiar tal cual a tu modelo de
%   hardware (qs3_lqr_ctrl_simulinkFInal.slx) en lugar de la ganancia
%   "u = -K*x": misma entrada X (4x1), misma salida u (V).

here = fileparts(mfilename('fullpath'));
addpath(here);
mdl = 'qube_nn_nonlinear_sim';

evalin('base', 'cargar_variables_nn');

if bdIsLoaded(mdl), close_system(mdl, 0); end
if exist(fullfile(here, [mdl '.slx']), 'file'), delete(fullfile(here, [mdl '.slx'])); end
new_system(mdl);
open_system(mdl);
set_param(mdl, 'Solver', 'ode4', 'FixedStep', 'Ts', 'StopTime', '8', ...
          'InitFcn', 'cargar_variables_nn', 'ReturnWorkspaceOutputs', 'on');

% ------------------------------------------------------------- bloques
blk = @(name) [mdl '/' name];

% Planta no lineal
add_block('simulink/User-Defined Functions/MATLAB Function', blk('Planta no lineal'), ...
          'Position', [520 180 640 240]);
set_mfunction(blk('Planta no lineal'), fullfile(here, 'planta_block.m'), {'qp'});
add_block('simulink/Continuous/Integrator', blk('x'), ...
          'InitialCondition', 'x0_sim', 'Position', [690 195 720 225]);

% Sensor ideal
add_block('simulink/Discrete/Zero-Order Hold', blk('ZOH ideal'), ...
          'SampleTime', 'Ts', 'Position', [790 90 830 130]);

% Sensor "real": ángulos -> cuantización encoder -> derivada filtrada
add_block('simulink/Math Operations/Gain', blk('Selecciona angulos'), ...
          'Gain', '[eye(2) zeros(2)]', 'Multiplication', 'Matrix(K*u)', ...
          'Position', [780 260 840 300]);
add_block('simulink/Discrete/Zero-Order Hold', blk('ZOH enc'), ...
          'SampleTime', 'Ts', 'Position', [870 260 910 300]);
add_block('simulink/Discontinuities/Quantizer', blk('Encoder 2048 cuentas'), ...
          'QuantizationInterval', 'q_enc', 'Position', [940 260 990 300]);
% Derivada filtrada 50s/(s+50) discretizada (Euler hacia atrás), 2 canales
add_block('simulink/Discrete/Discrete State-Space', blk('Derivada filtrada'), ...
          'A', 'A_f', 'B', 'B_f', 'C', 'C_f', 'D', 'D_f', 'InitialCondition', 'IC_f', ...
          'SampleTime', 'Ts', 'Position', [1030 320 1110 360]);
add_block('simulink/Signal Routing/Mux', blk('Mux sensor'), 'Inputs', '2', ...
          'Position', [1150 260 1155 340]);

add_block('simulink/Sources/Constant', blk('usar_sensor_real'), 'Value', 'usar_sensor_real', ...
          'Position', [1150 180 1200 200]);
add_block('simulink/Signal Routing/Switch', blk('Sensor'), 'Criteria', 'u2 ~= 0', ...
          'Position', [1240 150 1280 300]);

% Controladores
add_block('simulink/User-Defined Functions/MATLAB Function', blk('Controlador NN'), ...
          'Position', [1340 110 1460 170]);
set_mfunction(blk('Controlador NN'), fullfile(here, 'nn_controller_block.m'), ...
              {'W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'x_scale', 'Vmax'});
add_block('simulink/Math Operations/Gain', blk('u = -K*x'), 'Gain', '-K', ...
          'Multiplication', 'Matrix(K*u)', 'Position', [1360 260 1440 300]);
add_block('simulink/Sources/Constant', blk('usar_nn'), 'Value', 'usar_nn', ...
          'Position', [1490 195 1530 215]);
add_block('simulink/Signal Routing/Switch', blk('Controlador'), 'Criteria', 'u2 ~= 0', ...
          'Position', [1570 130 1610 290]);
add_block('simulink/Discontinuities/Saturation', blk('Saturacion'), ...
          'UpperLimit', 'Vmax', 'LowerLimit', '-Vmax', 'Position', [1650 195 1690 225]);

% Registro y visualización
add_block('simulink/Math Operations/Gain', blk('rad2deg'), 'Gain', '180/pi', ...
          'Position', [790 400 830 430]);
add_block('simulink/Sinks/Scope', blk('Estados en grados'), 'Position', [880 395 910 435]);
add_block('simulink/Sinks/Scope', blk('Vm (V)'), 'Position', [1740 120 1770 150]);
add_block('simulink/Sinks/To Workspace', blk('x_log'), 'VariableName', 'x_log', ...
          'SaveFormat', 'Timeseries', 'Position', [790 470 850 500]);
add_block('simulink/Sinks/To Workspace', blk('u_log'), 'VariableName', 'u_log', ...
          'SaveFormat', 'Timeseries', 'Position', [1740 260 1800 290]);

% ------------------------------------------------------------- líneas
L = @(a, b) add_line(mdl, a, b, 'autorouting', 'on');
L('Planta no lineal/1', 'x/1');
L('x/1', 'ZOH ideal/1');
L('x/1', 'Selecciona angulos/1');
L('x/1', 'rad2deg/1');
L('x/1', 'x_log/1');
L('rad2deg/1', 'Estados en grados/1');
L('Selecciona angulos/1', 'ZOH enc/1');
L('ZOH enc/1', 'Encoder 2048 cuentas/1');
L('Encoder 2048 cuentas/1', 'Mux sensor/1');
L('Encoder 2048 cuentas/1', 'Derivada filtrada/1');
L('Derivada filtrada/1', 'Mux sensor/2');
L('Mux sensor/1', 'Sensor/1');
L('usar_sensor_real/1', 'Sensor/2');
L('ZOH ideal/1', 'Sensor/3');
L('Sensor/1', 'Controlador NN/1');
L('Sensor/1', 'u = -K*x/1');
L('Controlador NN/1', 'Controlador/1');
L('usar_nn/1', 'Controlador/2');
L('u = -K*x/1', 'Controlador/3');
L('Controlador/1', 'Saturacion/1');
L('Saturacion/1', 'Planta no lineal/2');
L('Saturacion/1', 'Vm (V)/1');
L('Saturacion/1', 'u_log/1');

save_system(mdl, fullfile(here, [mdl '.slx']));
fprintf('Modelo creado: %s\n', fullfile(here, [mdl '.slx']));

% ------------------------------------------------------------- simulación NN vs LQR
assignin('base', 'usar_sensor_real', 1);
res = struct();
for usar = [1 0]
    assignin('base', 'usar_nn', usar);
    out = sim(mdl);
    res(usar + 1).t = out.x_log.Time;
    res(usar + 1).x = squeeze(out.x_log.Data);
    res(usar + 1).tu = out.u_log.Time;
    res(usar + 1).u = squeeze(out.u_log.Data);
end
assignin('base', 'usar_nn', 1);

fig = figure('Name', 'Simulink: NN vs LQR (planta no lineal)');
nm = {'\theta (deg)', '\alpha (deg)'};
for i = 1:2
    subplot(3, 1, i);
    plot(res(1).t, fix_dims(res(1).x, i)*180/pi, 'r--', ...
         res(2).t, fix_dims(res(2).x, i)*180/pi, 'b', 'LineWidth', 1.2);
    ylabel(nm{i}); grid on;
    if i == 1, legend('LQR', 'NN'); title('qube\_nn\_nonlinear\_sim'); end
end
subplot(3, 1, 3);
plot(res(1).tu, res(1).u, 'r--', res(2).tu, res(2).u, 'b');
ylabel('V_m (V)'); xlabel('t (s)'); grid on;
end

% =====================================================================
function set_mfunction(block_path, src_file, param_names)
%SET_MFUNCTION Pone el código de src_file en un bloque MATLAB Function y
%  declara param_names como parámetros (se leen del workspace).
rt = sfroot;
ch = rt.find('-isa', 'Stateflow.EMChart', 'Path', block_path);
ch.Script = fileread(src_file);
for k = 1:numel(param_names)
    d = ch.find('-isa', 'Stateflow.Data', 'Name', param_names{k});
    d.Scope = 'Parameter';
end
end

function v = fix_dims(x, i)
% x puede venir como 4xN o Nx4 según la versión de Simulink
if size(x, 1) == 4, v = x(i, :); else, v = x(:, i); end
end
