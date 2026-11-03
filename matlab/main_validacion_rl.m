%% MAIN_VALIDACION_RL
% Validación en MATLAB, sobre el MODELO NO LINEAL, de los controladores
% entrenados en Python:  Deep Q-Learning (DQN)  y  Q-learning tabular,
% comparados con tu LQR.  Ejecutar ANTES de conectarlos al Simulink/Qube.
%
%   1) Verifica que los bloques de Simulink dan la misma acción que Python
%   2) Respuesta a alpha0 = 10° (encoders + derivada filtrada)
%   3) Perturbación (golpe al péndulo) en t = 2 s
%   4) Monte Carlo: parámetros ±15 %, condiciones iniciales aleatorias
%
% Funciona en MATLAB (sin toolboxes) y en GNU Octave.

clc; clear; close all;
here = fileparts(mfilename('fullpath'));
addpath(here);
cargar_controlador_rl;                          % red_dqn, tabla_q, Ts_agente
fig_dir = fullfile(here, '..', 'models', 'figuras');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

p = qube_params();
d2r = pi/180;

ctrls = {};                                     % {nombre, handle, Ts_ctrl, estilo}
ctrls(end+1, :) = {'LQR', @(y) -p.K*y, p.Ts, 'k--'};
archivos = {'dqn_qube.mat', 'qlearning_qube.mat'};
nombres  = {'DQN', 'Q-learning'};
estilos  = {'b', 'm'};
for i = 1:2
    f = fullfile(here, '..', 'models', archivos{i});
    if ~exist(f, 'file'), continue; end
    m = load(f);
    if i == 1, c = red_dqn; blkfun = @(x) dqn_controller_block(x, red_dqn);
    else,      c = tabla_q; blkfun = @(x) qlearning_controller_block(x, tabla_q); end

    % ---- 1) verificación contra Python
    u_vec = politica_rl(m.x_test', c);
    u_blk = arrayfun(@(k) blkfun(m.x_test(k, :)'), 1:size(m.x_test, 1));
    fprintf('%-10s  acciones distintas vs Python: vectorizado %d/%d, bloque %d/%d\n', ...
            nombres{i}, sum(abs(u_vec(:) - m.u_test(:)) > 1e-9), numel(m.u_test), ...
            sum(abs(u_blk(:) - m.u_test(:)) > 1e-9), numel(m.u_test));
    ctrls(end+1, :) = {nombres{i}, @(y) politica_rl(y, c), m.Ts_agente, estilos{i}}; %#ok<SAGROW>
end
nc = size(ctrls, 1);

%% 2-3) Escenarios
escenarios = {
    'alpha_0 = 10 deg (encoders + filtro)',  [0; 10*d2r; 0; 0], struct('sensor', 'real')
    'Golpe al pendulo en t = 2 s',           [0;  0;     0; 0], struct('sensor', 'real', 'dist_t', 2, 'dist_v', 2.5)
};
for e = 1:size(escenarios, 1)
    [titulo, x0, opts] = escenarios{e, :};
    fig = figure('Name', titulo, 'Position', [100 100 900 700]);
    for c = 1:nc
        opts.Ts_ctrl = ctrls{c, 3};
        s = simular_lazo_cerrado(ctrls{c, 2}, x0, p, 6, opts);
        fprintf('%-38s %-10s  éxito=%d  |alpha|max=%5.1f°  |theta|final=%5.1f°\n', titulo, ctrls{c, 1}, ...
                tasa_exito(s.X), max(abs(s.X(2, 1, :)))/d2r, abs(s.X(1, 1, end))/d2r);
        subplot(3, 1, 1); hold on; plot(s.t, squeeze(s.X(2, 1, :))/d2r, ctrls{c, 4}, 'LineWidth', 1.2);
        subplot(3, 1, 2); hold on; plot(s.t, squeeze(s.X(1, 1, :))/d2r, ctrls{c, 4}, 'LineWidth', 1.2);
        subplot(3, 1, 3); hold on; plot(s.t(1:end-1), squeeze(s.U(1, 1, :)), ctrls{c, 4});
    end
    subplot(3, 1, 1); ylabel('\alpha (deg)'); grid on; title(titulo); legend(ctrls(:, 1));
    subplot(3, 1, 2); ylabel('\theta (deg)'); grid on;
    subplot(3, 1, 3); ylabel('V_m (V)'); xlabel('t (s)'); grid on;
    print(fig, fullfile(fig_dir, sprintf('rl_escenario_%d.png', e)), '-dpng', '-r110');
end

%% 4) Monte Carlo
rng(1);
N = 500;
pm = p;
for f = {'Rm', 'km', 'mr', 'r', 'br', 'mp', 'Lp', 'bp'}
    pm.(f{1}) = p.(f{1}) * (1 + 0.15*(2*rand(1, N) - 1));
end
x0 = [(2*rand(1, N)-1)*30*d2r; (2*rand(1, N)-1)*12*d2r; (2*rand(1, N)-1)*1; (2*rand(1, N)-1)*2];
fprintf('\nMonte Carlo (N=%d, ±15%% parámetros, encoders, ruido 0.05 rad/s):\n', N);
for c = 1:nc
    opts = struct('sensor', 'real', 'noise', 0.05, 'Ts_ctrl', ctrls{c, 3});
    s = simular_lazo_cerrado(ctrls{c, 2}, x0, pm, 5, opts);
    fprintf('   %-10s éxito = %5.1f %%\n', ctrls{c, 1}, 100*mean(tasa_exito(s.X)));
end
fprintf('\nFiguras en %s\n', fig_dir);
