%% PRUEBA_DQN_MODELO_NO_LINEAL
% Prueba de la red DQN (entrenada en Python) sobre el MODELO NO LINEAL del
% Qube-Servo 3, con encoders cuantizados y TUS observadores de velocidad.
% No necesita Simulink ni QUARC (funciona también en GNU Octave).
%
%   1) Verifica que el bloque de Simulink da la misma acción que Python
%   2) alpha0 = 10°  -> respuesta temporal
%   3) Golpe al péndulo en t = 2 s
%   4) Monte Carlo: 500 casos, parámetros ±15 %, ruido en velocidades

clc; clear; close all;
here = fileparts(mfilename('fullpath'));
addpath(here);
cargar_red_dqn;                                  % red_dqn, Ts_agente
fig_dir = fullfile(here, '..', 'models', 'figuras');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

p   = qube_params();
d2r = pi/180;
ctrl = @(y) politica_dqn(y, red_dqn);

%% 1) Verificación contra Python
m_ = load(fullfile(here, '..', 'models', 'dqn_qube.mat'));
u_vec = politica_dqn(m_.x_test', red_dqn);
u_blk = zeros(1, size(m_.x_test, 1));
for k = 1:numel(u_blk)
    u_blk(k) = dqn_controller_block(m_.x_test(k, :)', red_dqn);
end
fprintf('Verificación vs Python: acciones distintas  vectorizado %d/%d,  bloque Simulink %d/%d\n', ...
        sum(abs(u_vec(:) - m_.u_test(:)) > 1e-9), numel(m_.u_test), ...
        sum(abs(u_blk(:) - m_.u_test(:)) > 1e-9), numel(m_.u_test));

%% 2-3) Escenarios
escenarios = {
    'DQN: alpha_0 = 10 deg',             [0; 10*d2r; 0; 0], struct()
    'DQN: golpe al pendulo en t = 2 s',  [0;  0;     0; 0], struct('dist_t', 2, 'dist_v', 2.5)
};
for e = 1:size(escenarios, 1)
    [titulo, x0, opts] = escenarios{e, :};
    opts.Ts_ctrl = Ts_agente;                    % sensor = observador (defecto)
    s = simular_lazo_cerrado(ctrl, x0, p, 6, opts);
    fprintf('%-34s éxito=%d  |alpha|max=%5.1f°  |theta|max=%5.1f°  |theta|final=%4.1f°\n', titulo, ...
            tasa_exito(s.X), max(abs(s.X(2,1,:)))/d2r, max(abs(s.X(1,1,:)))/d2r, abs(s.X(1,1,end))/d2r);
    fig = figure('Name', titulo, 'Position', [100 100 900 700]);
    subplot(3,1,1); plot(s.t, squeeze(s.X(2,1,:))/d2r, 'b', 'LineWidth', 1.2);
    ylabel('\alpha (deg)'); grid on; title(titulo);
    subplot(3,1,2); plot(s.t, squeeze(s.X(1,1,:))/d2r, 'b', 'LineWidth', 1.2);
    ylabel('\theta (deg)'); grid on;
    subplot(3,1,3); stairs(s.t(1:end-1), squeeze(s.U(1,1,:)), 'b');
    ylabel('V_m (V)'); xlabel('t (s)'); grid on;
    print(fig, fullfile(fig_dir, sprintf('dqn_escenario_%d.png', e)), '-dpng', '-r110');
end

%% 4) Monte Carlo
rng(1);
N = 500;
pm = p;
for f = {'Rm', 'km', 'mr', 'r', 'br', 'mp', 'Lp', 'bp'}
    pm.(f{1}) = p.(f{1}) * (1 + 0.15*(2*rand(1, N) - 1));
end
x0 = [(2*rand(1,N)-1)*30*d2r; (2*rand(1,N)-1)*12*d2r; (2*rand(1,N)-1)*1; (2*rand(1,N)-1)*2];
s = simular_lazo_cerrado(ctrl, x0, pm, 5, struct('noise', 0.05, 'Ts_ctrl', Ts_agente));
fprintf('\nMonte Carlo (N=%d, ±15%% parámetros, encoders + observador, ruido): éxito = %.1f %%\n', ...
        N, 100*mean(tasa_exito(s.X)));
fprintf('Figuras en %s\n', fig_dir);
