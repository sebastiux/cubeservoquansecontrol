%% CARGAR_CONTROLADOR_RL
% Carga en el workspace los controladores entrenados en Python:
%   red_dqn  : struct para el bloque "Controlador DQN"
%   tabla_q  : struct para el bloque "Controlador Q-learning"
%   Ts_agente: periodo de decisión del agente (s)
%   usar_rl  : 1 = controlador RL, 0 = LQR original  (se puede cambiar en
%              External Mode mientras corre el Qube)
% El modelo qs3_rl_ctrl.slx lo ejecuta en su InitFcn.

rl_dir_ = fullfile(fileparts(mfilename('fullpath')), '..', 'models');

if exist(fullfile(rl_dir_, 'dqn_qube.mat'), 'file')
    d_ = load(fullfile(rl_dir_, 'dqn_qube.mat'));
    red_dqn = struct('W1', d_.W1, 'b1', d_.b1, 'W2', d_.W2, 'b2', d_.b2, ...
                     'W3', d_.W3, 'b3', d_.b3, 'x_scale', d_.x_scale, ...
                     'acciones', d_.acciones);
    Ts_agente = d_.Ts_agente;
end
if exist(fullfile(rl_dir_, 'qlearning_qube.mat'), 'file')
    d_ = load(fullfile(rl_dir_, 'qlearning_qube.mat'));
    tabla_q = struct('Q', d_.Q, 'acciones', d_.acciones, ...
                     'bordes_theta', d_.bordes_theta, 'bordes_alpha', d_.bordes_alpha, ...
                     'bordes_theta_d', d_.bordes_theta_d, 'bordes_alpha_d', d_.bordes_alpha_d, ...
                     'n_bins', d_.n_bins);
    Ts_agente = d_.Ts_agente;
end
if ~exist('usar_rl', 'var'), usar_rl = 1; end
clear rl_dir_ d_
