%% CARGAR_RED_DQN
% Carga en el workspace la red DQN entrenada en Python (models/dqn_qube.mat):
%   red_dqn   : struct que usa el bloque "Controlador DQN" (parámetro)
%   Ts_agente : periodo de decisión de la red (s) -> ZOH antes del bloque
% y, si no existen, las ganancias de TUS observadores (mismas fórmulas que
% ProyectoControlFinal.m, a = 100):  beta, L, m
%
% Lo ejecutan solos (InitFcn) los modelos qube_dqn_pruebas.slx y
% qs3_dqn_ctrl.slx, así que normalmente no hace falta llamarlo a mano.

dqn_dir_ = fileparts(mfilename('fullpath'));
d_ = load(fullfile(dqn_dir_, '..', 'models', 'dqn_qube.mat'));
red_dqn = struct('W1', d_.W1, 'b1', d_.b1, 'W2', d_.W2, 'b2', d_.b2, ...
                 'W3', d_.W3, 'b3', d_.b3, 'x_scale', d_.x_scale, ...
                 'acciones', d_.acciones);
Ts_agente = d_.Ts_agente;

% Observadores de alta ganancia (subsistema "State X"): polos triples en -a
if ~exist('L', 'var') || ~exist('beta', 'var') || ~exist('m', 'var') ...
        || ~isnumeric(L) || ~isnumeric(beta) || ~isnumeric(m)
    a_obs_ = 100;
    beta = 3*a_obs_;
    L    = 3*a_obs_^2;
    m    = a_obs_^3 - L*beta;
    clear a_obs_
end
clear dqn_dir_ d_
