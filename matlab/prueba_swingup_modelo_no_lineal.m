%% PRUEBA_SWINGUP_MODELO_NO_LINEAL
% Prueba, solo con MATLAB (sin Simulink ni QUARC), de la red DQN que LEVANTA
% el péndulo y lo mantiene arriba. Usa el mismo código del bloque de Simulink.
%
%   1) Verifica que el bloque reproduce los voltajes calculados en Python
%   2) Swing-up desde colgando (modelo no lineal + encoders + observador)
%   3) Monte Carlo: parámetros físicos +-15 % y ruido
%
% Necesita en la misma carpeta (o en ../models): dqn_swingup_qube.mat

clc; close all;
aqui_ = fileparts(mfilename('fullpath'));
addpath(aqui_);
mat_ = '';
cand_ = {fullfile(aqui_, 'dqn_swingup_qube.mat'), fullfile(aqui_, '..', 'models', 'dqn_swingup_qube.mat'), ...
         fullfile(pwd, 'dqn_swingup_qube.mat'), which('dqn_swingup_qube.mat')};
for k_ = 1:numel(cand_)
    if isempty(mat_) && ~isempty(cand_{k_}) && exist(cand_{k_}, 'file') == 2, mat_ = cand_{k_}; end
end
if isempty(mat_)
    error('No encuentro dqn_swingup_qube.mat. Ponlo en la misma carpeta que este script (%s).', aqui_);
end
fprintf('Red: %s\n', mat_);
S = load(mat_);
red_swingup = S.red_swingup;
p   = qube_params();
d2r = pi/180;

%% 1) Verificación contra Python (misma secuencia de mediciones -> mismos voltajes)
clear dqn_swingup_block
n_ = size(S.x_test, 1);
u_ = zeros(n_, 1);
for k_ = 1:n_
    u_(k_) = dqn_swingup_block(S.x_test(k_, :)', red_swingup);
end
dif_ = sum(abs(u_ - S.u_test(:)) > 1e-9);
fprintf('Verificación vs Python: %d de %d pasos con voltaje distinto (%.2f %%)\n', dif_, n_, 100*dif_/n_);

%% 2) Swing-up desde colgando
s = simular_swingup(red_swingup, p, [0; -pi; 0; 0], 10);
fprintf('Swing-up nominal: éxito=%d  tiempo de subida=%.2f s  |theta|max=%.0f°  |alpha| final=%.2f°\n', ...
        s.exito, s.t_subida, max(abs(s.X(1,:)))/d2r, abs(s.alpha_env(end))/d2r);
fig = figure('Name', 'Swing-up DQN (modelo no lineal)', 'Position', [100 100 900 700]);
subplot(3,1,1); plot(s.t, s.alpha_env/d2r, 'b'); ylabel('\alpha (deg)'); grid on;
title('Swing-up + balance con la red DQN (0° = arriba, \pm180° = colgando)');
subplot(3,1,2); plot(s.t, s.X(1,:)/d2r, 'b'); ylabel('\theta (deg)'); grid on;
subplot(3,1,3); stairs(s.t(1:end-1), s.U, 'b'); ylabel('V_m (V)'); xlabel('t (s)'); grid on;
try
    print(fig, fullfile(fileparts(mat_), 'swingup_dqn.png'), '-dpng', '-r110');
catch
end

%% 3) Monte Carlo
rng(1);
N = 40;
ok_ = false(1, N);  ts_ = nan(1, N);  thm_ = zeros(1, N);
for i_ = 1:N
    pm = p;
    for f_ = {'Rm', 'km', 'mr', 'r', 'br', 'mp', 'Lp', 'bp'}
        pm.(f_{1}) = p.(f_{1})*(1 + 0.15*(2*rand - 1));
    end
    x0 = [0.2*(2*rand-1); -pi + 0.1*(2*rand-1); 0; 0.5*(2*rand-1)];
    si = simular_swingup(red_swingup, pm, x0, 8, 0.05);
    ok_(i_) = si.exito;  ts_(i_) = si.t_subida;  thm_(i_) = max(abs(si.X(1,:)))/d2r;
end
fprintf('Monte Carlo (N=%d, parámetros +-15 %%, ruido): éxito = %.0f %%,  |theta| máx = %.0f°\n', ...
        N, 100*mean(ok_), max(thm_));
if any(ok_)
    fprintf('   tiempo de subida: medio %.2f s, máximo %.2f s\n', mean(ts_(ok_)), max(ts_(ok_)));
end
clear aqui_ mat_ cand_ k_ n_ u_ dif_ i_ f_
