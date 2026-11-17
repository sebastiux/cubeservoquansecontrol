%% PRUEBA_SWINGUP_MODELO_NO_LINEAL
% Prueba, solo con MATLAB (sin Simulink ni QUARC), de la red DQN que LEVANTA
% el péndulo y lo mantiene arriba. Usa el mismo código del bloque de Simulink.
%
%   1) Verifica que el bloque reproduce los voltajes calculados en Python
%   2) Swing-up desde colgando (modelo no lineal + encoders + observador)
%   3) Monte Carlo: parámetros físicos +-15 % y ruido
%
% Necesita en la misma carpeta (o en ../models): dqn_swingup_v2_qube.mat

clc; close all;
aqui_ = fileparts(mfilename('fullpath'));
addpath(aqui_);
mat_ = '';
cand_ = {fullfile(aqui_, 'dqn_swingup_v2_qube.mat'), fullfile(aqui_, '..', 'models', 'dqn_swingup_v2_qube.mat'), ...
         fullfile(pwd, 'dqn_swingup_v2_qube.mat'), which('dqn_swingup_v2_qube.mat')};
for k_ = 1:numel(cand_)
    if isempty(mat_) && ~isempty(cand_{k_}) && exist(cand_{k_}, 'file') == 2, mat_ = cand_{k_}; end
end
if isempty(mat_)
    error('No encuentro dqn_swingup_v2_qube.mat. Ponlo en la misma carpeta que este script (%s).', aqui_);
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
dif_ = sum(abs(u_ - S.u_test(:)) > 1e-6);
fprintf('Verificación vs Python: %d de %d pasos con voltaje distinto (%.2f %%)\n', dif_, n_, 100*dif_/n_);

%% 2) Swing-up desde colgando (Qube ideal y Qube "difícil")
% El caso "brazo ligero" imita al Qube real: inercia del brazo 4 veces menor que
% mr*r^2/3 y 4 ms de retardo (con eso la red anterior temblaba como en el hardware).
casos = {'modelo ideal', struct(), 1; 'brazo ligero (inercia/4) + 4 ms', struct('retardo', 2), 0.25};
fig = figure('Name', 'Swing-up DQN (modelo no lineal)', 'Position', [100 100 900 700]);
est = {'b', 'r'};
for c_ = 1:size(casos, 1)
    pc = p;  pc.mr = p.mr*casos{c_, 3};
    s = simular_swingup(red_swingup, pc, [0; -pi; 0; 0], 10, casos{c_, 2});
    fprintf('Swing-up %-34s éxito=%d  subida=%.2f s  |theta|max=%.0f°  temblor=%.2f° RMS\n', ...
            casos{c_, 1}, s.exito, s.t_subida, max(abs(s.X(1,:)))/d2r, s.temblor);
    subplot(3,1,1); hold on; plot(s.t, s.alpha_env/d2r, est{c_});
    subplot(3,1,2); hold on; plot(s.t, s.X(1,:)/d2r, est{c_});
    subplot(3,1,3); hold on; stairs(s.t(1:end-1), s.U, est{c_});
end
subplot(3,1,1); ylabel('\alpha (deg)'); grid on; legend(casos(:,1));
title('Swing-up + balance con la red DQN (0° = arriba, \pm180° = colgando)');
subplot(3,1,2); ylabel('\theta (deg)'); grid on;
subplot(3,1,3); ylabel('V_m (V)'); xlabel('t (s)'); grid on;
try
    dir_fig_ = fullfile(fileparts(mat_), 'figuras');
    if ~exist(dir_fig_, 'dir'), dir_fig_ = fileparts(mat_); end
    print(fig, fullfile(dir_fig_, 'swingup_dqn.png'), '-dpng', '-r110');
catch
end

%% 3) Monte Carlo: Qube aleatorio (inercia del brazo, ganancia, retardo, fricción)
rng(1);
N = 40;
ok_ = false(1, N);  ts_ = nan(1, N);  thm_ = zeros(1, N);  tb_ = nan(1, N);
for i_ = 1:N
    pm = p;
    for f_ = {'Rm', 'km', 'r', 'mp', 'Lp'}
        pm.(f_{1}) = p.(f_{1})*(1 + 0.15*(2*rand - 1));
    end
    pm.mr = p.mr*(0.2 + 0.9*rand);                   % inercia del brazo x0.2 a x1.1
    pm.br = p.br*(0.5 + 2.5*rand);  pm.bp = p.bp*(0.5 + 2.5*rand);
    pm.tau_c = 3e-3*rand;
    opc = struct('retardo', randi([0 4]), 'k_u', 0.8 + 0.45*rand, 'offset', (2*rand - 1)*d2r);
    x0 = [0.2*(2*rand-1); -pi + 0.1*(2*rand-1); 0; 0.5*(2*rand-1)];
    si = simular_swingup(red_swingup, pm, x0, 8, opc);
    ok_(i_) = si.exito;  ts_(i_) = si.t_subida;  thm_(i_) = max(abs(si.X(1,:)))/d2r;  tb_(i_) = si.temblor;
end
fprintf('Monte Carlo (N=%d, Qube aleatorio): éxito = %.0f %%,  |theta| máx = %.0f°\n', N, 100*mean(ok_), max(thm_));
if any(ok_)
    fprintf('   subida: media %.2f s, máxima %.2f s.   temblor de alpha: medio %.2f° RMS, máximo %.2f°\n', ...
            mean(ts_(ok_)), max(ts_(ok_)), mean(tb_(ok_)), max(tb_(ok_)));
end
clear aqui_ mat_ cand_ k_ n_ u_ dif_ i_ f_ c_ dir_fig_
