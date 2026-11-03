%% MAIN_SIMULACION_NN
% Simulación y validación en MATLAB del controlador neuronal entrenado en
% Python (python/train.py) sobre el MODELO NO LINEAL del Qube-Servo 3.
%
% Pasos:
%   1) Carga parámetros, LQR y la red (models/nn_qube_policy.mat)
%   2) Verifica que el forward en MATLAB == PyTorch
%   3) Escenarios: alpha0 = 10°, alpha0 = 25° y perturbación (NN vs LQR)
%   4) Monte Carlo con parámetros ±15 % y sensores reales
%   5) Región de atracción (alpha0 vs alpha_dot0) NN vs LQR
%
% Funciona en MATLAB (sin toolboxes obligatorios) y en GNU Octave.

clc; clear; close all;

here = fileparts(mfilename('fullpath'));
addpath(here);
mat_file = fullfile(here, '..', 'models', 'nn_qube_policy.mat');
fig_dir  = fullfile(here, '..', 'models', 'figuras');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% 1) Parámetros, LQR y red
p   = qube_params();
net = load(mat_file);
p.Vmax = net.Vmax;
fprintf('K LQR (MATLAB)  = [%s]\n', sprintf(' %.4f', p.K));
fprintf('K LQR (Python)  = [%s]\n', sprintf(' %.4f', net.K_lqr));

ctrl_nn  = @(y) nn_controller(y, net);
ctrl_lqr = @(y) -p.K*y;

%% 2) Verificación del forward (MATLAB vs PyTorch)
if isfield(net, 'x_test')
    u_m = nn_controller(net.x_test', net);
    err = max(abs(u_m(:) - net.u_test(:)));
    fprintf('Forward MATLAB vs PyTorch: error máx = %.2e V\n', err);
    assert(err < 1e-4, 'El forward en MATLAB no coincide con PyTorch');
    % bloque de Simulink (misma red)
    u_b = zeros(1, 20);
    for i = 1:20
        u_b(i) = nn_controller_block(net.x_test(i,:)', net.W1, net.b1, net.W2, ...
                                     net.b2, net.W3, net.b3, net.x_scale, net.Vmax);
    end
    fprintf('Bloque Simulink vs PyTorch: error máx = %.2e V\n', ...
            max(abs(u_b(:) - net.u_test(1:20))));
end

%% 3) Escenarios individuales
d2r = pi/180;
escenarios = {
    'alpha_0 = 10°, sensor ideal',            [0; 10*d2r; 0; 0],  struct('sensor','ideal')
    'alpha_0 = 25°, encoders + filtro',       [0; 25*d2r; 0; 0],  struct('sensor','real')
    'Perturbación en t = 2 s (encoders)',     [0; 2*d2r;  0; 0],  struct('sensor','real','dist_t',2,'dist_v',3.0)
};
T = 6;
for e = 1:size(escenarios, 1)
    [titulo, x0, opts] = escenarios{e, :};
    s_nn  = simular_lazo_cerrado(ctrl_nn,  x0, p, T, opts);
    s_lqr = simular_lazo_cerrado(ctrl_lqr, x0, p, T, opts);
    fprintf('%-40s  éxito NN=%d  LQR=%d\n', titulo, ...
            tasa_exito(s_nn.X), tasa_exito(s_lqr.X));

    fig = figure('Name', titulo, 'Position', [100 100 900 700]);
    subplot(3,1,1);
    plot(s_lqr.t, squeeze(s_lqr.X(2,1,:))/d2r, 'r--', s_nn.t, squeeze(s_nn.X(2,1,:))/d2r, 'b', 'LineWidth', 1.3);
    ylabel('\alpha (deg)'); grid on; legend('LQR', 'NN'); title(titulo);
    subplot(3,1,2);
    plot(s_lqr.t, squeeze(s_lqr.X(1,1,:))/d2r, 'r--', s_nn.t, squeeze(s_nn.X(1,1,:))/d2r, 'b', 'LineWidth', 1.3);
    ylabel('\theta (deg)'); grid on;
    subplot(3,1,3);
    plot(s_lqr.t(1:end-1), squeeze(s_lqr.U(1,1,:)), 'r--', s_nn.t(1:end-1), squeeze(s_nn.U(1,1,:)), 'b', 'LineWidth', 1.0);
    ylabel('V_m (V)'); xlabel('t (s)'); grid on;
    print(fig, fullfile(fig_dir, sprintf('escenario_%d.png', e)), '-dpng', '-r110');
end

%% 4) Monte Carlo: parámetros ±15 %, estados iniciales aleatorios, sensores reales
rng(1);
N = 500;
pm = p;
for f = {'Rm','km','mr','r','br','mp','Lp','bp'}
    pm.(f{1}) = p.(f{1}) * (1 + 0.15*(2*rand(1, N) - 1));
end
x0 = [ (2*rand(1,N)-1)*45*d2r;  (2*rand(1,N)-1)*25*d2r;
       (2*rand(1,N)-1)*2;       (2*rand(1,N)-1)*4 ];
opts = struct('sensor', 'real', 'noise', 0.05);
mc_nn  = simular_lazo_cerrado(ctrl_nn,  x0, pm, 4, opts);
mc_lqr = simular_lazo_cerrado(ctrl_lqr, x0, pm, 4, opts);
ok_nn = tasa_exito(mc_nn.X);  ok_lqr = tasa_exito(mc_lqr.X);
fprintf('\nMonte Carlo (N=%d, ±15%% parámetros, encoders, ruido):\n', N);
fprintf('   éxito NN  = %5.1f %%\n   éxito LQR = %5.1f %%\n', 100*mean(ok_nn), 100*mean(ok_lqr));

%% 5) Región de atracción: alpha0 vs alpha_dot0 (theta0 = theta_dot0 = 0)
a0  = linspace(-40, 40, 41)*d2r;
ad0 = linspace(-10, 10, 41);
[AA, AD] = meshgrid(a0, ad0);
x0 = [zeros(1, numel(AA)); AA(:)'; zeros(1, numel(AA)); AD(:)'];
roa_nn  = reshape(tasa_exito(simular_lazo_cerrado(ctrl_nn,  x0, p, 4).X), size(AA));
roa_lqr = reshape(tasa_exito(simular_lazo_cerrado(ctrl_lqr, x0, p, 4).X), size(AA));
fprintf('Región de atracción (área relativa): NN = %.1f %%, LQR = %.1f %%\n', ...
        100*mean(roa_nn(:)), 100*mean(roa_lqr(:)));

fig = figure('Name', 'Región de atracción', 'Position', [100 100 1000 420]);
subplot(1,2,1); imagesc(a0/d2r, ad0, roa_lqr); axis xy; colormap(gray);
xlabel('\alpha_0 (deg)'); ylabel('\alpha''_0 (rad/s)'); title(sprintf('LQR (%.0f %%)', 100*mean(roa_lqr(:))));
subplot(1,2,2); imagesc(a0/d2r, ad0, roa_nn);  axis xy; colormap(gray);
xlabel('\alpha_0 (deg)'); title(sprintf('Red neuronal (%.0f %%)', 100*mean(roa_nn(:))));
print(fig, fullfile(fig_dir, 'region_atraccion.png'), '-dpng', '-r110');

fprintf('\nFiguras guardadas en %s\n', fig_dir);
