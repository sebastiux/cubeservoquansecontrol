function R = analizar_corrida(varargin)
%ANALIZAR_CORRIDA Guarda en .mat y analiza una corrida del Qube con la red DQN.
%
%   Después de correr el modelo (QUARC -> Monitor & Tune) y detenerlo:
%
%       analizar_corrida                  % usa datos_x y datos_u del workspace,
%                                         % los guarda en corrida_AAAAMMDD_HHMMSS.mat
%       analizar_corrida('corrida_20261117_153000.mat')   % reanaliza un archivo
%       R = analizar_corrida(...)         % devuelve las señales y métricas
%
%   Solo necesita, en la misma carpeta, el .mat de la red (dqn_swingup_*_qube.mat).
%   Es un único archivo: no depende de otros scripts.
%
%   Qué hace
%   1) Guarda la corrida (datos_x, datos_u) en un .mat con fecha y hora.
%   2) Figura "Corrida": alpha, theta y voltaje del motor, con métricas
%      (tiempo de subida, temblor, rizo de voltaje, dV/dt, inversiones).
%   3) Figura "Voltaje": voltaje ordenado por la red antes del filtro
%      (reconstruido repitiendo el bloque sobre los ángulos medidos) frente al
%      voltaje filtrado que llegó al motor, con acercamiento a la subida y al
%      balance.
%   4) Figura "Modelo vs Qube": el modelo no lineal oficial de Quanser con la
%      MISMA red, desde el mismo estado inicial, frente a lo medido.
%   5) Figura "Validación del modelo": se aplica al modelo el voltaje medido,
%      partiendo cada 0.10 s del estado medido; el error de predicción dice
%      cuánto se parece el modelo al Qube real.
%
%   Las figuras se guardan como PNG junto al archivo de la corrida.

% ------------------------------------------------------------------ datos
[t, X, U, archivo] = leer_corrida(varargin{:});
dt = median(diff(t));
d2r = pi/180;
fprintf('Corrida: %s   (%.1f s, %d muestras, paso %.1f ms)\n', archivo, t(end) - t(1), numel(t), dt*1e3);

th = X(:, 1);  al = X(:, 2);                     % alpha envuelto: 0 = arriba
al_des = desenrollar(al);

% ------------------------------------------------------------------ red
[red, mat_red] = cargar_red();
bloque = [];
if ~isempty(red)
    fprintf('Red: %s\n', mat_red);
    bloque = preparar_bloque(mat_red);
end

% ------------------------------------------------------------------ métricas
arriba = abs(al) < 10*d2r;
i_ult = find(~arriba, 1, 'last');
if isempty(i_ult), i_ult = 0; end
M.t_subida = NaN;  M.sostiene = false;
if i_ult < numel(t) - round(2/dt)                % terminó arriba al menos 2 s
    M.sostiene = true;
    M.t_subida = t(min(i_ult + 1, numel(t))) - t(1);
end
cola = false(size(t));
if M.sostiene, cola = t >= max(t(end) - 3, t(1) + M.t_subida + 0.5); end
M.temblor_alpha = NaN;  M.rizo_v = NaN;  M.dvdt = NaN;  M.inversiones = NaN;  M.theta_media = NaN;
if any(cola)
    M.temblor_alpha = std(al(cola))/d2r;
    M.rizo_v = std(U(cola));
    M.dvdt = sqrt(mean((diff(U(cola))/dt).^2));
    s = sign(U(cola));  s(abs(U(cola)) < 0.05) = 0;
    M.inversiones = sum(s(1:end-1).*s(2:end) < 0)/(sum(cola)*dt);
    M.theta_media = mean(th(cola))/d2r;
end
M.theta_max = max(abs(th))/d2r;
M.v_max = max(abs(U));
fprintf(['  sostiene arriba: %d   tiempo de subida: %.2f s   |theta| máx: %.0f°\n' ...
         '  balance (últimos 3 s): temblor alpha %.2f° RMS, rizo de voltaje %.2f V RMS, ' ...
         'dV/dt %.0f V/s, %.1f inversiones/s, theta media %.1f°\n'], ...
        M.sostiene, M.t_subida, M.theta_max, M.temblor_alpha, M.rizo_v, M.dvdt, M.inversiones, M.theta_media);

[carpeta, base] = fileparts(archivo);
if isempty(carpeta), carpeta = pwd; end

% ------------------------------------------------------------------ figura 1
f1 = figure('Name', 'Corrida', 'Position', [60 60 950 720]);
subplot(3,1,1); plot(t, al/d2r, 'b'); grid on; ylabel('\alpha (deg)');
title(sprintf('Corrida %s: subida %.2f s, temblor %.2f° RMS', strrep(base, '_', '\_'), M.t_subida, M.temblor_alpha));
subplot(3,1,2); plot(t, th/d2r, 'b'); grid on; ylabel('\theta (deg)');
subplot(3,1,3); plot(t, U, 'b'); grid on; ylabel('V_m (V)'); xlabel('t (s)');
guardar_png(f1, carpeta, [base '_corrida']);

% ------------------------------------------------------------------ figura 2: voltaje
Ucmd = [];  Ublk = [];
if ~isempty(bloque)
    [Ublk, Ucmd] = repetir_bloque(bloque, red, th, al);
    f2 = figure('Name', 'Voltaje', 'Position', [90 90 950 720]);
    ventanas = {[t(1) t(end)], [t(1) min(t(end), t(1) + 2)], [max(t(1), t(end) - 2) t(end)]};
    titulos  = {'Toda la corrida', 'Subida (primeros 2 s)', 'Balance (últimos 2 s)'};
    for k = 1:3
        subplot(3,1,k); hold on; grid on;
        stairs(t, Ucmd, 'Color', [0.85 0.4 0.1]);
        plot(t, U, 'b', 'LineWidth', 1.2);
        xlim(ventanas{k}); ylabel('V'); title(titulos{k});
        if k == 1, legend('ordenado por la red (antes del filtro)', 'filtrado, al motor'); end
    end
    xlabel('t (s)');
    guardar_png(f2, carpeta, [base '_voltaje']);
    err = max(abs(Ublk - U));
    fprintf('  bloque repetido sobre los ángulos medidos: diferencia máxima con el voltaje registrado %.3f V\n', err);
    if any(cola)
        fprintf('  rizo del voltaje ordenado %.2f V RMS -> filtrado %.2f V RMS\n', std(Ucmd(cola)), M.rizo_v);
    end
end

% ------------------------------------------------------------------ figura 3: modelo vs Qube
p = parametros_oficiales();
Xs = [];
if ~isempty(bloque)
    x0 = [th(1); al_des(1); 0; 0];
    [Xs, Us] = simular_modelo_con_red(bloque, red, p, x0, numel(t), dt);
    al_s = mod(Xs(2,:)' + pi, 2*pi) - pi;
    f3 = figure('Name', 'Modelo vs Qube', 'Position', [120 120 950 720]);
    subplot(3,1,1); plot(t, al/d2r, 'b', t, al_s/d2r, 'r--'); grid on; ylabel('\alpha (deg)');
    legend('Qube real', 'modelo oficial + misma red'); title('Misma red: Qube real frente al modelo no lineal oficial');
    subplot(3,1,2); plot(t, th/d2r, 'b', t, Xs(1,:)'/d2r, 'r--'); grid on; ylabel('\theta (deg)');
    subplot(3,1,3); plot(t, U, 'b', t, Us(:), 'r--'); grid on; ylabel('V_m (V)'); xlabel('t (s)');
    guardar_png(f3, carpeta, [base '_modelo_vs_qube']);
end

% ------------------------------------------------------------------ figura 4: validación del modelo
[th_p, al_p, e_th, e_al] = validar_modelo(p, t, th, al_des, U, dt, 0.10);
M.error_modelo_theta = e_th/d2r;  M.error_modelo_alpha = e_al/d2r;
fprintf('  validación del modelo (predicción a 0.10 s con el voltaje medido): error RMS theta %.2f°, alpha %.2f°\n', ...
        M.error_modelo_theta, M.error_modelo_alpha);
f4 = figure('Name', 'Validación del modelo', 'Position', [150 150 950 560]);
subplot(2,1,1); plot(t, th/d2r, 'b', t, th_p/d2r, 'r.', 'MarkerSize', 3); grid on; ylabel('\theta (deg)');
legend('medido', 'modelo con el voltaje medido');
title(sprintf('Validación del modelo oficial: error RMS \\theta %.2f°, \\alpha %.2f°', M.error_modelo_theta, M.error_modelo_alpha));
subplot(2,1,2); plot(t, al_des/d2r, 'b', t, al_p/d2r, 'r.', 'MarkerSize', 3); grid on;
ylabel('\alpha desenrollado (deg)'); xlabel('t (s)');
guardar_png(f4, carpeta, [base '_validacion_modelo']);

if nargout > 0
    R = struct('t', t, 'theta', th, 'alpha', al, 'alpha_desenrollado', al_des, 'voltaje', U, ...
               'voltaje_ordenado', Ucmd, 'modelo', Xs, 'metricas', M, 'archivo', archivo);
end
end

% =====================================================================
function [t, X, U, archivo] = leer_corrida(varargin)
% Lee datos_x/datos_u del workspace (y los guarda) o de un .mat ya guardado.
if nargin >= 1 && ischar(varargin{1})
    archivo = varargin{1};
    S = load(archivo);
    if ~isfield(S, 'datos_x') || ~isfield(S, 'datos_u')
        error('analizar:archivo', '%s no contiene datos_x y datos_u.', archivo);
    end
    dx = S.datos_x;  du = S.datos_u;
else
    if ~evalin('base', 'exist(''datos_x'', ''var'') && exist(''datos_u'', ''var'')')
        error('analizar:workspace', ['No hay datos_x / datos_u en el workspace.\n' ...
              'Corre el modelo (los bloques To Workspace los crean al detenerlo) o pasa un archivo:\n' ...
              '   analizar_corrida(''corrida_....mat'')']);
    end
    dx = evalin('base', 'datos_x');  du = evalin('base', 'datos_u');
    archivo = fullfile(pwd, ['corrida_' datestr(now, 'yyyymmdd_HHMMSS') '.mat']);
    datos_x = dx;  datos_u = du; %#ok<NASGU>
    save(archivo, 'datos_x', 'datos_u');
    fprintf('Corrida guardada en %s\n', archivo);
end
[tx, X] = separar(dx);
[tu, U] = separar(du);
n = min(numel(tx), numel(tu));
t = tx(1:n);  X = X(1:n, :);  U = U(1:n, 1);
if size(X, 2) < 2, error('analizar:formato', 'datos_x debe tener al menos theta y alpha.'); end
end

function [t, V] = separar(d)
% Acepta "Structure With Time", timeseries o matriz [t, valores].
if isstruct(d) && isfield(d, 'time')
    t = d.time(:);  V = d.signals.values;
elseif isobject(d) && isprop(d, 'Time')
    t = d.Time(:);  V = d.Data;
elseif isnumeric(d)
    t = d(:, 1);  V = d(:, 2:end);
else
    error('analizar:formato', 'Formato de datos no reconocido.');
end
V = squeeze(V);
if size(V, 1) ~= numel(t), V = V.'; end
end

function a = desenrollar(al)
d = diff(al);
d = d - 2*pi*round(d/(2*pi));
a = al(1) + [0; cumsum(d)];
end

function [red, archivo] = cargar_red()
red = [];  archivo = '';
cand = [dir('dqn_swingup*_qube.mat'); dir(fullfile('..', 'models', 'dqn_swingup*_qube.mat'))];
mejor = -1;
for k = 1:numel(cand)
    f = fullfile(cand(k).folder, cand(k).name);
    S = load(f, 'red_swingup');
    v = sscanf(cand(k).name, 'dqn_swingup_v%d');
    if isempty(v), v = 1; end
    if isfield(S, 'red_swingup') && v > mejor
        mejor = v;  red = S.red_swingup;  archivo = f;
    end
end
if isempty(red)
    fprintf(['(No encontré el .mat de la red en esta carpeta: se omiten el voltaje ordenado ' ...
             'y la comparación con el modelo.)\n']);
end
end

function fh = preparar_bloque(mat_red)
% Usa dqn_swingup_block.m si está en el path; si no, lo escribe a partir del
% código guardado dentro del .mat de la red.
fh = [];
S = load(mat_red, 'codigo_bloque');
if isfield(S, 'codigo_bloque')
    codigo = S.codigo_bloque;
    if size(codigo, 1) > 1, codigo = codigo.'; codigo = codigo(:).'; end
    carpeta = fullfile(tempdir, 'bloque_dqn_swingup');
    if ~exist(carpeta, 'dir'), mkdir(carpeta); end
    fid = fopen(fullfile(carpeta, 'dqn_swingup_block.m'), 'w');
    fwrite(fid, codigo);
    fclose(fid);
    addpath(carpeta);                         % primero en el path: versión de ESTA red
    rehash;
end
if exist('dqn_swingup_block', 'file') == 2
    fh = @dqn_swingup_block;
else
    fprintf('(No encontré el código del bloque: se omite la comparación con el modelo.)\n');
end
end

function [Ublk, Ucmd] = repetir_bloque(bloque, red, th, al)
% Repite el bloque sobre los ángulos medidos. Devuelve su salida (filtrada) y
% el voltaje ordenado por la red antes del filtro (invirtiendo el filtro).
clear dqn_swingup_block
n = numel(th);
Ublk = zeros(n, 1);
for k = 1:n
    Ublk(k) = bloque([th(k); al(k); 0; 0], red);
end
Ucmd = Ublk;
if isfield(red, 'filtro_w') && red.filtro_w > 0
    w = red.filtro_w;  z = red.filtro_zeta;  Ts = red.Ts;
    f1 = Ublk;  f2 = [diff(f1); 0]/Ts;              % f1(k+1) = f1(k) + Ts f2(k)
    df2 = [diff(f2); 0]/Ts;                         % f2(k+1) = f2(k) + Ts (w^2 (u - f1) - 2 z w f2)
    Ucmd = f1 + (df2 + 2*z*w*f2)/(w*w);
    Ucmd(end-1:end) = Ucmd(max(end-2, 1));
    if isfield(red, 'acciones') && ~(isfield(red, 'incremental') && red.incremental > 0.5)
        acc = red.acciones(:)';                     % ajustar al nivel de voltaje más cercano
        [~, i] = min(abs(Ucmd - acc), [], 2);
        Ucmd = acc(i)';
    end
end
end

function [X, U] = simular_modelo_con_red(bloque, red, p, x0, n, dt)
% Modelo no lineal oficial + el mismo bloque, desde el mismo estado inicial.
clear dqn_swingup_block
q = 2*pi/p.enc_counts;
x = x0;  X = zeros(4, n);  U = zeros(1, n);
for k = 1:n
    X(:, k) = x;
    y_th = q*round(x(1)/q);
    y_al = mod(q*round(x(2)/q) + pi, 2*pi) - pi;
    u = bloque([y_th; y_al; 0; 0], red);
    U(k) = u;
    x = paso_rk4(x, u, p, dt);
end
end

function [th_p, al_p, e_th, e_al] = validar_modelo(p, t, th, al_des, U, dt, ventana)
% Predicción del modelo con el voltaje medido, reiniciando en el estado medido
% cada "ventana" segundos (las velocidades se estiman derivando los ángulos).
n = numel(t);
vth = derivada_suave(th, dt);  val = derivada_suave(al_des, dt);
th_p = nan(n, 1);  al_p = nan(n, 1);
m = max(2, round(ventana/dt));
for i0 = 1:m:n-1
    x = [th(i0); al_des(i0); vth(i0); val(i0)];
    for k = i0:min(i0 + m - 1, n)
        th_p(k) = x(1);  al_p(k) = x(2);
        x = paso_rk4(x, U(k), p, dt);
    end
end
ok = ~isnan(th_p);
e_th = sqrt(mean((th_p(ok) - th(ok)).^2));
e_al = sqrt(mean((al_p(ok) - al_des(ok)).^2));
end

function v = derivada_suave(y, dt)
% Derivada central sobre +-5 muestras (10 ms a cada lado)
n = numel(y);  k = 5;
v = zeros(n, 1);
i = (1 + k):(n - k);
v(i) = (y(i + k) - y(i - k))/(2*k*dt);
v(1:k) = v(k + 1);  v(n - k + 1:n) = v(n - k);
end

function x = paso_rk4(x, u, p, dt)
k1 = dinamica_oficial(x, u, p);
k2 = dinamica_oficial(x + dt/2*k1, u, p);
k3 = dinamica_oficial(x + dt/2*k2, u, p);
k4 = dinamica_oficial(x + dt*k3, u, p);
x = x + dt/6*(k1 + 2*k2 + 2*k3 + k4);
end

function dx = dinamica_oficial(x, vm, p)
% Modelo no lineal oficial de Quanser (Concept Review "Pendulum State Space
% Model", ecs. 1.1-1.3) con alpha = 0 arriba. Igual que qube_dinamica_oficial.m.
th_d = x(3);  al = x(2);  al_d = x(4);
s = sin(al);  c = cos(al);
Jr = p.mr*p.r^2/3;  Jp = p.mp*p.Lp^2/3;  l = p.Lp/2;  mlr = p.mp*l*p.r;
tau = p.km*(vm - p.km*th_d)/p.Rm;
m11 = Jr + Jp*s^2;  m12 = -mlr*c;  m22 = Jp;
c1 = tau - p.br*th_d - 2*Jp*s*c*th_d*al_d - mlr*s*al_d^2;
c2 = -p.bp*al_d + Jp*s*c*th_d^2 + p.mp*p.g*l*s;
det_m = m11*m22 - m12^2;
dx = [th_d; al_d; (m22*c1 - m12*c2)/det_m; (-m12*c1 + m11*c2)/det_m];
end

function p = parametros_oficiales()
% qube3_rotpen_param.m (Quanser)
p.Rm = 7.5;  p.km = 0.0422;  p.mr = 0.095;  p.r = 0.085;  p.br = 1e-3;
p.mp = 0.024;  p.Lp = 0.129;  p.bp = 5e-5;  p.g = 9.81;  p.enc_counts = 2048;
end

function guardar_png(fig, carpeta, nombre)
try
    print(fig, fullfile(carpeta, [nombre '.png']), '-dpng', '-r110');
catch
end
end
