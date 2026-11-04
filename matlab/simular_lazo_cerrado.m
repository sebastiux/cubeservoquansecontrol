function out = simular_lazo_cerrado(ctrl, x0, p, T, opts)
%SIMULAR_LAZO_CERRADO Simula la planta NO lineal con un controlador discreto.
%   out = simular_lazo_cerrado(ctrl, x0, p, T, opts)
%   ctrl : handle  u = ctrl(y)   (y: 4xN medición, u: 1xN)
%   x0   : 4xN estados iniciales (N simulaciones en paralelo)
%   p    : struct de qube_params() (los campos físicos pueden ser 1xN)
%   T    : tiempo de simulación (s)
%   opts : struct opcional
%       .sensor  'observador' (encoders + TU observador de 3er orden, como
%                el subsistema "State X" del Simulink original) [defecto]
%                'filtro' (encoders + derivada filtrada 50s/(s+50))
%                'ideal'  (estado exacto)
%       .noise   desviación estándar de ruido en velocidades (rad/s)  [0]
%       .dist_t  instante de una perturbación (impulso en alpha_dot) [inf]
%       .dist_v  magnitud del impulso en alpha_dot (rad/s)            [0]
%       .Ts_ctrl periodo del controlador (s); la acción se mantiene (ZOH)
%                entre decisiones (Ts_agente del DQN = 0.01)          [p.Ts]
%
%   Planta: RK4 a paso fijo p.Ts. Sensores y observador: Euler a p.Ts (igual
%   que el solver ode1 de tu modelo). Mismo orden de cálculo que python/rl_env.py.

if nargin < 5, opts = struct(); end
if ~isfield(opts, 'sensor'),  opts.sensor  = 'observador'; end
if ~isfield(opts, 'noise'),   opts.noise   = 0;    end
if ~isfield(opts, 'dist_t'),  opts.dist_t  = inf;  end
if ~isfield(opts, 'dist_v'),  opts.dist_v  = 0;    end
if ~isfield(opts, 'Ts_ctrl'), opts.Ts_ctrl = p.Ts; end
n_hold = max(1, round(opts.Ts_ctrl/p.Ts));

dt = p.Ts;
nT = round(T/dt);
N  = size(x0, 2);
X  = zeros(4, N, nT+1);
U  = zeros(1, N, nT);
x  = x0;
X(:,:,1) = x;

q  = 2*pi/p.enc_counts;
wc = 50;
a_obs = 100;                                   % ProyectoControlFinal.m
beta = 3*a_obs;  Lo = 3*a_obs^2;  mo = a_obs^3 - Lo*beta;

% medición inicial (observador ya convergido)
y   = medir_angulos(x, opts.sensor, q);
vel = x(3:4,:);
p_o = y;  v_o = x(3:4,:);  z_o = zeros(2, N);

for k = 1:nT
    % ---- control (recalculado cada n_hold pasos)
    if mod(k-1, n_hold) == 0
        ym = [y; vel];
        if opts.noise > 0
            ym(3:4,:) = ym(3:4,:) + opts.noise*randn(2, N);
        end
        u = ctrl(ym);
        u = min(max(u, -p.Vmax), p.Vmax);
    end
    U(:,:,k) = u;

    % ---- planta no lineal (RK4)
    k1 = qube_nonlinear_dynamics(x,           u, p);
    k2 = qube_nonlinear_dynamics(x + dt/2*k1, u, p);
    k3 = qube_nonlinear_dynamics(x + dt/2*k2, u, p);
    k4 = qube_nonlinear_dynamics(x + dt*k3,   u, p);
    x  = x + dt/6*(k1 + 2*k2 + 2*k3 + k4);
    if abs(k*dt - opts.dist_t) < dt/2
        x(4,:) = x(4,:) + opts.dist_v;
    end
    X(:,:,k+1) = x;

    % ---- sensores
    y_new = medir_angulos(x, opts.sensor, q);
    switch lower(opts.sensor)
        case 'observador'
            vel = v_o;                         % salida en k; luego Euler con y_k
            e   = y_new - p_o;
            p_o = p_o + dt*v_o;
            v_o = v_o + dt*(Lo*e + mo*z_o);
            z_o = z_o + dt*(e - beta*z_o);
        case 'filtro'
            vel = (vel + wc*(y_new - y))/(1 + wc*dt);
        otherwise
            vel = x(3:4,:);
    end
    y = y_new;
end

out.t = (0:nT)*dt;
out.X = X;          % 4 x N x (nT+1)
out.U = U;          % 1 x N x nT
end

function ya = medir_angulos(xx, sensor, q)
if strcmpi(sensor, 'ideal')
    ya = xx(1:2,:);
else
    ya = q*round(xx(1:2,:)/q);                 % encoder 2048 cuentas/rev
end
end
