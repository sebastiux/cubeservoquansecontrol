function out = simular_lazo_cerrado(ctrl, x0, p, T, opts)
%SIMULAR_LAZO_CERRADO Simula la planta NO lineal con un controlador discreto.
%   out = simular_lazo_cerrado(ctrl, x0, p, T, opts)
%   ctrl : handle  u = ctrl(y)   (y: 4xN medición, u: 1xN)
%   x0   : 4xN estados iniciales (N simulaciones en paralelo)
%   p    : struct de qube_params() (puede traer parámetros perturbados)
%   T    : tiempo de simulación (s)
%   opts : struct opcional
%       .sensor  'ideal' (estado exacto) | 'real' (encoders cuantizados +
%                derivada filtrada 50s/(s+50), como tu Simulink)  [def 'ideal']
%       .noise   desviación estándar de ruido en velocidades (rad/s)  [0]
%       .dist_t  instante de una perturbación (impulso en alpha_dot) [inf]
%       .dist_v  magnitud del impulso en alpha_dot (rad/s)            [0]
%       .Ts_ctrl periodo del controlador (s); la acción se mantiene (ZOH)
%                entre decisiones. Agentes RL: Ts_agente = 0.01     [p.Ts]
%
%   Integración RK4 a paso fijo p.Ts con u retenido (ZOH), igual que en el
%   entrenamiento en Python.

if nargin < 5, opts = struct(); end
if ~isfield(opts, 'sensor'), opts.sensor = 'ideal'; end
if ~isfield(opts, 'noise'),  opts.noise  = 0;       end
if ~isfield(opts, 'dist_t'), opts.dist_t = inf;     end
if ~isfield(opts, 'dist_v'), opts.dist_v = 0;       end
if ~isfield(opts, 'Ts_ctrl'), opts.Ts_ctrl = p.Ts;  end
n_hold = max(1, round(opts.Ts_ctrl/p.Ts));

dt = p.Ts;
nT = round(T/dt);
N  = size(x0, 2);
X  = zeros(4, N, nT+1);
U  = zeros(1, N, nT);
x  = x0;
X(:,:,1) = x;

real_sensor = strcmpi(opts.sensor, 'real');
q  = 2*pi/p.enc_counts;
wc = 50;
y_prev = q*round(x(1:2,:)/q);
v_f    = x(3:4,:);

for k = 1:nT
    % ---- medición
    if real_sensor
        y_ang = q*round(x(1:2,:)/q);
        if k > 1
            v_f = (v_f + wc*(y_ang - y_prev))/(1 + wc*dt);
        end
        y_prev = y_ang;
        y = [y_ang; v_f];
    else
        y = x;
    end
    if opts.noise > 0
        y(3:4,:) = y(3:4,:) + opts.noise*randn(2, N);
    end

    % ---- control (saturado), recalculado cada n_hold pasos
    if mod(k-1, n_hold) == 0
        u = ctrl(y);
        u = min(max(u, -p.Vmax), p.Vmax);
    end
    U(:,:,k) = u;

    % ---- planta no lineal (RK4)
    k1 = qube_nonlinear_dynamics(x,            u, p);
    k2 = qube_nonlinear_dynamics(x + dt/2*k1,  u, p);
    k3 = qube_nonlinear_dynamics(x + dt/2*k2,  u, p);
    k4 = qube_nonlinear_dynamics(x + dt*k3,    u, p);
    x  = x + dt/6*(k1 + 2*k2 + 2*k3 + k4);

    if abs(k*dt - opts.dist_t) < dt/2
        x(4,:) = x(4,:) + opts.dist_v;
    end
    X(:,:,k+1) = x;
end

out.t = (0:nT)*dt;
out.X = X;          % 4 x N x (nT+1)
out.U = U;          % 1 x N x nT
end
