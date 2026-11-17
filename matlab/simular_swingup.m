function s = simular_swingup(red_swingup, p, x0, T, ruido)
%SIMULAR_SWINGUP Simula el Qube NO lineal controlado por dqn_swingup_block.
%   s = simular_swingup(red_swingup, p, x0, T, ruido)
%   red_swingup : struct de dqn_swingup_qube.mat
%   p           : struct de qube_params() (puede traer parámetros perturbados)
%   x0          : [theta; alpha; theta_dot; alpha_dot], alpha = 0 arriba, +-pi colgando
%   T           : duración (s)          ruido: desv. estándar en theta_dot medida (rad/s) [0]
%
%   Reproduce lo que pasa en tu Simulink: planta (RK4 a 2 ms), encoders de
%   2048 cuentas/rev, alpha envuelto a (-pi, pi], theta_dot de tu observador
%   (Euler a 2 ms, como ode1) y el MISMO código del bloque de Simulink.
if nargin < 5, ruido = 0; end
dt = red_swingup.Ts;
nT = round(T/dt);
q  = 2*pi/p.enc_counts;
L = red_swingup.L;  beta = red_swingup.beta;  m = red_swingup.m;

clear dqn_swingup_block                      % reinicia la memoria del bloque
x = x0(:);
y_th = q*round(x(1)/q);
p_o = y_th;  v_o = x(3);  z_o = 0;           % observador de theta (convergido)
X = zeros(4, nT+1);  U = zeros(1, nT);
X(:,1) = x;
for k = 1:nT
    % --- lo que entrega "Qube With Pendulum"
    y_th = q*round(x(1)/q);
    y_al = q*round(x(2)/q);
    y_al = mod(y_al + pi, 2*pi) - pi;        % alpha envuelto, como en el Simulink
    th_d = v_o + ruido*randn();
    e   = y_th - p_o;                        % observador de theta (Euler)
    p_n = p_o + dt*v_o;
    v_n = v_o + dt*(L*e + m*z_o);
    z_o = z_o + dt*(e - beta*z_o);
    p_o = p_n;  v_o = v_n;

    % --- bloque de Simulink (red DQN)
    u = dqn_swingup_block([y_th; y_al; th_d; 0], red_swingup);
    U(k) = u;

    % --- planta no lineal (RK4)
    k1 = qube_nonlinear_dynamics(x,           u, p);
    k2 = qube_nonlinear_dynamics(x + dt/2*k1, u, p);
    k3 = qube_nonlinear_dynamics(x + dt/2*k2, u, p);
    k4 = qube_nonlinear_dynamics(x + dt*k3,   u, p);
    x  = x + dt/6*(k1 + 2*k2 + 2*k3 + k4);
    X(:,k+1) = x;
end
s.t = (0:nT)*dt;
s.X = X;
s.U = U;
s.alpha_env = mod(X(2,:) + pi, 2*pi) - pi;   % alpha envuelto (rad)
cola = s.t >= T - 2;
s.exito = max(abs(s.alpha_env(cola))) < 5*pi/180 && max(abs(X(1,cola))) < 45*pi/180 ...
          && max(abs(X(1,:))) < red_swingup.theta_max;
fuera = find(abs(s.alpha_env) > 10*pi/180, 1, 'last');
s.t_subida = s.t(fuera);
end
