function s = simular_swingup(red_swingup, p, x0, T, opc)
%SIMULAR_SWINGUP Simula el Qube NO lineal controlado por dqn_swingup_block.
%   s = simular_swingup(red_swingup, p, x0, T, opc)
%   red_swingup : struct del .mat de la red
%   p           : struct de qube_params(); opcional p.tau_c = fricción seca
%                 del brazo (N·m)
%   x0          : [theta; alpha; theta_dot; alpha_dot], alpha = 0 arriba, +-pi colgando
%   T           : duración (s)
%   opc         : struct opcional con lo que el modelo ideal no tiene:
%                 .retardo (muestras de 2 ms) [0]   .k_u (ganancia de voltaje) [1]
%                 .offset (error de cero de alpha, rad) [0]
%
%   Reproduce tu Simulink: planta (RK4 a 2 ms), encoders de 2048 cuentas/rev,
%   alpha envuelto a (-pi, pi] y el MISMO código del bloque de Simulink.
if nargin < 5, opc = struct(); end
if ~isfield(opc, 'retardo'), opc.retardo = 0; end
if ~isfield(opc, 'k_u'),     opc.k_u = 1;     end
if ~isfield(opc, 'offset'),  opc.offset = 0;  end
dt = red_swingup.Ts;
nT = round(T/dt);
q  = 2*pi/p.enc_counts;

clear dqn_swingup_block                      % reinicia la memoria del bloque
x = x0(:);
X = zeros(4, nT+1);  U = zeros(1, nT);
X(:,1) = x;
fifo = zeros(1, opc.retardo + 1);
for k = 1:nT
    % --- lo que entrega "Qube With Pendulum" (solo se usan los ángulos)
    y_th = q*round(x(1)/q);
    y_al = q*round((x(2) + opc.offset)/q);
    y_al = mod(y_al + pi, 2*pi) - pi;        % alpha envuelto, como en el Simulink

    % --- bloque de Simulink (red DQN)
    u = dqn_swingup_block([y_th; y_al; 0; 0], red_swingup);
    U(k) = u;
    fifo = [u, fifo(1:end-1)];
    ur = opc.k_u*fifo(end);                  % voltaje que realmente llega al motor

    % --- planta no lineal (RK4)
    k1 = qube_nonlinear_dynamics(x,           ur, p);
    k2 = qube_nonlinear_dynamics(x + dt/2*k1, ur, p);
    k3 = qube_nonlinear_dynamics(x + dt/2*k2, ur, p);
    k4 = qube_nonlinear_dynamics(x + dt*k3,   ur, p);
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
s.temblor  = std(s.alpha_env(cola))*180/pi;  % grados RMS en los últimos 2 s
end
