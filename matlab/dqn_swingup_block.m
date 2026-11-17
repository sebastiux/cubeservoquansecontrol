function u = dqn_swingup_block(x, red_swingup)
%DQN_SWINGUP_BLOCK Red DQN que LEVANTA el pendulo (swing-up) y lo MANTIENE
%   arriba. Control inverso: estado medido -> voltaje del motor.  (version 2)
%
%   x          : senal X de "Qube With Pendulum". Solo se usan los angulos de
%                los encoders: x(1) = theta, x(2) = alpha envuelto a (-pi, pi]
%                (0 = arriba, +-pi = colgando).
%   red_swingup: struct guardado en el .mat de la red
%   u          : voltaje del motor (V)
%
%   Las velocidades se estiman AQUI con el observador de 3er orden del
%   proyecto, en dos versiones (lenta a = 100 y rapida a = 250), usando alpha
%   DESENROLLADO (sin el salto de +180 a -180 grados al pasar por abajo).
%   La red recibe ademas el voltaje anterior para poder suavizar la accion.
%
%   Debe ejecutarse cada red_swingup.Ts = 2 ms (ZOH a la entrada). La red
%   decide cada n_hold pasos y mantiene el voltaje entre decisiones.
%   Seguridad: si |theta| > theta_max el voltaje queda en 0 hasta reiniciar.
%#codegen
persistent a_prev a_unw p_o v_o z_o cnt u_hold apagado
theta = x(1);  alpha = x(2);
if isempty(a_prev)
    a_prev = alpha;  a_unw = alpha;
    p_o = [theta theta; alpha alpha];     % filas: theta, alpha; columnas: lento, rapido
    v_o = zeros(2, 2);  z_o = zeros(2, 2);
    cnt = 0;  u_hold = 0;  apagado = false;
end

% --- alpha desenrollado (sin el salto de 2*pi)
d = alpha - a_prev;
d = d - 2*pi*round(d/(2*pi));
a_unw  = a_unw + d;
a_prev = alpha;

% --- observadores de velocidad (Euler a Ts, como ode1)
Lg = [1; 1]*red_swingup.obs_L(:)';
mg = [1; 1]*red_swingup.obs_m(:)';
bg = [1; 1]*red_swingup.obs_beta(:)';
vel = v_o;
e   = [theta theta; a_unw a_unw] - p_o;
p_n = p_o + red_swingup.Ts*v_o;
v_n = v_o + red_swingup.Ts*(Lg.*e + mg.*z_o);
z_n = z_o + red_swingup.Ts*(e - bg.*z_o);
p_o = p_n;  v_o = v_n;  z_o = z_n;

% --- red neuronal: decide cada n_hold pasos
if cnt == 0
    obs = [theta; sin(alpha); cos(alpha); vel(1,1); vel(2,1); vel(1,2); vel(2,2); u_hold] ...
          ./red_swingup.x_scale(:);
    h1 = max(red_swingup.W1*obs + red_swingup.b1(:), 0);
    h2 = max(red_swingup.W2*h1  + red_swingup.b2(:), 0);
    q  = red_swingup.W3*h2 + red_swingup.b3(:);
    [~, i] = max(q);
    u_hold = red_swingup.acciones(i);
end
cnt = cnt + 1;
if cnt >= red_swingup.n_hold, cnt = 0; end

% --- seguridad: brazo demasiado lejos -> cortar voltaje
if abs(theta) > red_swingup.theta_max, apagado = true; end
if apagado
    u = 0;
else
    u = u_hold;
end
end
