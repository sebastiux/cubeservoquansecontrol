function u = dqn_swingup_block(x, red_swingup)
%DQN_SWINGUP_BLOCK Red DQN que LEVANTA el pendulo (swing-up) y lo MANTIENE
%   arriba. Control inverso: estado medido -> voltaje del motor.
%
%   x          : 4x1 = senal X de "Qube With Pendulum"
%                [theta; alpha; theta_dot; alpha_dot], con alpha envuelto a
%                (-pi, pi]: 0 = arriba, +-pi = colgando. Solo se usan x(1:3):
%                la velocidad del pendulo se estima AQUI con el mismo
%                observador de 3er orden del modelo, pero con alpha
%                DESENROLLADO (el observador del modelo da picos cuando
%                alpha salta de +180 a -180 grados al pasar por abajo).
%   red_swingup: struct guardado en dqn_swingup_qube.mat
%   u          : voltaje del motor (V)
%
%   Debe ejecutarse cada red_swingup.Ts = 2 ms (ZOH a la entrada). La red
%   decide cada n_hold pasos (10 ms) y mantiene el voltaje entre decisiones.
%   Seguridad: si |theta| > theta_max el voltaje queda en 0 hasta reiniciar.
%#codegen
persistent a_prev a_unw p_o v_o z_o cnt u_hold apagado
theta = x(1);  alpha = x(2);  theta_d = x(3);
if isempty(a_prev)
    a_prev = alpha;  a_unw = alpha;
    p_o = alpha;  v_o = 0;  z_o = 0;
    cnt = 0;  u_hold = 0;  apagado = false;
end

% --- alpha desenrollado (sin el salto de 2*pi)
d = alpha - a_prev;
d = d - 2*pi*round(d/(2*pi));
a_unw  = a_unw + d;
a_prev = alpha;

% --- observador de velocidad del pendulo (Euler a Ts, como ode1)
alpha_d = v_o;
e   = a_unw - p_o;
p_n = p_o + red_swingup.Ts*v_o;
v_n = v_o + red_swingup.Ts*(red_swingup.L*e + red_swingup.m*z_o);
z_n = z_o + red_swingup.Ts*(e - red_swingup.beta*z_o);
p_o = p_n;  v_o = v_n;  z_o = z_n;

% --- red neuronal: decide cada n_hold pasos
if cnt == 0
    obs = [theta; sin(alpha); cos(alpha); theta_d; alpha_d]./red_swingup.x_scale(:);
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
