function u = dqn_swingup_block(x, red_swingup)
%DQN_SWINGUP_BLOCK Red DQN que LEVANTA el pendulo (swing-up) y lo MANTIENE
%   arriba. Control inverso: estado medido -> voltaje del motor.  (version 3)
%
%   x          : senal X de "Qube With Pendulum". Solo se usan los angulos de
%                los encoders: x(1) = theta, x(2) = alpha envuelto a (-pi, pi]
%                (0 = arriba, +-pi = colgando).
%   red_swingup: struct guardado en el .mat de la red
%   u          : voltaje del motor (V), YA FILTRADO
%
%   Dentro del bloque:
%   1) Velocidades con el observador de 3er orden del proyecto, en dos
%      versiones (lenta a = 100 y rapida a = 250), usando alpha DESENROLLADO.
%   2) La red decide cada n_hold pasos: un voltaje, o un incremento de voltaje
%      si red_swingup.incremental = 1.
%   3) FILTRO PASA BAJAS de 2.o orden a la salida (filtro_w rad/s, filtro_zeta),
%      integrado con Euler a Ts igual que el bloque continuo con ode1. La red
%      se entreno con este filtro: NO agregar otro filtro despues del bloque.
%
%   Debe ejecutarse cada red_swingup.Ts = 2 ms (ZOH a la entrada).
%   Seguridad: si |theta| > theta_max el voltaje queda en 0 hasta reiniciar.
%#codegen
persistent a_prev a_unw p_o v_o z_o cnt u_cmd f1 f2 apagado
theta = x(1);  alpha = x(2);
if isempty(a_prev)
    a_prev = alpha;  a_unw = alpha;
    p_o = [theta theta; alpha alpha];     % filas: theta, alpha; columnas: lento, rapido
    v_o = zeros(2, 2);  z_o = zeros(2, 2);
    cnt = 0;  u_cmd = 0;  f1 = 0;  f2 = 0;  apagado = false;
end
Ts = red_swingup.Ts;

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
p_n = p_o + Ts*v_o;
v_n = v_o + Ts*(Lg.*e + mg.*z_o);
z_n = z_o + Ts*(e - bg.*z_o);
p_o = p_n;  v_o = v_n;  z_o = z_n;

% --- red neuronal: decide cada n_hold pasos
if cnt == 0
    obs = [theta; sin(alpha); cos(alpha); vel(1,1); vel(2,1); vel(1,2); vel(2,2); ...
           u_cmd; f1; f2]./red_swingup.x_scale(:);
    h1 = max(red_swingup.W1*obs + red_swingup.b1(:), 0);
    h2 = max(red_swingup.W2*h1  + red_swingup.b2(:), 0);
    q  = red_swingup.W3*h2 + red_swingup.b3(:);
    [~, i] = max(q);
    if red_swingup.incremental > 0.5
        u_cmd = u_cmd + red_swingup.acciones(i);
    else
        u_cmd = red_swingup.acciones(i);
    end
    u_cmd = min(max(u_cmd, -red_swingup.Vmax), red_swingup.Vmax);
end
cnt = cnt + 1;
if cnt >= red_swingup.n_hold, cnt = 0; end

% --- filtro pasa bajas de 2.o orden: la salida es el estado (como un integrador)
u_f = f1;
w   = red_swingup.filtro_w;
f1n = f1 + Ts*f2;
f2n = f2 + Ts*(w*w*(u_cmd - f1) - 2*red_swingup.filtro_zeta*w*f2);
f1 = f1n;  f2 = f2n;

% --- seguridad: brazo demasiado lejos -> cortar voltaje
if abs(theta) > red_swingup.theta_max, apagado = true; end
if apagado
    u = 0;
else
    u = u_f;
end
end
