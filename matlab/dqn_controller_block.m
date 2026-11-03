function u = dqn_controller_block(x, red_dqn)
%DQN_CONTROLLER_BLOCK Código del bloque "MATLAB Function" (Deep Q-Learning).
%   x       : 4x1 [theta; alpha; theta_dot; alpha_dot]  (la señal X del Qube,
%             la misma que usa el LQR, SIN el signo menos del bloque Sum)
%   red_dqn : PARÁMETRO del bloque (struct del workspace, ver
%             cargar_controlador_rl.m) con W1,b1,W2,b2,W3,b3,x_scale,acciones
%   u       : voltaje del motor = acciones( argmax_a Q(x,a) )
%
%   Q(x,:) = W3*relu( W2*relu( W1*(x./x_scale) + b1 ) + b2 ) + b3
%   Solo operaciones matriciales -> compatible con codegen / QUARC.
%#codegen
xs = x(:)./red_dqn.x_scale(:);
h1 = max(red_dqn.W1*xs + red_dqn.b1(:), 0);
h2 = max(red_dqn.W2*h1 + red_dqn.b2(:), 0);
q  = red_dqn.W3*h2 + red_dqn.b3(:);
[~, i] = max(q);
u = red_dqn.acciones(i);
end
