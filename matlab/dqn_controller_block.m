function u = dqn_controller_block(x, red_dqn)
%DQN_CONTROLLER_BLOCK Código del bloque "MATLAB Function" Controlador DQN.
%   Control inverso con la red neuronal DQN: la red recibe el estado medido
%   y entrega directamente el voltaje del motor (sustituye a u = -K*x).
%
%   x       : 4x1 [theta; alpha; theta_dot; alpha_dot] = señal X que sale de
%             "Qube With Pendulum" (encoders + observadores)
%   red_dqn : PARÁMETRO del bloque (struct del workspace, cargar_red_dqn.m)
%   u       : voltaje (V) = acciones( argmax_a Q(x,a) )
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
