function u = politica_dqn(Y, red_dqn)
%POLITICA_DQN Versión vectorizada (4xN) del bloque dqn_controller_block.
%   u = politica_dqn(Y, red_dqn)   ->  acciones( argmax_a Q(Y, a) ),  1xN
%   Mismo cálculo que el bloque de Simulink, para N simulaciones a la vez.
H  = max(red_dqn.W1*(Y./red_dqn.x_scale(:)) + red_dqn.b1(:), 0);
H  = max(red_dqn.W2*H + red_dqn.b2(:), 0);
Qv = red_dqn.W3*H + red_dqn.b3(:);              % nA x N
[~, a] = max(Qv, [], 1);
u = reshape(red_dqn.acciones(a), 1, []);
end
