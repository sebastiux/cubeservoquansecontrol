function u = politica_rl(Y, ctrl)
%POLITICA_RL Versión vectorizada (4xN) de los bloques DQN / Q-learning.
%   u = politica_rl(Y, red_dqn)   ->  acciones(argmax Q) de la red DQN
%   u = politica_rl(Y, tabla_q)   ->  acciones(argmax Q) de la tabla
%   Mismo resultado que dqn_controller_block / qlearning_controller_block,
%   pero para N simulaciones a la vez.

if isfield(ctrl, 'W1')                         % ---- DQN
    H = max(ctrl.W1*(Y./ctrl.x_scale(:)) + ctrl.b1(:), 0);
    H = max(ctrl.W2*H + ctrl.b2(:), 0);
    Qv = ctrl.W3*H + ctrl.b3(:);               % nA x N
    [~, a] = max(Qv, [], 1);
else                                           % ---- Q-learning tabular
    n = ctrl.n_bins;
    i1 = sum(Y(1,:) > ctrl.bordes_theta(:), 1);
    i2 = sum(Y(2,:) > ctrl.bordes_alpha(:), 1);
    i3 = sum(Y(3,:) > ctrl.bordes_theta_d(:), 1);
    i4 = sum(Y(4,:) > ctrl.bordes_alpha_d(:), 1);
    s  = i1 + n(1)*(i2 + n(2)*(i3 + n(3)*i4)) + 1;
    [~, a] = max(ctrl.Q(s, :), [], 2);
end
u = reshape(ctrl.acciones(a), 1, []);
end
