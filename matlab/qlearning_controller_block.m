function u = qlearning_controller_block(x, tabla_q)
%QLEARNING_CONTROLLER_BLOCK Código del bloque "MATLAB Function" (Q-learning tabular).
%   x       : 4x1 [theta; alpha; theta_dot; alpha_dot]
%   tabla_q : PARÁMETRO del bloque (struct del workspace, ver
%             cargar_controlador_rl.m) con Q, bordes_*, n_bins, acciones
%   u       : voltaje = acciones( argmax_a Q(s(x), a) )
%
%   Celda: i_k = número de bordes menores que x_k,
%          s = i1 + n1*(i2 + n2*(i3 + n3*i4)) + 1   (igual que en Python)
%#codegen
i1 = sum(x(1) > tabla_q.bordes_theta(:));
i2 = sum(x(2) > tabla_q.bordes_alpha(:));
i3 = sum(x(3) > tabla_q.bordes_theta_d(:));
i4 = sum(x(4) > tabla_q.bordes_alpha_d(:));
n  = tabla_q.n_bins;
s  = i1 + n(1)*(i2 + n(2)*(i3 + n(3)*i4)) + 1;
[~, a] = max(tabla_q.Q(s, :));
u = tabla_q.acciones(a);
end
