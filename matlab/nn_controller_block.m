function u = nn_controller_block(x, W1, b1, W2, b2, W3, b3, x_scale, Vmax)
%NN_CONTROLLER_BLOCK Código para el bloque "MATLAB Function" de Simulink.
%   Entrada x (4x1) = [theta; alpha; theta_dot; alpha_dot] (la misma señal X
%   que hoy entra a la ganancia "u = -K*x" de tu modelo).
%   W1..b3, x_scale, Vmax se declaran como PARÁMETROS del bloque (Scope =
%   Parameter) y se toman del workspace después de:
%       load('nn_qube_policy.mat')
%   Solo usa operaciones matriciales -> compatible con generación de código
%   (Simulink Coder / QUARC) para correr en el Qube real a 500 Hz.
%#codegen
xs = x(:)./x_scale(:);
% red impar: u(-x) = -u(x)  ->  u(0) = 0 exacto
gp = W3*tanh(W2*tanh(W1*xs + b1(:)) + b2(:)) + b3(1);
gn = W3*tanh(W2*tanh(-W1*xs + b1(:)) + b2(:)) + b3(1);
u  = Vmax*tanh(0.5*(gp - gn));
end
