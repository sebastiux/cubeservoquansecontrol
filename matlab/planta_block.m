function dx = planta_block(x, u, qp)
%PLANTA_BLOCK Código del bloque MATLAB Function "Planta no lineal" en Simulink.
%   dx/dt = f(x, u). Se integra con un bloque Integrator (estado continuo).
%#codegen
dx = qube_nonlinear_dynamics(x(:), u, qp);
end
