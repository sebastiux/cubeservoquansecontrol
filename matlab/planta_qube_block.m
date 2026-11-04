function dx = planta_qube_block(x, u, qp)
%PLANTA_QUBE_BLOCK Código del bloque MATLAB Function "Dinámica no lineal"
%   del entorno de pruebas: dx/dt = f(x, u). Lo integra un bloque Integrator.
%#codegen
dx = qube_nonlinear_dynamics(x(:), u, qp);
end
