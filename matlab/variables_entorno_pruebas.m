%% VARIABLES_ENTORNO_PRUEBAS
% Variables del Qube SIMULADO de simulink/qube_dqn_pruebas.slx
% (lo ejecuta la InitFcn del modelo después de cargar_red_dqn).
%
% Puedes cambiar x0_sim antes de dar Run, por ejemplo:
%   x0_sim = [0; 12*pi/180; 0; 0];

qp = qube_params();
qp = rmfield(qp, {'A', 'B'});              % la planta solo usa escalares
Ts = qp.Ts;                                % 2 ms, como tu modelo
q_enc = 2*pi/qp.enc_counts;                % resolución del encoder

if ~exist('x0_sim', 'var')
    x0_sim = [0; 10*pi/180; 0; 0];         % [theta; alpha; theta_dot; alpha_dot]
end

% Observador de 3er orden por canal (igual al subsistema "State X"):
%   pos' = vel;  vel' = L (y - pos) + m z;  z' = (y - pos) - beta z
A1 = [0 1 0; -L 0 m; -1 0 -beta];
B1 = [0; L; 1];
C1 = [0 1 0];                              % salida: velocidad estimada
A_obs = blkdiag(A1, A1);
B_obs = blkdiag(B1, B1);                   % entradas [theta; alpha]
C_obs = blkdiag(C1, C1);                   % salidas  [theta_dot; alpha_dot]
IC_obs = [x0_sim(1); x0_sim(3); 0; x0_sim(2); x0_sim(4); 0];   % observador convergido
clear A1 B1 C1
