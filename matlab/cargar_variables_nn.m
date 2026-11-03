%% CARGAR_VARIABLES_NN
% Carga en el workspace todas las variables que usa el Simulink
% qube_nn_nonlinear_sim.slx (se ejecuta solo como InitFcn del modelo).
% También sirve para tu modelo de hardware: deja W1..b3, x_scale, Vmax
% listos para el bloque "Controlador NN".

nn_dir_ = fileparts(mfilename('fullpath'));
addpath(nn_dir_);
nn_ = load(fullfile(nn_dir_, '..', 'models', 'nn_qube_policy.mat'));

% --- Red neuronal (parámetros del bloque MATLAB Function)
W1 = nn_.W1;  b1 = nn_.b1;
W2 = nn_.W2;  b2 = nn_.b2;
W3 = nn_.W3;  b3 = nn_.b3;
x_scale = nn_.x_scale;
Vmax    = nn_.Vmax;

% --- Planta y LQR
qp = qube_params();
qp = rmfield(qp, {'A', 'B', 'Q', 'R'});   % la planta solo necesita escalares
qp.Vmax = Vmax;
K  = qp.K;           % ganancia LQR (para comparar)
Ts = qp.Ts;

% --- Condición inicial y selectores
if ~exist('x0_sim', 'var'),           x0_sim = [0; 12*pi/180; 0; 0]; end
if ~exist('usar_nn', 'var'),          usar_nn = 1;          end   % 1 = NN, 0 = LQR
if ~exist('usar_sensor_real', 'var'), usar_sensor_real = 1; end   % 1 = encoders + filtro

% --- Sensor: cuantización del encoder y derivada filtrada 50s/(s+50)
q_enc = 2*pi/qp.enc_counts;
wc_ = 50;
c1_ = 1/(1 + wc_*Ts);  c2_ = wc_/(1 + wc_*Ts);
% v[k] = c1 v[k-1] + c2 (y[k] - y[k-1])   (Euler hacia atrás, 2 canales)
A_f = kron(eye(2), [c1_ -c2_; 0 0]);
B_f = kron(eye(2), [c2_; 1]);
C_f = kron(eye(2), [c1_ -c2_]);
D_f = c2_*eye(2);
IC_f = [0; x0_sim(1); 0; x0_sim(2)];

clear nn_ nn_dir_ wc_ c1_ c2_
