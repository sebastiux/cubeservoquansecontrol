function p = qube_parametros_oficiales()
%QUBE_PARAMETROS_OFICIALES Parámetros oficiales del Qube-Servo 3 con péndulo.
%   Fuente: quanser/Quanser_Academic_Resources (rama dev-windows),
%   6_teaching/1_Controls/Qube_Servo_3/sp6_pendulum_control/1b_lqr_control/
%   hardware/matlab/qube3_rotpen_param.m
%   No incluye ningún modelo linealizado.
p.Rm = 7.5;        % Resistencia (Ohm)
p.kt = 0.0422;     % Constante de par (N-m/A)
p.km = 0.0422;     % Constante contra-electromotriz (V-s/rad)
p.mr = 0.095;      % Masa del brazo (kg)
p.r  = 0.085;      % Longitud del brazo (m)
p.br = 1e-3;       % Amortiguamiento del brazo (N-m-s/rad)
p.mp = 0.024;      % Masa del péndulo (kg)
p.Lp = 0.129;      % Longitud del péndulo (m)
p.bp = 5e-5;       % Amortiguamiento del péndulo (N-m-s/rad)
p.g  = 9.81;
p.Ts = 0.002;      % paso del Simulink (500 Hz)
p.enc_counts = 2048;
p.Vmax = 10;
end
