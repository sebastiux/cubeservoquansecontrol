function p = qube_params()
%QUBE_PARAMS Parámetros del Qube-Servo 3 + péndulo (mismos que ProyectoControlFinal.m)
%   p = qube_params() devuelve un struct con los parámetros físicos, el
%   modelo lineal (A, B) y la ganancia LQR K con Q = diag([1 20 .1 .1]), R = 10.
%   Deben coincidir con python/qube_model.py (QubeParams).

% Motor
p.Rm = 7.5;        % Resistencia (Ohm)
p.kt = 0.0422;     % Constante de par (N-m/A)
p.km = 0.0422;     % Constante back-emf (V-s/rad)

% Brazo rotatorio
p.mr = 0.095;               % Masa (kg)
p.r  = 0.085;               % Longitud total (m)
p.Jr = p.mr*p.r^2/3;        % Inercia respecto al pivote (kg-m^2)
p.br = 1e-3;                % Amortiguamiento (N-m-s/rad)

% Péndulo
p.mp = 0.024;               % Masa (kg)
p.Lp = 0.129;               % Longitud total (m)
p.l  = p.Lp/2;              % Centro de masa (m)
p.Jp = p.mp*p.Lp^2/3;       % Inercia respecto al pivote (kg-m^2)
p.bp = 5e-5;                % Amortiguamiento (N-m-s/rad)
p.g  = 9.81;

% Actuación / muestreo
p.Vmax = 10;                % Saturación (V) usada en el entrenamiento
p.Ts   = 0.002;             % Paso fijo del Simulink (500 Hz)
p.enc_counts = 2048;        % Cuentas/rev en cuadratura (512 CPR x 4)

% Modelo lineal (idéntico a ProyectoControlFinal.m)
Jt  = (p.Jr + p.mp*p.r^2)*p.Jp - (p.mp^2)*(p.l^2)*(p.r^2);
A32 = ((p.mp^2)*(p.l^2)*p.r*p.g)/Jt;
A33 = -((p.br*p.Jp)/Jt) - (((p.km^2)*(p.Jp))/(p.Rm*Jt));
A34 = -(p.mp*p.l*p.r*p.bp)/Jt;
A42 = (p.mp*p.g*p.l)*(p.Jr+p.mp*p.r^2)/Jt;
A43 = -((p.mp*p.l*p.r*p.br)/Jt) - (p.km^2*p.mp*p.l*p.r)/(p.Rm*Jt);
A44 = -(p.Jr + p.mp*p.r^2)*p.bp/Jt;
B3  = (p.km*p.Jp)/(p.Rm*Jt);
B4  = (p.km*p.mp*p.l*p.r)/(p.Rm*Jt);
p.A = [0 0 1 0; 0 0 0 1; 0 A32 A33 A34; 0 A42 A43 A44];
p.B = [0; 0; B3; B4];

p.Q = diag([1 20 .1 .1]);
p.R = 10;
if exist('lqr', 'file') == 2 || exist('lqr', 'builtin') == 5
    p.K = lqr(p.A, p.B, p.Q, p.R);
else
    % Sin Control System Toolbox: valor calculado con scipy (mismo Q, R)
    p.K = [-0.316227766 23.245193248 -0.594815610 1.870086840];
end
end
