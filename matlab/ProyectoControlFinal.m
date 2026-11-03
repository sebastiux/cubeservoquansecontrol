%% Parametros del Simulador
%Motor
clc; clear;
Rm = 7.5; % Resistance
kt = 0.0422; % Current-torque (N-m/A)
km = 0.0422; % Back-emf constant (V-s/rad)

%Rotary Arm
mr = 0.095; % Mass (kg)
r = 0.085; % Total length (m)
Jr = mr*r^2/3; % Moment of inertia about pivot (kg-m^2)
br = 1e-3; % Equivalent Viscous Damping Coefficient (N-m-s/rad)

%Pendulum Link
mp = 0.024; % Mass (kg)
Lp = 0.129; % Total length (m)
l = Lp/2; % Pendulum center of mass (m)
Jp = mp*Lp^2/3; % Moment of inertia about pivot (kg-m^2)
bp = 5e-5; % Equivalent Viscous Damping Coefficient (N-m-s/rad)
g = 9.81; % Gravity Constant

Jt=(Jr + mp*r^2)*Jp - (mp^2)*(l^2)*(r^2); %Inercia Total

% Load Model
%qube3_rotpen_param;
% Set open-loop state-space model of rotary single-inverted pendulum (SIP)
%rotpen_ABCD_eqns_ip;
% Display matrices

A32 = ((mp^2)*(l^2)*r*g)/Jt;
A33 = -((br*Jp)/Jt) - (((km^2)*(Jp))/(Rm*Jt));
A34 = -(mp*l*r*bp)/Jt;
A42 = (mp*g*l)*(Jr+mp*r^2)/Jt;
A43 = -((mp*l*r*br)/Jt) - (km^2*mp*l*r)/(Rm*Jt);
A44 = -(Jr + mp*r^2)*bp/Jt;

B3 = (km*Jp)/(Rm*Jt);
B4 = (km*mp*l*r)/(Rm*Jt);

A = [0  0   1   0;
     0  0   0   1;
     0  A32 A33 A34;
     0  A42 A43 A44]

B = [0; 0; B3; B4]



% Balance Control
% Find Q, R and K matrices
% LQR Weighting Matrices
Q = diag([1 20  .1 .1]);
R = 10;
K = lqr(A,B,Q,R); %Ganancia para el controlador
display(K)

eig(A-B*K) %Polos del controlador




% Observador
%%% Buscar l m beta
syms teta tetaG tetaGP q s l m beta
a = 100;

Aobs=[0 1 0;
    -l 0 m;
    -1 0 -beta];

Bobs=[0;
    l;
    1];
%%% Buscar polinomio caracteristico

Ps = det(eye(size(Aobs))*s-Aobs)

Pd = (s+a)*(s+a)*(s+a)

pd = expand(Pd)

coeficientes = sym2poly(pd)



beta= 3*a
L = 3*a^2
m = a^3 - (L*beta)
Aobs=[0 1 0;
     -L 0 m;
     -1 0 -beta]
eig(Aobs)