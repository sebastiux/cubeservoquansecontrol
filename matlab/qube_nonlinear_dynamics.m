function dx = qube_nonlinear_dynamics(x, u, p)
%QUBE_NONLINEAR_DYNAMICS Modelo NO lineal del péndulo rotatorio (Furuta).
%   dx = qube_nonlinear_dynamics(x, u, p)
%   x : 4xN  [theta; alpha; theta_dot; alpha_dot]   (alpha = 0 -> arriba)
%   u : 1xN  voltaje del motor (V)
%   p : struct de qube_params(). Los campos físicos pueden ser escalares o
%       vectores 1xN (un juego de parámetros por simulación -> Monte Carlo).
%   Vectorizado: cada columna es una simulación independiente.
%
%   Ecuaciones de Euler-Lagrange (Jr, Jp respecto al pivote):
%   (Jr + mp r^2 + Jp sin^2 a) th'' - mp l r cos a a''
%        + 2 Jp sin a cos a th' a' + mp l r sin a a'^2 = tau - br th'
%   -mp l r cos a th'' + Jp a'' - Jp sin a cos a th'^2 - mp g l sin a = -bp a'
%   tau = km (Vm - km th')/Rm
%   Su linealización en x = 0 da exactamente p.A y p.B (ProyectoControlFinal.m).
%   Mismas ecuaciones que python/qube_model.py:dynamics.

th_d = x(3,:);  al = x(2,:);  al_d = x(4,:);
s = sin(al);    c = cos(al);

Jp  = p.mp.*p.Lp.^2/3;
Jr  = p.mr.*p.r.^2/3;
l   = p.Lp/2;
mlr = p.mp.*l.*p.r;

tau = p.km.*(u - p.km.*th_d)./p.Rm;

m11 = Jr + p.mp.*p.r.^2 + Jp.*s.^2;
m12 = -mlr.*c;
m22 = Jp;

c1 = tau - p.br.*th_d - 2*Jp.*s.*c.*th_d.*al_d - mlr.*s.*al_d.^2;
c2 = -p.bp.*al_d + Jp.*s.*c.*th_d.^2 + p.mp.*p.g.*l.*s;

det_m = m11.*m22 - m12.^2;
th_dd = (m22.*c1 - m12.*c2)./det_m;
al_dd = (-m12.*c1 + m11.*c2)./det_m;

dx = [th_d; al_d; th_dd; al_dd];
end
