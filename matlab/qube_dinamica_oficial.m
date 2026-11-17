function dx = qube_dinamica_oficial(x, vm, p)
%QUBE_DINAMICA_OFICIAL Modelo NO LINEAL OFICIAL de Quanser del péndulo rotatorio.
%   dx = qube_dinamica_oficial(x, vm, p)
%   x  : [theta; alpha; theta_dot; alpha_dot]  con alpha = 0 ARRIBA
%   vm : voltaje del motor (V)        p : qube_parametros_oficiales()
%
%   Fuente: Quanser, Concept Review "Pendulum State Space Model"
%   (4_concept_reviews/Modeling/007_pendulum_state_space_model.pdf), ecs. 1.1-1.3,
%   dadas para el péndulo colgando (alpha_d = 0 abajo):
%     (Jr + Jp sin^2 a) th'' + mp l r cos a a'' + 2 Jp sin a cos a th' a'
%                                    - mp l r sin a a'^2 = tau - br th'
%     Jp a'' + mp l r cos a th'' - Jp sin a cos a th'^2 + mp g l sin a = -bp a'
%     tau = km (vm - km th')/Rm,   Jr = mr r^2/3,   Jp = mp Lp^2/3,   l = Lp/2
%   Aquí se usan con alpha = alpha_d - pi (0 arriba), lo que cambia el signo de
%   sin y cos. Mismo código que python/qube_model_oficial.py.
%   Opcional: p.tau_c = fricción seca del brazo (N-m), que no está en el oficial.
th_d = x(3);  al = x(2);  al_d = x(4);
s = sin(al);  c = cos(al);
Jr = p.mr*p.r^2/3;  Jp = p.mp*p.Lp^2/3;  l = p.Lp/2;  mlr = p.mp*l*p.r;

tau = p.km*(vm - p.km*th_d)/p.Rm;

m11 = Jr + Jp*s^2;   m12 = -mlr*c;   m22 = Jp;
c1 = tau - p.br*th_d - 2*Jp*s*c*th_d*al_d - mlr*s*al_d^2;
if isfield(p, 'tau_c'), c1 = c1 - p.tau_c*tanh(th_d/0.1); end
c2 = -p.bp*al_d + Jp*s*c*th_d^2 + p.mp*p.g*l*s;

det_m = m11*m22 - m12^2;
dx = [th_d; al_d; (m22*c1 - m12*c2)/det_m; (-m12*c1 + m11*c2)/det_m];
end
