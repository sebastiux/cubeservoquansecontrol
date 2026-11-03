function ok = tasa_exito(X)
%TASA_EXITO Criterio de éxito por simulación (mismo que python/sim.py).
%   ok = tasa_exito(X)   X: 4 x N x nT   ->  ok: 1xN lógico
%   Éxito = el péndulo nunca pasa de 60° y en el último 10 % del tiempo
%   |alpha| < 3° y |theta| < 15°.
a  = abs(squeeze(X(2,:,:)))*180/pi;   % N x nT
th = abs(squeeze(X(1,:,:)))*180/pi;
if size(X,2) == 1, a = a(:)'; th = th(:)'; end
nT   = size(a, 2);
tail = (nT - floor(nT/10) + 1):nT;
ok = (max(a, [], 2) < 60) & (max(a(:,tail), [], 2) < 3) & (max(th(:,tail), [], 2) < 15);
ok = ok(:)';
end
