function u = nn_controller(x, net)
%NN_CONTROLLER Evalúa la red neuronal entrenada en Python (python/policy.py).
%   u = nn_controller(x, net)
%   x   : 4xN  [theta; alpha; theta_dot; alpha_dot]
%   net : struct cargado con load('../models/nn_qube_policy.mat')
%   u   : 1xN  voltaje (V), saturado suavemente a ±net.Vmax
%
%   g(x) = W3*tanh( W2*tanh( W1*(x./x_scale) + b1 ) + b2 ) + b3
%   u    = Vmax * tanh( (g(x) - g(-x))/2 )      % red impar: u(-x) = -u(x)

xs = x ./ net.x_scale(:);
u  = net.Vmax * tanh(0.5*(mlp(xs, net) - mlp(-xs, net)));
end

function h = mlp(h, net)
nL = double(net.n_layers);
for i = 1:nL
    W = net.(sprintf('W%d', i));
    b = net.(sprintf('b%d', i));
    h = W*h + b(:);
    if i < nL
        h = tanh(h);
    end
end
end
