function nuevo = conectar_al_simulink_original(tipo, modelo_original)
%CONECTAR_AL_SIMULINK_ORIGINAL Inserta el controlador RL en tu Simulink del Qube.
%
%   conectar_al_simulink_original()                 % DQN (por defecto)
%   conectar_al_simulink_original('qlearning')      % Q-learning tabular
%   conectar_al_simulink_original('dqn', 'ruta/qs3_lqr_ctrl_simulinkFInal.slx')
%
%   NO modifica tu modelo: crea una copia  qs3_dqn_ctrl.slx / qs3_qlearning_ctrl.slx
%   en la misma carpeta. Cambios en la copia:
%
%     Qube With Pendulum (X) --> ZOH (Ts_agente) --> Controlador RL --> Rate Transition --+
%                                                                                         |
%     u = -K*x (LQR original) -----------------------------------------------------+     |
%                                                                                  v     v
%                                         Selector LQR/RL (usar_rl) --> Enable Balance Control Switch (dato "1")
%
%   * El controlador RL toma X DIRECTAMENTE del Qube. (Tu ganancia K recibe -X
%     a través del bloque Sum = 0 - X; si la red tomara esa señal, el signo
%     quedaría invertido.)
%   * Se conserva tu switch |alpha| <= 15° y el LQR para comparar:
%     usar_rl = 1 -> RL, usar_rl = 0 -> LQR (se puede cambiar en External Mode).
%   * La InitFcn del modelo carga red_dqn / tabla_q / Ts_agente desde models/.
%
%   Antes de correr en el Qube: ejecuta ProyectoControlFinal.m (K, L, beta, m
%   de tus observadores), igual que con el modelo original.

if nargin < 1 || isempty(tipo), tipo = 'dqn'; end
tipo = lower(tipo);
here = fileparts(mfilename('fullpath'));
if nargin < 2 || isempty(modelo_original)
    modelo_original = fullfile(here, '..', 'simulink', 'qs3_lqr_ctrl_simulinkFInal.slx');
end
switch tipo
    case 'dqn'
        bloque = 'Controlador DQN';   codigo = 'dqn_controller_block.m';        param = 'red_dqn';
    case {'qlearning', 'q'}
        tipo = 'qlearning';
        bloque = 'Controlador Q-learning'; codigo = 'qlearning_controller_block.m'; param = 'tabla_q';
    otherwise
        error('tipo debe ser ''dqn'' o ''qlearning''');
end

addpath(here);
evalin('base', 'cargar_controlador_rl');
if ~evalin('base', sprintf('exist(''%s'', ''var'')', param))
    error('No se encontró %s: entrena primero (python/train_%s.py).', param, tipo);
end

% ---------------------------------------------------------- copia del modelo
[carpeta, ~] = fileparts(modelo_original);
nuevo = ['qs3_' tipo '_ctrl'];
destino = fullfile(carpeta, [nuevo '.slx']);
if bdIsLoaded(nuevo), close_system(nuevo, 0); end
copyfile(modelo_original, destino, 'f');
load_system(destino);

% ---------------------------------------------------------- bloques existentes
qube = find_unico(nuevo, 'Name', 'Qube With Pendulum');
gain = find_unico(nuevo, 'BlockType', 'Gain', 'Gain', 'K');
sw   = find_unico(nuevo, 'BlockType', 'MultiPortSwitch');

ph_gain = get_param(gain, 'PortHandles');
ph_sw   = get_param(sw,   'PortHandles');
ph_qube = get_param(qube, 'PortHandles');

% desconectar  LQR -> switch (entrada de datos "1" = puerto 3 del switch)
l = get_param(ph_gain.Outport(1), 'Line');
if l ~= -1, delete_line(l); end

% ---------------------------------------------------------- bloques nuevos
pg = get_param(gain, 'Position');           % [x1 y1 x2 y2]
x0 = pg(1); y0 = pg(4) + 80;
blk = @(n) [nuevo '/' n];

add_block('simulink/Discrete/Zero-Order Hold', blk('ZOH agente'), ...
          'SampleTime', 'Ts_agente', 'Position', [x0-140 y0 x0-100 y0+30]);
add_block('simulink/User-Defined Functions/MATLAB Function', blk(bloque), ...
          'Position', [x0-60 y0-5 x0+70 y0+35]);
set_mfunction(blk(bloque), fullfile(here, codigo), {param});
add_block('simulink/Signal Attributes/Rate Transition', blk('Rate Transition RL'), ...
          'Position', [x0+100 y0 x0+140 y0+30]);
add_block('simulink/Sources/Constant', blk('usar_rl'), 'Value', 'usar_rl', ...
          'Position', [x0+90 y0-50 x0+130 y0-30]);
add_block('simulink/Signal Routing/Switch', blk('Selector LQR//RL'), ...
          'Criteria', 'u2 ~= 0', 'Position', [x0+170 pg(2)-10 x0+200 y0+35]);

ph_zoh = get_param(blk('ZOH agente'), 'PortHandles');
ph_ctl = get_param(blk(bloque), 'PortHandles');
ph_rt  = get_param(blk('Rate Transition RL'), 'PortHandles');
ph_c   = get_param(blk('usar_rl'), 'PortHandles');
ph_sel = get_param(blk('Selector LQR//RL'), 'PortHandles');

L = @(a, b) add_line(nuevo, a, b, 'autorouting', 'on');
L(ph_qube.Outport(1), ph_zoh.Inport(1));      % X (sin el signo menos)
L(ph_zoh.Outport(1),  ph_ctl.Inport(1));
L(ph_ctl.Outport(1),  ph_rt.Inport(1));
L(ph_rt.Outport(1),   ph_sel.Inport(1));      % u1: RL
L(ph_c.Outport(1),    ph_sel.Inport(2));      % u2: usar_rl
L(ph_gain.Outport(1), ph_sel.Inport(3));      % u3: LQR original
L(ph_sel.Outport(1),  ph_sw.Inport(3));       % -> switch de balance (dato "1")

% ---------------------------------------------------------- callbacks
init = get_param(nuevo, 'InitFcn');
cmd  = sprintf('addpath(''%s''); cargar_controlador_rl;', here);
if ~contains(init, 'cargar_controlador_rl')
    set_param(nuevo, 'InitFcn', strtrim(sprintf('%s\n%s', init, cmd)));
end

save_system(nuevo, destino);
open_system(nuevo);
fprintf('Listo: %s\n  Bloque "%s" conectado. usar_rl = 1 (RL) / 0 (LQR).\n', destino, bloque);
end

% =====================================================================
function b = find_unico(mdl, varargin)
b = find_system(mdl, 'SearchDepth', 1, varargin{:});
if numel(b) ~= 1
    error('Se esperaba 1 bloque con %s en %s y se encontraron %d.', ...
          strjoin(cellfun(@num2str, varargin, 'UniformOutput', false), ' '), mdl, numel(b));
end
b = b{1};
end

function set_mfunction(block_path, src_file, param_names)
% Pone el código en el bloque MATLAB Function y declara sus parámetros.
rt = sfroot;
ch = rt.find('-isa', 'Stateflow.EMChart', 'Path', block_path);
ch.Script = fileread(src_file);
for k = 1:numel(param_names)
    d = ch.find('-isa', 'Stateflow.Data', 'Name', param_names{k});
    d.Scope = 'Parameter';
end
end
