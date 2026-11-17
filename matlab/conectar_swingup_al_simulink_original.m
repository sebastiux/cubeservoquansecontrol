function nuevo = conectar_swingup_al_simulink_original(modelo_original)
%CONECTAR_SWINGUP_AL_SIMULINK_ORIGINAL Pone la red DQN de SWING-UP en tu modelo
%   de Quanser: el péndulo se levanta solo y se mantiene arriba.
%
%   conectar_swingup_al_simulink_original()
%   conectar_swingup_al_simulink_original('C:\...\qs3_lqr_ctrl_simulinkFInal.slx')
%
%   Solo necesita MATLAB/Simulink y dos archivos:
%     dqn_swingup_v4_qube.mat             (la red entrenada; junto a este script)
%     qs3_lqr_ctrl_simulinkFInal.slx   (tu modelo original; no se modifica)
%
%   Crea una copia  qs3_dqn_swingup_v4_ctrl.slx  junto a tu modelo. En la copia:
%     Qube With Pendulum (X) -> ZOH (2 ms) -> Controlador DQN swing-up
%                            -> Enable Balance Control Switch -> Qube
%   * Se eliminan la ganancia K y el bloque Sum (ya no hay LQR).
%   * El switch ya no depende de |alpha| <= 15°: su control pasa a la constante
%     "Habilitar control" (1 = red activa, 0 = 0 V). Sirve de paro manual.
%   * El bloque carga la red con coder.load('dqn_swingup_v4_qube.mat'), así que no
%     hay parámetros que configurar.

aqui = fileparts(mfilename('fullpath'));
NOMBRE_MAT = 'dqn_swingup_v4_qube.mat';
ORIGINAL   = 'qs3_lqr_ctrl_simulinkFInal.slx';

% ---------------------------------------------------------- localizar archivos
archivo_mat = primero_que_exista({fullfile(aqui, NOMBRE_MAT), ...
    fullfile(aqui, '..', 'models', NOMBRE_MAT), fullfile(pwd, NOMBRE_MAT), which(NOMBRE_MAT)});
if isempty(archivo_mat)
    error('swingup:mat', 'No encuentro %s. Ponlo en la misma carpeta que este script:\n   %s', NOMBRE_MAT, aqui);
end
if nargin < 1 || isempty(modelo_original)
    modelo_original = primero_que_exista({fullfile(aqui, ORIGINAL), ...
        fullfile(aqui, '..', 'simulink', ORIGINAL), fullfile(pwd, ORIGINAL), which(ORIGINAL)});
    if isempty(modelo_original)
        [f, d] = uigetfile('*.slx', 'Selecciona tu modelo original de Quanser (qs3_lqr_ctrl_simulinkFInal.slx)');
        if isequal(f, 0), error('swingup:slx', 'No se seleccionó el modelo original.'); end
        modelo_original = fullfile(d, f);
    end
end
if exist(modelo_original, 'file') ~= 2
    error('swingup:slx', 'No encuentro el modelo original:\n   %s', modelo_original);
end
S = load(archivo_mat);
fprintf('Red:              %s\nModelo original:  %s\n', archivo_mat, modelo_original);

% ---------------------------------------------------------- copia del modelo
carpeta = fileparts(modelo_original);
if isempty(carpeta), carpeta = pwd; end
nuevo   = 'qs3_dqn_swingup_v4_ctrl';
destino = fullfile(carpeta, [nuevo '.slx']);
if bdIsLoaded(nuevo), close_system(nuevo, 0); end
copyfile(modelo_original, destino, 'f');
if exist(fullfile(carpeta, NOMBRE_MAT), 'file') ~= 2      % la red, junto al modelo
    copyfile(archivo_mat, fullfile(carpeta, NOMBRE_MAT), 'f');
end
addpath(carpeta);
load_system(destino);

% ---------------------------------------------------------- bloques existentes
qube = unico(nuevo, 'Name', 'Qube With Pendulum');
gain = unico(nuevo, 'BlockType', 'Gain', 'Gain', 'K');
sumb = unico(nuevo, 'BlockType', 'Sum');
sw   = unico(nuevo, 'BlockType', 'MultiPortSwitch');
ph_qube = get_param(qube, 'PortHandles');
ph_gain = get_param(gain, 'PortHandles');
ph_sum  = get_param(sumb, 'PortHandles');
ph_sw   = get_param(sw,   'PortHandles');
pos     = get_param(gain, 'Position');           % la red va donde estaba K

% ---------------------------------------------------------- quitar LQR y la condición de 15°
% (se borra solo la rama que llega a cada puerto; el resto de la señal se conserva)
borrar_linea_de_entrada(ph_sum.Inport(1));       % Ground -> Sum
borrar_linea_de_entrada(ph_sum.Inport(2));       % X -> Sum
borrar_linea_de_entrada(ph_gain.Inport(1));      % Sum -> K
borrar_linea_de_entrada(ph_sw.Inport(3));        % K -> switch (dato "1")
borrar_linea_de_entrada(ph_sw.Inport(1));        % |alpha|<=15° -> control del switch
delete_block(gain);
delete_block(sumb);

% ---------------------------------------------------------- bloques nuevos
x0 = pos(1); y0 = pos(2);
blk = @(n) [nuevo '/' n];
add_block('simulink/Discrete/Zero-Order Hold', blk('ZOH 2 ms'), ...
          'SampleTime', num2str(S.red_swingup.Ts), 'Position', [x0-170 y0 x0-130 y0+30]);
add_block('simulink/User-Defined Functions/MATLAB Function', blk('Controlador DQN swing-up'), ...
          'Position', [x0-100 y0-5 x0+40 y0+35]);
poner_codigo(blk('Controlador DQN swing-up'), codigo_del_bloque(aqui, S, NOMBRE_MAT));
add_block('simulink/Sources/Constant', blk('Habilitar control'), 'Value', '1', ...
          'Position', [x0-100 y0-70 x0-60 y0-50]);

ph_zoh = get_param(blk('ZOH 2 ms'), 'PortHandles');
ph_ctl = get_param(blk('Controlador DQN swing-up'), 'PortHandles');
ph_hab = get_param(blk('Habilitar control'), 'PortHandles');
add_line(nuevo, ph_qube.Outport(1), ph_zoh.Inport(1), 'autorouting', 'on');   % X
add_line(nuevo, ph_zoh.Outport(1),  ph_ctl.Inport(1), 'autorouting', 'on');
add_line(nuevo, ph_ctl.Outport(1),  ph_sw.Inport(3),  'autorouting', 'on');   % dato "1"
add_line(nuevo, ph_hab.Outport(1),  ph_sw.Inport(1),  'autorouting', 'on');   % control

% ---------------------------------------------------------- InitFcn
% Carpeta del modelo en el path (para coder.load) y ganancias de tus observadores
init = get_param(nuevo, 'InitFcn');
cmd = sprintf(['addpath(fileparts(get_param(''%s'', ''FileName'')));\n' ...
    'if ~exist(''L'',''var'') || ~exist(''beta'',''var'') || ~exist(''m'',''var'') || ' ...
    '~isnumeric(L) || ~isnumeric(beta) || ~isnumeric(m)\n' ...
    '    a_obs_ = 100; beta = 3*a_obs_; L = 3*a_obs_^2; m = a_obs_^3 - L*beta; clear a_obs_\nend'], nuevo);
if isempty(strfind(init, 'a_obs_')) %#ok<STREMP>
    set_param(nuevo, 'InitFcn', strtrim(sprintf('%s\n%s', init, cmd)));
end

save_system(nuevo, destino);
open_system(nuevo);
fprintf(['\nListo: %s\n' ...
         '  1) Péndulo colgando y QUIETO, brazo centrado y libre de obstáculos.\n' ...
         '  2) QUARC -> Monitor & Tune.  La red levanta el péndulo sola.\n' ...
         '  3) Paro: Stop, o pon "Habilitar control" en 0.\n'], destino);
end

% =====================================================================
function f = primero_que_exista(lista)
f = '';
for k = 1:numel(lista)
    if ~isempty(lista{k}) && exist(lista{k}, 'file') == 2, f = lista{k}; return; end
end
end

function b = unico(mdl, varargin)
b = find_system(mdl, 'SearchDepth', 1, varargin{:});
if numel(b) ~= 1
    error('swingup:bloque', 'Esperaba 1 bloque con %s en %s y encontré %d. ¿Es el modelo original?', ...
          strjoin(cellfun(@num2str, varargin, 'UniformOutput', false), ' '), mdl, numel(b));
end
b = b{1};
end

function borrar_linea_de_entrada(puerto)
l = get_param(puerto, 'Line');
if l ~= -1, delete_line(l); end
end

function codigo = codigo_del_bloque(aqui, S, nombre_mat)
% Función principal del bloque (carga la red como constante) + el código de
% dqn_swingup_block como función local.
archivo = fullfile(aqui, 'dqn_swingup_block.m');
if exist(archivo, 'file') == 2
    cuerpo = fileread(archivo);
elseif isfield(S, 'codigo_bloque')
    cuerpo = S.codigo_bloque;
    if size(cuerpo, 1) > 1, cuerpo = cuerpo'; cuerpo = cuerpo(:)'; end
else
    error('swingup:codigo', 'No encuentro dqn_swingup_block.m ni codigo_bloque dentro del .mat.');
end
cabecera = sprintf(['function u = controlador_dqn(x)\n' ...
    '%%#codegen\n' ...
    '%% Red DQN de swing-up + balance. La red se carga al compilar.\n' ...
    'S = coder.load(''%s'', ''red_swingup'');\n' ...
    'u = dqn_swingup_block(x, S.red_swingup);\n' ...
    'end\n\n'], nombre_mat);
codigo = [cabecera, cuerpo];
end

function poner_codigo(bloque, codigo)
% Escribe el código en el bloque MATLAB Function (API nueva o Stateflow)
try
    cfg = get_param(bloque, 'MATLABFunctionConfiguration');
    cfg.FunctionScript = codigo;
catch
    rt = sfroot;
    ch = rt.find('-isa', 'Stateflow.EMChart', 'Path', bloque);
    ch.Script = codigo;
end
end
