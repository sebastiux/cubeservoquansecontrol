function nuevo = conectar_al_simulink_original(modelo_original)
%CONECTAR_AL_SIMULINK_ORIGINAL Pone la red DQN en tu modelo de Quanser (sin LQR).
%
%   conectar_al_simulink_original()
%   conectar_al_simulink_original('ruta/qs3_lqr_ctrl_simulinkFInal.slx')
%
%   NO modifica tu modelo: crea una copia  qs3_dqn_ctrl.slx  en la misma
%   carpeta. En la copia:
%
%     ANTES:  Qube (X) -> Sum (0 - X) -> u = -K*x -> Enable Balance Control Switch
%     AHORA:  Qube (X) -> ZOH (Ts_agente) -> Controlador DQN -> Rate Transition
%                                                -> Enable Balance Control Switch
%
%   * Se eliminan la ganancia K y el bloque Sum: ya no se usa LQR.
%   * La red toma X DIRECTAMENTE de "Qube With Pendulum" (sin el signo menos
%     del Sum: con -X el lazo quedaría con el signo invertido).
%   * Se conservan tu switch de balance (|alpha| <= 15°), los observadores,
%     scopes y la configuración de QUARC.
%   * La InitFcn del modelo ejecuta cargar_red_dqn (red_dqn, Ts_agente y las
%     ganancias L, beta, m de tus observadores si no existen).

% ===================== CONFIGURA AQUÍ TU RUTA (opcional) =====================
% Carpeta del proyecto (la que contiene matlab/, models/ y simulink/), o la
% carpeta donde copiaste todo junto con instalar_en_carpeta.
% Ejemplo: RUTA_PROYECTO = 'C:\Users\csohs\OneDrive\Documents\MATLAB';
% Déjala vacía ('') para que el script la busque solo.
RUTA_PROYECTO = '';
% ==============================================================================
here = ubicar_carpeta_matlab(mfilename('fullpath'), RUTA_PROYECTO);   % carpeta matlab/ del proyecto
if nargin < 1 || isempty(modelo_original)
    modelo_original = fullfile(here, '..', 'simulink', 'qs3_lqr_ctrl_simulinkFInal.slx');
    if exist(modelo_original, 'file') ~= 2                  % todo junto en una carpeta
        modelo_original = fullfile(here, 'qs3_lqr_ctrl_simulinkFInal.slx');
    end
end
if exist(modelo_original, 'file') ~= 2
    error('cubeservo:modelo', ['No encuentro tu modelo original:\n   %s\n' ...
          'Pásalo como argumento: conectar_al_simulink_original(''C:\\...\\qs3_lqr_ctrl_simulinkFInal.slx'')'], ...
          modelo_original);
end
addpath(here);
evalin('base', 'cargar_red_dqn');

% ---------------------------------------------------------- copia del modelo
carpeta = fileparts(modelo_original);
nuevo = 'qs3_dqn_ctrl';
destino = fullfile(carpeta, [nuevo '.slx']);
if bdIsLoaded(nuevo), close_system(nuevo, 0); end
copyfile(modelo_original, destino, 'f');
load_system(destino);

% ---------------------------------------------------------- bloques existentes
qube = find_unico(nuevo, 'Name', 'Qube With Pendulum');
gain = find_unico(nuevo, 'BlockType', 'Gain', 'Gain', 'K');
sumb = find_unico(nuevo, 'BlockType', 'Sum');
sw   = find_unico(nuevo, 'BlockType', 'MultiPortSwitch');
gnd  = find_unico(nuevo, 'BlockType', 'Ground');

ph_qube = get_param(qube, 'PortHandles');
ph_gain = get_param(gain, 'PortHandles');
ph_sum  = get_param(sumb, 'PortHandles');
ph_sw   = get_param(sw,   'PortHandles');
pos     = get_param(gain, 'Position');           % la red va donde estaba K

% ---------------------------------------------------------- quitar el LQR
% Ground y X tienen ramas hacia otros bloques: se borra solo la rama al Sum
nm = @(b) get_param(b, 'Name');
delete_line(nuevo, [nm(gnd) '/1'],  [nm(sumb) '/1']);        % Ground -> Sum
delete_line(nuevo, [nm(qube) '/1'], [nm(sumb) '/2']);        % X -> Sum
delete_line(get_param(ph_sum.Outport(1),  'Line'));          % Sum -> K
delete_line(get_param(ph_gain.Outport(1), 'Line'));          % K -> switch
delete_block(gain);
delete_block(sumb);

% ---------------------------------------------------------- red DQN
x0 = pos(1); y0 = pos(2);
blk = @(n) [nuevo '/' n];
add_block('simulink/Discrete/Zero-Order Hold', blk('ZOH agente'), ...
          'SampleTime', 'Ts_agente', 'Position', [x0-170 y0 x0-130 y0+30]);
add_block('simulink/User-Defined Functions/MATLAB Function', blk('Controlador DQN'), ...
          'Position', [x0-100 y0-5 x0+20 y0+35]);
set_mfunction(blk('Controlador DQN'), fullfile(here, 'dqn_controller_block.m'), {'red_dqn'});
add_block('simulink/Signal Attributes/Rate Transition', blk('Rate Transition DQN'), ...
          'OutPortSampleTime', get_param(nuevo, 'FixedStep'), ...
          'Position', [x0+45 y0 x0+85 y0+30]);

ph_zoh = get_param(blk('ZOH agente'), 'PortHandles');
ph_ctl = get_param(blk('Controlador DQN'), 'PortHandles');
ph_rt  = get_param(blk('Rate Transition DQN'), 'PortHandles');

L = @(a, b) add_line(nuevo, a, b, 'autorouting', 'on');
L(ph_qube.Outport(1), ph_zoh.Inport(1));      % X (sin signo menos)
L(ph_zoh.Outport(1),  ph_ctl.Inport(1));
L(ph_ctl.Outport(1),  ph_rt.Inport(1));
L(ph_rt.Outport(1),   ph_sw.Inport(3));       % dato "1" del switch de balance

% ---------------------------------------------------------- InitFcn
init = get_param(nuevo, 'InitFcn');
cmd  = sprintf(['d_ = fullfile(fileparts(get_param(''%s'', ''FileName'')), ''..'', ''matlab'');\n' ...
                'if ~exist(fullfile(d_, ''cargar_red_dqn.m''), ''file''), d_ = ''%s''; end\n' ...
                'addpath(d_); clear d_\ncargar_red_dqn;'], nuevo, strrep(here, '''', ''''''));
if ~contains(init, 'cargar_red_dqn')
    set_param(nuevo, 'InitFcn', strtrim(sprintf('%s\n%s', init, cmd)));
end

save_system(nuevo, destino);
open_system(nuevo);
fprintf('Listo: %s\n  LQR eliminado; "Controlador DQN" conectado al switch de balance.\n', destino);
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

function d = ubicar_carpeta_matlab(archivo, ruta_usuario)
% Devuelve la carpeta matlab/ del proyecto (la que contiene cargar_red_dqn.m
% y, al lado, models/dqn_qube.mat).
%   ruta_usuario: RUTA_PROYECTO definida arriba ('' = buscar sola)
base = fileparts(archivo);
if ~isempty(ruta_usuario)
    cand = {ruta_usuario, fullfile(ruta_usuario, 'matlab')};
else
    cand = {base, fullfile(base, 'matlab'), fullfile(base, '..', 'matlab'), ...
            fullfile(base, 'cubeservoquansecontrol', 'matlab'), pwd, fullfile(pwd, 'matlab')};
    w = which('cargar_red_dqn');
    if ~isempty(w), cand{end+1} = fileparts(w); end
end
for k = 1:numel(cand)
    if es_carpeta_matlab(cand{k}), d = cand{k}; return; end
end
% No se encontró: pedirla al usuario (solo si no fijó RUTA_PROYECTO)
if isempty(ruta_usuario) && usejava('desktop')
    sel = uigetdir(pwd, 'Selecciona la carpeta del proyecto cubeservoquansecontrol (o su carpeta matlab)');
    if ischar(sel)
        for c = {sel, fullfile(sel, 'matlab')}
            if es_carpeta_matlab(c{1}), d = c{1}; return; end
        end
        cand = [cand, {sel, fullfile(sel, 'matlab')}];
    end
end
error('cubeservo:rutas', '%s', diagnostico_rutas(archivo, cand));
end

function ok = es_carpeta_matlab(d)
ok = exist(fullfile(d, 'cargar_red_dqn.m'), 'file') == 2 && ~isempty(buscar_red(d));
end

function f = buscar_red(d)
% Ubicación de dqn_qube.mat: proyecto completo (../models) o todo junto en una carpeta
f = '';
for c = {fullfile(d, '..', 'models', 'dqn_qube.mat'), fullfile(d, 'models', 'dqn_qube.mat'), fullfile(d, 'dqn_qube.mat')}
    if exist(c{1}, 'file') == 2, f = c{1}; return; end
end
end


function msg = diagnostico_rutas(archivo, cand)
msg = sprintf(['No encuentro los archivos del proyecto.\n' ...
               'Este script está en:   %s\n' ...
               'Carpeta actual (pwd):  %s\n' ...
               'Busqué en (cargar_red_dqn.m | dqn_qube.mat):\n'], archivo, pwd);
for k = 1:numel(cand)
    msg = [msg, sprintf('   %-60s  %s | %s\n', cand{k}, ...
           si_no(exist(fullfile(cand{k}, 'cargar_red_dqn.m'), 'file') == 2), ...
           si_no(~isempty(buscar_red(cand{k}))))]; %#ok<AGROW>
end
msg = [msg, sprintf(['Solución: escribe en RUTA_PROYECTO (al inicio de este archivo) la carpeta\n' ...
                     'del proyecto (la que contiene matlab/, models/ y simulink/), o la carpeta donde\n' ...
                     'copiaste todo junto con instalar_en_carpeta (p. ej. Documents\\MATLAB).'])];
end

function s = si_no(b)
if b, s = 'sí'; else, s = 'NO'; end
end
