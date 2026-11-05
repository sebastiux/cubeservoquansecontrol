function construir_entorno_pruebas(simular)
%CONSTRUIR_ENTORNO_PRUEBAS Crea simulink/qube_dqn_pruebas.slx (NO necesita QUARC).
%
%   construir_entorno_pruebas()        % crea el modelo y lo simula
%   construir_entorno_pruebas(false)   % solo lo crea
%
%   Misma estructura que tu modelo de hardware, pero "Qube With Pendulum"
%   se sustituye por un Qube SIMULADO (modelo no lineal + encoders + tus
%   observadores), así puedes probar la red sin licencia de Quanser:
%
%   +--------------------- Qube simulado (sin QUARC) ---------------------+
%   | u -> Dinámica no lineal -> Integrador -> encoders -> observadores -> X |
%   +----------------------------------------------------------------------+
%        ^                                                              |
%        |                                                              v
%   Balance habilitado (|alpha|<=15°) <- Rate Transition <- Controlador DQN <- ZOH (Ts_agente)
%
%   Cambia la condición inicial con  x0_sim = [theta; alpha; theta_dot; alpha_dot]
%   (radianes) en el workspace antes de dar Run.

if nargin < 1, simular = true; end
here = ubicar_carpeta_matlab(mfilename('fullpath'));   % carpeta matlab/ del proyecto
addpath(here);
mdl = 'qube_dqn_pruebas';
carpeta_slx = fullfile(here, '..', 'simulink');
if ~exist(carpeta_slx, 'dir'), mkdir(carpeta_slx); end
destino = fullfile(carpeta_slx, [mdl '.slx']);

evalin('base', 'cargar_red_dqn; variables_entorno_pruebas;');

if bdIsLoaded(mdl), close_system(mdl, 0); end
if exist(destino, 'file'), delete(destino); end
new_system(mdl);
open_system(mdl);
init = sprintf(['d_ = fullfile(fileparts(get_param(''%s'', ''FileName'')), ''..'', ''matlab'');\n' ...
                'if ~exist(fullfile(d_, ''cargar_red_dqn.m''), ''file''), d_ = ''%s''; end\n' ...
                'addpath(d_); clear d_\n' ...
                'cargar_red_dqn;\nvariables_entorno_pruebas;'], mdl, strrep(here, '''', ''''''));
set_param(mdl, 'Solver', 'ode4', 'FixedStep', 'Ts', 'StopTime', '10', ...
          'ReturnWorkspaceOutputs', 'on', 'InitFcn', init);

% ============================================================ Qube simulado
sub = [mdl '/Qube simulado (sin QUARC)'];
add_block('simulink/Ports & Subsystems/Subsystem', sub, 'Position', [520 180 680 260]);
delete_line(sub, 'In1/1', 'Out1/1');
set_param([sub '/In1'],  'Name', 'u (V)', 'Position', [20 93 50 107]);
set_param([sub '/Out1'], 'Name', 'X', 'Position', [900 93 930 107]);
add_block('simulink/Sinks/Out1', [sub '/x real'], 'Position', [900 193 930 207]);

add_block('simulink/User-Defined Functions/MATLAB Function', [sub '/Dinamica no lineal'], ...
          'Position', [100 70 220 130]);
set_mfunction([sub '/Dinamica no lineal'], fullfile(here, 'planta_qube_block.m'), {'qp'});
add_block('simulink/Continuous/Integrator', [sub '/Integrador'], ...
          'InitialCondition', 'x0_sim', 'Position', [270 85 300 115]);
add_block('simulink/Math Operations/Gain', [sub '/Angulos'], ...
          'Gain', '[eye(2) zeros(2)]', 'Multiplication', 'Matrix(K*u)', 'Position', [350 80 410 120]);
add_block('simulink/Discontinuities/Quantizer', [sub '/Encoders 2048 cuentas'], ...
          'QuantizationInterval', 'q_enc', 'Position', [450 80 510 120]);
add_block('simulink/Continuous/State-Space', [sub '/Observadores (State X)'], ...
          'A', 'A_obs', 'B', 'B_obs', 'C', 'C_obs', 'D', 'zeros(2)', 'Position', [580 150 680 190]);
try
    set_param([sub '/Observadores (State X)'], 'InitialCondition', 'IC_obs');
catch
    set_param([sub '/Observadores (State X)'], 'X0', 'IC_obs');
end
add_block('simulink/Signal Routing/Mux', [sub '/Mux X'], 'Inputs', '2', 'Position', [780 70 785 130]);

S = @(a, b) add_line(sub, a, b, 'autorouting', 'on');
S('u (V)/1', 'Dinamica no lineal/2');
S('Dinamica no lineal/1', 'Integrador/1');
S('Integrador/1', 'Dinamica no lineal/1');
S('Integrador/1', 'Angulos/1');
S('Integrador/1', 'x real/1');
S('Angulos/1', 'Encoders 2048 cuentas/1');
S('Encoders 2048 cuentas/1', 'Mux X/1');
S('Encoders 2048 cuentas/1', 'Observadores (State X)/1');
S('Observadores (State X)/1', 'Mux X/2');
S('Mux X/1', 'X/1');

% ============================================================ control
add_block('simulink/Discrete/Zero-Order Hold', [mdl '/ZOH agente'], ...
          'SampleTime', 'Ts_agente', 'Position', [760 300 800 330]);
add_block('simulink/User-Defined Functions/MATLAB Function', [mdl '/Controlador DQN'], ...
          'Position', [600 290 720 340]);
set_mfunction([mdl '/Controlador DQN'], fullfile(here, 'dqn_controller_block.m'), {'red_dqn'});
add_block('simulink/Signal Attributes/Rate Transition', [mdl '/Rate Transition'], ...
          'OutPortSampleTime', 'Ts', 'Position', [500 300 540 330]);
set_param([mdl '/ZOH agente'], 'Orientation', 'left');
set_param([mdl '/Controlador DQN'], 'Orientation', 'left');
set_param([mdl '/Rate Transition'], 'Orientation', 'left');

% habilitación como en tu modelo: control solo si |alpha| <= 15°
add_block('simulink/Math Operations/Gain', [mdl '/alpha (deg)'], ...
          'Gain', '[0 180/pi 0 0]', 'Multiplication', 'Matrix(K*u)', 'Position', [760 380 820 410]);
add_block('simulink/Math Operations/Abs', [mdl '/Abs'], 'Position', [680 380 710 410]);
add_block('simulink/Logic and Bit Operations/Compare To Constant', [mdl '/alpha menor a 15 grados'], ...
          'relop', '<=', 'const', '15', 'Position', [580 380 630 410]);
set_param([mdl '/alpha (deg)'], 'Orientation', 'left');
set_param([mdl '/Abs'], 'Orientation', 'left');
set_param([mdl '/alpha menor a 15 grados'], 'Orientation', 'left');
add_block('simulink/Sources/Constant', [mdl '/0 V'], 'Value', '0', 'Position', [380 440 410 460]);
add_block('simulink/Signal Routing/Switch', [mdl '/Balance habilitado'], ...
          'Criteria', 'u2 ~= 0', 'Position', [380 200 420 260]);
set_param([mdl '/Balance habilitado'], 'Orientation', 'left');

% ============================================================ visualización
add_block('simulink/Math Operations/Gain', [mdl '/rad2deg'], 'Gain', '180/pi', 'Position', [760 470 800 500]);
add_block('simulink/Sinks/Scope', [mdl '/theta alpha (deg)'], 'Position', [840 465 870 505]);
add_block('simulink/Sinks/Scope', [mdl '/Vm (V)'], 'Position', [380 100 410 130]);
add_block('simulink/Sinks/To Workspace', [mdl '/x_log'], 'VariableName', 'x_log', ...
          'SaveFormat', 'Timeseries', 'Position', [840 530 900 560]);
add_block('simulink/Sinks/To Workspace', [mdl '/u_log'], 'VariableName', 'u_log', ...
          'SaveFormat', 'Timeseries', 'Position', [380 30 440 60]);

T = @(a, b) add_line(mdl, a, b, 'autorouting', 'on');
T('Qube simulado (sin QUARC)/1', 'ZOH agente/1');
T('ZOH agente/1', 'Controlador DQN/1');
T('Controlador DQN/1', 'Rate Transition/1');
T('Rate Transition/1', 'Balance habilitado/1');
T('Qube simulado (sin QUARC)/1', 'alpha (deg)/1');
T('alpha (deg)/1', 'Abs/1');
T('Abs/1', 'alpha menor a 15 grados/1');
T('alpha menor a 15 grados/1', 'Balance habilitado/2');
T('0 V/1', 'Balance habilitado/3');
T('Balance habilitado/1', 'Qube simulado (sin QUARC)/1');
T('Balance habilitado/1', 'Vm (V)/1');
T('Balance habilitado/1', 'u_log/1');
T('Qube simulado (sin QUARC)/2', 'rad2deg/1');
T('rad2deg/1', 'theta alpha (deg)/1');
T('Qube simulado (sin QUARC)/2', 'x_log/1');

save_system(mdl, destino);
fprintf('Entorno de pruebas creado: %s\n', destino);

if simular
    out = sim(mdl);
    t = out.x_log.Time;
    x = squeeze(out.x_log.Data);
    if size(x, 1) ~= 4, x = x.'; end
    tu = out.u_log.Time;  u = squeeze(out.u_log.Data);
    figure('Name', 'Entorno de pruebas: DQN en el Qube simulado');
    subplot(3,1,1); plot(t, x(2,:)*180/pi, 'b'); ylabel('\alpha (deg)'); grid on;
    title('qube\_dqn\_pruebas (sin QUARC)');
    subplot(3,1,2); plot(t, x(1,:)*180/pi, 'b'); ylabel('\theta (deg)'); grid on;
    subplot(3,1,3); stairs(tu, u, 'b'); ylabel('V_m (V)'); xlabel('t (s)'); grid on;
    cola = t > 0.9*t(end);
    fprintf('Último 10%% de la simulación: |alpha|max = %.2f°, |theta|max = %.2f°\n', ...
            max(abs(x(2,cola)))*180/pi, max(abs(x(1,cola)))*180/pi);
end
end

% =====================================================================
function set_mfunction(block_path, src_file, param_names)
% Pone el código de src_file en un bloque MATLAB Function y declara
% param_names como parámetros (se leen del workspace).
rt = sfroot;
ch = rt.find('-isa', 'Stateflow.EMChart', 'Path', block_path);
ch.Script = fileread(src_file);
for k = 1:numel(param_names)
    d = ch.find('-isa', 'Stateflow.Data', 'Name', param_names{k});
    d.Scope = 'Parameter';
end
end

function d = ubicar_carpeta_matlab(archivo)
% Devuelve la carpeta matlab/ del proyecto (la que contiene cargar_red_dqn.m
% y, al lado, models/dqn_qube.mat), aunque este archivo se haya copiado o
% renombrado en otra carpeta.
base = fileparts(archivo);
cand = {base, fullfile(base, 'matlab'), fullfile(base, '..', 'matlab'), ...
        fullfile(base, 'cubeservoquansecontrol', 'matlab'), pwd, fullfile(pwd, 'matlab')};
w = which('cargar_red_dqn');
if ~isempty(w), cand{end+1} = fileparts(w); end
for k = 1:numel(cand)
    if es_carpeta_matlab(cand{k}), d = cand{k}; return; end
end
% No se encontró: pedirla al usuario
if usejava('desktop')
    sel = uigetdir(pwd, 'Selecciona la carpeta del proyecto cubeservoquansecontrol (o su carpeta matlab)');
    if ischar(sel)
        for c = {sel, fullfile(sel, 'matlab')}
            if es_carpeta_matlab(c{1}), d = c{1}; return; end
        end
    end
end
error('cubeservo:rutas', '%s', sprintf(['No encuentro los archivos del proyecto (cargar_red_dqn.m y models/dqn_qube.mat).\n' ...
       'Ejecuta el script desde la carpeta "matlab" del repositorio cubeservoquansecontrol\n' ...
       '(no copies el archivo suelto a otra carpeta), o agrega esa carpeta al path:\n' ...
       '   addpath(''C:/ruta/cubeservoquansecontrol/matlab'')']));
end

function ok = es_carpeta_matlab(d)
ok = exist(fullfile(d, 'cargar_red_dqn.m'), 'file') == 2 && ...
     exist(fullfile(d, '..', 'models', 'dqn_qube.mat'), 'file') == 2;
end
