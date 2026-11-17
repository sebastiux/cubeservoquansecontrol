function instalar_en_carpeta(destino)
%INSTALAR_EN_CARPETA Copia todo lo necesario a UNA sola carpeta (sin subcarpetas).
%
%   instalar_en_carpeta('C:\Users\csohs\OneDrive\Documents\MATLAB')
%   instalar_en_carpeta()          % por defecto: tu carpeta userpath
%                                  % (normalmente Documents\MATLAB)
%
%   Ejecútalo UNA vez desde la carpeta matlab/ del proyecto descargado.
%   Copia a "destino":
%     - todos los .m de matlab/ (scripts, bloques y auxiliares)
%     - models/*.mat   (redes entrenadas: dqn_qube.mat, dqn_swingup_qube.mat)
%     - simulink/qs3_lqr_ctrl_simulinkFInal.slx (tu modelo original de Quanser)
%   Después ya puedes ejecutar todo directamente desde esa carpeta:
%     prueba_dqn_modelo_no_lineal
%     construir_entorno_pruebas
%     conectar_al_simulink_original

origen = fileparts(mfilename('fullpath'));
raiz   = fullfile(origen, '..');
if nargin < 1 || isempty(destino)
    destino = strtok(userpath, pathsep);
end
if ~exist(destino, 'dir'), mkdir(destino); end

archivos = {};
lista = dir(fullfile(origen, '*.m'));
for k = 1:numel(lista)
    archivos{end+1} = fullfile(origen, lista(k).name); %#ok<AGROW>
end
lista = dir(fullfile(raiz, 'models', '*.mat'));           % redes entrenadas
for k = 1:numel(lista)
    archivos{end+1} = fullfile(raiz, 'models', lista(k).name); %#ok<AGROW>
end
archivos{end+1} = fullfile(raiz, 'simulink', 'qs3_lqr_ctrl_simulinkFInal.slx');

fprintf('Copiando a %s\n', destino);
for k = 1:numel(archivos)
    [~, nom, ext] = fileparts(archivos{k});
    if exist(archivos{k}, 'file') ~= 2
        fprintf('   (no encontrado, se omite) %s\n', archivos{k});
        continue
    end
    aviso = '';
    if exist(fullfile(destino, [nom ext]), 'file') == 2, aviso = '  (reemplazado)'; end
    copyfile(archivos{k}, fullfile(destino, [nom ext]), 'f');
    fprintf('   %s%s%s\n', nom, ext, aviso);
end
addpath(destino);
fprintf(['\nListo. Ahora puedes ejecutar desde %s:\n' ...
         '   prueba_swingup_modelo_no_lineal          (swing-up, solo MATLAB)\n' ...
         '   conectar_swingup_al_simulink_original    (swing-up en el Qube real)\n' ...
         '   prueba_dqn_modelo_no_lineal, construir_entorno_pruebas, conectar_al_simulink_original (solo balance)\n'], destino);
end
