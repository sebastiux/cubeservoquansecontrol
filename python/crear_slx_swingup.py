"""
Genera simulink/qs3_dqn_swingup_v4_ctrl.slx a partir del modelo original de
Quanser (qs3_lqr_ctrl_simulinkFInal.slx) SIN usar Simulink: un .slx es un
zip de archivos XML y aquí se editan directamente.

Cambios respecto al original:
  * El bloque "MATLAB Function" (antes: condición |alpha| <= 15°) pasa a ser
    el controlador DQN de swing-up + balance. Carga la red con
    coder.load('dqn_swingup_v4_qube.mat') y recibe la señal X completa.
  * Se eliminan: ganancia "u = -K*x", Sum, switch de balance y constante 0 V.
  * Se añade un Zero-Order Hold de 2 ms entre X y el controlador.
  * Las ganancias L, beta, m de los observadores quedan escritas como números
    (a = 100), así el modelo no necesita variables en el workspace.
  * Dos bloques To Workspace (datos_x, datos_u) registran a 2 ms la señal X y
    el voltaje, para calibrar el modelo con mediciones del Qube real.

Uso:
    python crear_slx_swingup.py
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from xml.dom import minidom
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = ROOT / "simulink" / "qs3_lqr_ctrl_simulinkFInal.slx"
DESTINO = ROOT / "simulink" / "qs3_dqn_swingup_v4_ctrl.slx"
BLOQUE = ROOT / "matlab" / "dqn_swingup_block.m"
NOMBRE_MAT = "dqn_swingup_v4_qube.mat"

SID_ZOH, SID_LOGX, SID_LOGU = 302, 303, 304      # SID nuevos (el original llega a 301)


def quitar_bloque(s: str, sid: int) -> str:
    patron = re.compile(r'  <Block BlockType="[^"]+" Name="[^"]*" SID="%d">.*?</Block>\n' % sid, re.S)
    s2, n = patron.subn("", s, count=1)
    assert n == 1, f"no se encontró el bloque SID {sid}"
    return s2


def quitar_linea(s: str, src: str) -> str:
    patron = re.compile(r'  <Line>\n(?:(?!  </Line>\n).)*?<P Name="Src">%s</P>.*?  </Line>\n'
                        % re.escape(src), re.S)
    s2, n = patron.subn("", s, count=1)
    assert n == 1, f"no se encontró la línea con origen {src}"
    return s2


def reemplazar(s: str, viejo: str, nuevo: str, veces: int = 1) -> str:
    assert s.count(viejo) == veces, f"se esperaba {veces} vez(es): {viejo[:60]!r} ({s.count(viejo)})"
    return s.replace(viejo, nuevo)


def editar_raiz(s: str) -> str:
    s = reemplazar(s, '<P Name="SIDHighWatermark">301</P>', f'<P Name="SIDHighWatermark">{SID_LOGU}</P>')

    # --- fuera LQR, switch de balance y constante 0 V
    for sid in (201, 199, 164, 202):
        s = quitar_bloque(s, sid)
    for src in ("202#out:1", "199#out:1", "164#out:1", "201#out:1", "221#out:1", "213#out:1"):
        s = quitar_linea(s, src)

    # --- Ground: ya solo alimenta al Mux
    s = reemplazar(s, """    <P Name="Src">301#out:1</P>
    <P Name="Points">[14, 0]</P>
    <Branch>
      <P Name="ZOrder">414</P>
      <P Name="Points">[0, -199; 528, 0; 0, 144]</P>
      <P Name="Dst">166#in:1</P>
    </Branch>
    <Branch>
      <P Name="ZOrder">413</P>
      <P Name="Dst">199#in:1</P>
    </Branch>
""", """    <P Name="Src">301#out:1</P>
    <P Name="Points">[14, 0; 0, -199; 528, 0; 0, 144]</P>
    <P Name="Dst">166#in:1</P>
""")

    # --- X del Qube: la rama que iba al Sum va ahora al ZOH
    s = reemplazar(s, """      <P Name="Points">[0, 128; -405, 0; 0, -93]</P>
      <P Name="Dst">199#in:2</P>""", f"""      <P Name="Points">[0, 128; -440, 0; 0, -103]</P>
      <P Name="Dst">{SID_ZOH}#in:1</P>""")

    # --- el bloque MATLAB Function se mueve a donde estaba la ganancia K
    s = reemplazar(s, """    <P Name="Position">[220, 287, 290, 333]</P>
    <P Name="ZOrder">4484</P>
    <P Name="BlockMirror">on</P>
""", """    <P Name="Position">[0, 417, 110, 463]</P>
    <P Name="ZOrder">4484</P>
""")

    nuevo = f"""  <Block BlockType="ZeroOrderHold" Name="ZOH 2 ms" SID="{SID_ZOH}">
    <P Name="Position">[-70, 425, -40, 455]</P>
    <P Name="ZOrder">4495</P>
    <P Name="SampleTime">0.002</P>
  </Block>
  <Block BlockType="ToWorkspace" Name="datos_x" SID="{SID_LOGX}">
    <P Name="Position">[0, 540, 70, 570]</P>
    <P Name="ZOrder">4496</P>
    <P Name="VariableName">datos_x</P>
    <P Name="MaxDataPoints">inf</P>
    <P Name="SampleTime">0.002</P>
    <P Name="SaveFormat">Structure With Time</P>
  </Block>
  <Block BlockType="ToWorkspace" Name="datos_u" SID="{SID_LOGU}">
    <P Name="Position">[185, 540, 255, 570]</P>
    <P Name="ZOrder">4497</P>
    <P Name="VariableName">datos_u</P>
    <P Name="MaxDataPoints">inf</P>
    <P Name="SampleTime">0.002</P>
    <P Name="SaveFormat">Structure With Time</P>
  </Block>
"""
    lineas = f"""  <Line>
    <P Name="ZOrder">415</P>
    <P Name="Src">{SID_ZOH}#out:1</P>
    <P Name="Points">[20, 0]</P>
    <Branch>
      <P Name="ZOrder">420</P>
      <P Name="Dst">213#in:1</P>
    </Branch>
    <Branch>
      <P Name="ZOrder">421</P>
      <P Name="Points">[0, 115]</P>
      <P Name="Dst">{SID_LOGX}#in:1</P>
    </Branch>
  </Line>
  <Line>
    <P Name="ZOrder">416</P>
    <P Name="Src">213#out:1</P>
    <P Name="Points">[45, 0]</P>
    <Branch>
      <P Name="ZOrder">422</P>
      <P Name="Points">[0, -25; 45, 0]</P>
      <Branch>
        <P Name="ZOrder">417</P>
        <P Name="Points">[0, 80]</P>
        <P Name="Dst">200#in:1</P>
      </Branch>
      <Branch>
        <P Name="ZOrder">418</P>
        <P Name="Dst">168#in:1</P>
      </Branch>
    </Branch>
    <Branch>
      <P Name="ZOrder">423</P>
      <P Name="Points">[0, 115]</P>
      <P Name="Dst">{SID_LOGU}#in:1</P>
    </Branch>
  </Line>
  <Line>
    <P Name="ZOrder">419</P>
    <P Name="Src">221#out:1</P>
    <P Name="Dst">224#in:1</P>
  </Line>
"""
    i = s.index("  <Line>\n")
    s = s[:i] + nuevo + s[i:]
    j = s.index("  <Annotation ")
    s = s[:j] + lineas + s[j:]

    # --- textos
    s = reemplazar(s, "Pendulum Control - LQR Balance Control", "Pendulum Control - DQN Swing-Up + Balance")
    s = reemplazar(s, """    <P Name="Name">Balance Control</P>
    <P Name="Position">[-14, 407, 59, 422]</P>""", """    <P Name="Name">Red DQN: swing-up + balance</P>
    <P Name="Position">[-20, 396, 140, 411]</P>""")
    return s


def editar_chart(s: str) -> str:
    cuerpo = BLOQUE.read_text(encoding="utf-8")
    assert all(ord(c) < 128 for c in cuerpo), "dqn_swingup_block.m debe ser ASCII"
    script = (
        "function y = fcn(alpha)\n"
        "%#codegen\n"
        "% Controlador DQN: swing-up + balance del pendulo.\n"
        "% La entrada (se llama \"alpha\" por compatibilidad con el bloque original)\n"
        "% es la senal X completa de \"Qube With Pendulum\" (4x1). Salida: voltaje.\n"
        f"% La red se carga al compilar desde {NOMBRE_MAT} (misma carpeta).\n"
        f"S = coder.load('{NOMBRE_MAT}', 'red_swingup');\n"
        "y = dqn_swingup_block(alpha, S.red_swingup);\n"
        "end\n\n" + cuerpo.rstrip("\n") + "\n"
    )
    patron = re.compile(r'<P Name="script">.*?</P>', re.S)
    assert len(patron.findall(s)) == 1
    return patron.sub(lambda _: f'<P Name="script">{escape(script)}</P>', s)


def editar_observador(s: str) -> str:
    """L, beta, m (a = 100) como números: beta = 3a, L = 3a^2, m = a^3 - L*beta."""
    s = reemplazar(s, '<P Name="Gain">L</P>', '<P Name="Gain">30000</P>')
    s = reemplazar(s, '<P Name="Gain">beta</P>', '<P Name="Gain">300</P>')
    s = reemplazar(s, '<P Name="Gain">m</P>', '<P Name="Gain">-8000000</P>')
    return s


EDICIONES = {
    "simulink/systems/system_root.xml": editar_raiz,
    "simulink/stateflow/chart_7.xml": editar_chart,
    "simulink/systems/system_279.xml": editar_observador,
    "simulink/systems/system_291.xml": editar_observador,
}


def verificar(archivos: dict[str, bytes]):
    """Comprobaciones de consistencia (XML bien formado, líneas y puertos)."""
    for nombre, datos in archivos.items():
        if nombre.endswith(".xml") or nombre.endswith(".rels"):
            minidom.parseString(datos)
    raiz = archivos["simulink/systems/system_root.xml"].decode("utf-8")
    sids = set(re.findall(r'<Block BlockType="[^"]+" Name="[^"]*" SID="(\d+)">', raiz))
    usados_in, usados_out = {}, {}
    for ref in re.findall(r'<P Name="Dst">(\d+)#in:(\d+)</P>', raiz):
        assert ref[0] in sids, f"destino a bloque inexistente {ref}"
        assert ref not in usados_in, f"entrada conectada dos veces {ref}"
        usados_in[ref] = True
    for ref in re.findall(r'<P Name="Src">(\d+)#out:(\d+)</P>', raiz):
        assert ref[0] in sids, f"origen en bloque inexistente {ref}"
        assert ref not in usados_out, f"salida con dos líneas {ref}"
        usados_out[ref] = True
    for requerido in [("213", "1"), ("168", "1"), ("200", "1"), (str(SID_ZOH), "1"), ("224", "1"),
                      (str(SID_LOGX), "1"), (str(SID_LOGU), "1")]:
        assert requerido in usados_in, f"entrada sin conectar {requerido}"
    assert "K</P>" not in raiz
    return len(sids), len(usados_in)


def main():
    with zipfile.ZipFile(ORIGINAL) as zin:
        infos = zin.infolist()
        archivos = {i.filename: zin.read(i.filename) for i in infos}
    for nombre, fn in EDICIONES.items():
        archivos[nombre] = fn(archivos[nombre].decode("utf-8")).encode("utf-8")
    n_bloques, n_conex = verificar(archivos)
    with zipfile.ZipFile(DESTINO, "w", zipfile.ZIP_DEFLATED) as zout:
        for i in infos:                           # mismo orden y metadatos que el original
            zout.writestr(i, archivos[i.filename], compress_type=i.compress_type)
    print(f"Creado {DESTINO}  ({n_bloques} bloques en la raíz, {n_conex} entradas conectadas)")


if __name__ == "__main__":
    sys.exit(main())
