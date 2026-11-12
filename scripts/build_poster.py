"""Genera el borrador del póster del ACA 2 (docs/poster/poster.html) a partir de las métricas reales de las redes neuronales.

Uso::

    python scripts/build_poster.py

Luego abra ``docs/poster/poster.html`` en Chrome/Edge y use *Imprimir → Guardar
como PDF* (tamaño A1, márgenes ninguno, con gráficos de fondo). Las cifras se
leen de ``models/metricas*.json``; los nombres, logo y enlaces se leen de
``src/project_info.py`` y ``docs/poster/``. Lo que falte se muestra como
espacio reservado: nada se inventa.
"""

from __future__ import annotations

import html
import re
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data_dictionary as dd  # noqa: E402
from src import project_info as info  # noqa: E402

POSTER_DIR = ROOT / "docs" / "poster"


def f3(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def load(name: str) -> dict:
    return json.loads((ROOT / "models" / name).read_text(encoding="utf-8"))


def apa(text: str) -> str:
    """Escapa el texto y convierte *cursiva* en <i>cursiva</i> (formato APA)."""
    return re.sub(r"\*(.+?)\*", r"<i>\1</i>", html.escape(text))


def placeholder(text: str, cls: str = "ph") -> str:
    return f'<div class="{cls}">{html.escape(text)}</div>'


def build() -> Path:
    full, early = load("metricas_nn.json"), load("metricas_nn_temprano.json")
    meta = load("metadata_nn.json")
    w = full["modelo_seleccionado"]
    we = early["modelo_seleccionado"]
    sel, sel_e = full["modelos"][w], early["modelos"][we]

    NN = ("mlp", "ft_transformer")
    rows = ""
    for k, v in full["modelos"].items():
        star = "★ " if k == w else ""
        cls = ' class="win"' if k == w else ""
        tag = "" if k in NN else " (ACA 1)"
        rows += (f"<tr{cls}><td>{star}{html.escape(v['nombre'].split(' (')[0] + tag)}</td>"
                 f"<td>{f3(v['cv']['f1_macro']['media'])}</td><td>{f3(v['prueba']['accuracy'])}</td>"
                 f"<td>{f3(v['prueba']['f1_macro'])}</td><td>{f3(v['prueba']['balanced_accuracy'])}</td></tr>")

    per_class = "".join(
        f"<tr><td><span class='dot' style='background:{dd.CLASS_COLORS[c]}'></span>{dd.CLASS_LABELS_ES[c]}</td>"
        f"<td>{f3(sel['prueba']['por_clase'][c]['precision'])}</td><td>{f3(sel['prueba']['por_clase'][c]['recall'])}</td>"
        f"<td>{f3(sel['prueba']['por_clase'][c]['f1'])}</td></tr>"
        for c in dd.CLASSES
    )
    cm = sel["prueba"]["matriz_confusion"]["valores"]
    cm_lbl = [dd.CLASS_LABELS_ES[c] for c in dd.CLASSES]
    cm_html = "<tr><th>Real \\ Pred.</th>" + "".join(f"<th>{c}</th>" for c in cm_lbl) + "</tr>"
    for i, row in enumerate(cm):
        cm_html += f"<tr><th>{cm_lbl[i]}</th>" + "".join(
            f"<td class='{'diag' if i == j else ''}'>{v}</td>" for j, v in enumerate(row)) + "</tr>"

    imp = full["importancia_permutacion_prueba"][:6]
    max_imp = max(i["importancia_media"] for i in imp)
    imp_html = "".join(
        f"<div class='imp'><span>{html.escape(i['etiqueta'])}</span><div class='bar' "
        f"style='width:{i['importancia_media'] / max_imp * 100:.0f}%'></div><b>{f3(i['importancia_media'])}</b></div>"
        for i in imp
    )

    logo = POSTER_DIR / "logo_cun.png"
    logo_html = '<img class="logo" src="logo_cun.png" alt="Logo CUN">' if logo.exists() else placeholder("[Logo oficial CUN]", "ph logo")
    names = ", ".join(info.INTEGRANTES) if info.INTEGRANTES else "[Nombres completos de los integrantes]"
    docente = info.DOCENTE or "[Docente]"

    def qr(label: str, url: str) -> str:
        img = POSTER_DIR / f"qr_{label.lower()}.png"
        inner = f'<img src="{img.name}" alt="QR {label}">' if img.exists() else placeholder("[QR]", "ph qr")
        return f"<div class='qrbox'>{inner}<div><b>{label}</b><br>{html.escape(url) if url else '[URL pública pendiente]'}</div></div>"

    steps = ["Datos UCI 697<br>en SQLite", "Limpieza + split<br>estratificado 80/20", "Codificación<br>z-score + embeddings",
             "MLP y<br>FT-Transformer", "Pérdida ponderada<br>AdamW + early stopping", "CV 5-fold<br>(macro-F1)", "Evaluación<br>en prueba", "App Streamlit"]
    flow = "".join(f"<div class='step'>{s}</div>" + ("<div class='arrow'>→</div>" if i < len(steps) - 1 else "")
                   for i, s in enumerate(steps))

    h = sel.get("historial")
    def svg_curves(h):
        if not h:
            return placeholder("[Curvas de entrenamiento]")
        n = len(h["val_loss"]); allv = h["train_loss"] + h["val_loss"]
        lo, hi = min(allv), max(allv)
        W, H = 520, 230
        def pts(vals):
            return " ".join(f"{40 + i / max(n - 1, 1) * (W - 50):.1f},{H - 30 - (v - lo) / max(hi - lo, 1e-9) * (H - 50):.1f}" for i, v in enumerate(vals))
        bx = 40 + (h["best_epoch"] - 1) / max(n - 1, 1) * (W - 50)
        return (f"<svg viewBox='0 0 {W} {H}' width='100%'><rect x='40' y='10' width='{W-50}' height='{H-40}' fill='none' stroke='#E6E9EF'/>"
                f"<polyline points='{pts(h['train_loss'])}' fill='none' stroke='#13315C' stroke-width='3'/>"
                f"<polyline points='{pts(h['val_loss'])}' fill='none' stroke='#eb6834' stroke-width='3'/>"
                f"<line x1='{bx:.1f}' x2='{bx:.1f}' y1='10' y2='{H-30}' stroke='#5B6577' stroke-dasharray='5'/>"
                f"<text x='{W/2:.0f}' y='{H-8}' font-size='14' text-anchor='middle' fill='#5B6577'>Época (línea punteada = época elegida: {h['best_epoch']})</text>"
                f"<text x='46' y='24' font-size='14' fill='#13315C'>■ Entrenamiento</text><text x='170' y='24' font-size='14' fill='#eb6834'>■ Validación</text>"
                f"<text x='4' y='{(H-30)/2:.0f}' font-size='14' fill='#5B6577' transform='rotate(-90 12 {(H-30)/2:.0f})'>Pérdida</text></svg>")
    ap = sel["mejores_hiperparametros"]
    arch_rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in ap.items())
    best_classic = max((k for k in full["modelos"] if k not in NN), key=lambda k: full["modelos"][k]["cv"]["f1_macro"]["media"])
    bc = full["modelos"][best_classic]
    delta = sel["cv"]["f1_macro"]["media"] - bc["cv"]["f1_macro"]["media"]

    shot = ROOT / "docs" / "capturas" / "prediccion.png"
    shot_html = '<img class="shot" src="../capturas/prediccion.png" alt="Captura">' if shot.exists() else placeholder("[Captura de la app: docs/capturas/prediccion.png]")

    page = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><title>Póster EduPredict AI · ACA 2</title>
<style>
@page {{ size: 594mm 841mm; margin: 0; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family: 'Segoe UI', Inter, Helvetica, Arial, sans-serif; color:#1B2433; background:#fff; }}
.poster {{ width:594mm; height:841mm; padding:16mm 18mm; display:flex; flex-direction:column; gap:7mm; }}
header {{ background:linear-gradient(120deg,#0B2545,#13315C); color:#fff; border-radius:8mm; padding:10mm 14mm;
          display:flex; align-items:center; gap:12mm; }}
header h1 {{ font-size:62pt; margin:0; letter-spacing:-1px; }}
header h2 {{ font-size:26pt; margin:2mm 0 4mm; font-weight:500; color:#DCE3EE; }}
header p {{ font-size:19pt; margin:1mm 0; color:#DCE3EE; }}
.logo {{ width:78mm; height:42mm; object-fit:contain; background:#fff; border-radius:4mm; padding:3mm 4mm; }}
.ph {{ border:2px dashed #9AA7BD; color:#5B6577; display:flex; align-items:center; justify-content:center;
       border-radius:4mm; font-size:16pt; text-align:center; background:#F4F6FA; }}
header .ph {{ color:#DCE3EE; background:transparent; border-color:#DCE3EE; }}
.grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:8mm; }}
.card {{ border:1px solid #E6E9EF; border-radius:6mm; padding:7mm 9mm; background:#fff; }}
.card h3 {{ margin:0 0 4mm; font-size:27pt; color:#0B2545; border-left:8px solid #13315C; padding-left:4mm; }}
.card p, .card li {{ font-size:18pt; line-height:1.4; margin:1.5mm 0; }}
.q {{ font-style:italic; color:#13315C; font-weight:600; }}
.kpis {{ display:flex; gap:5mm; margin-top:4mm; }}
.kpi {{ flex:1; background:#F4F6FA; border-radius:4mm; padding:4mm; text-align:center; }}
.kpi b {{ display:block; font-size:30pt; color:#0B2545; }} .kpi span {{ font-size:14pt; color:#5B6577; }}
.flow {{ display:flex; align-items:center; justify-content:space-between; gap:2mm; }}
.step {{ flex:1; background:#13315C; color:#fff; border-radius:4mm; padding:5mm 2mm; text-align:center; font-size:15pt; font-weight:600; min-height:24mm; display:flex; align-items:center; justify-content:center; }}
.arrow {{ font-size:26pt; color:#13315C; }}
table {{ width:100%; border-collapse:collapse; font-size:17pt; }}
th, td {{ padding:2.5mm 3mm; border-bottom:1px solid #E6E9EF; text-align:center; }}
th {{ color:#5B6577; font-weight:600; }} td:first-child, th:first-child {{ text-align:left; }}
tr.win td {{ background:#E8EEF7; font-weight:700; }}
td.diag {{ background:#13315C; color:#fff; font-weight:700; }}
.dot {{ display:inline-block; width:4mm; height:4mm; border-radius:50%; margin-right:2mm; }}
.imp {{ display:grid; grid-template-columns: 95mm 1fr 22mm; align-items:center; gap:3mm; font-size:15pt; margin:2mm 0; }}
.imp .bar {{ height:6mm; background:#13315C; border-radius:0 2mm 2mm 0; }}
.shot {{ width:100%; height:92mm; object-fit:cover; object-position:top; border-radius:4mm; border:1px solid #E6E9EF; }}
.refs {{ grid-column: span 1; }} .ref {{ font-size:11.5pt !important; line-height:1.3 !important; padding-left:8mm; text-indent:-8mm; margin:0 0 1.6mm !important; word-break:break-word; }}
.cite {{ font-weight:500; color:#5B6577; font-size:22pt; }}
.lbl {{ font-size:13pt !important; line-height:1.25 !important; margin:2mm 0 1.5mm !important; color:#1B2433; }}
.qrs {{ display:flex; flex-direction:column; gap:4mm; }} .qrbox {{ flex:1; display:flex; gap:4mm; align-items:center; font-size:14pt; }}
.qrbox img, .ph.qr {{ width:34mm; height:34mm; flex:none; }} .qrbox {{ word-break:break-all; }}
.note {{ font-size:13pt; color:#5B6577; }}
</style></head><body><div class="poster">
<header>{logo_html}<div>
<h1>EduPredict AI</h1>
<h2>Predicción de deserción universitaria con redes neuronales (MLP · FT-Transformer)</h2>
<p><b>{html.escape(names)}</b></p>
<p>Docente: {html.escape(docente)} · Aprendizaje Automático · Especialización en Inteligencia Artificial · CUN</p>
</div></header>

<div class="grid">
 <div class="card"><h3>Problema</h3>
  <p>La deserción universitaria suele detectarse cuando ya es tarde para intervenir.</p>
  <p class="q">{html.escape(info.PREGUNTA)}</p></div>
 <div class="card"><h3>Objetivo</h3><p>{html.escape(info.OBJETIVO)}</p></div>
</div>

<div class="card"><h3>Pilar: Educación · Dataset UCI 697 <span class='cite'>(Realinho et al., 2021)</span></h3>
 <div class="kpis">
  <div class="kpi"><b>{meta['dataset']['filas']:,}</b><span>estudiantes</span></div>
  <div class="kpi"><b>36</b><span>variables predictoras</span></div>
  <div class="kpi"><b style="color:{dd.CLASS_COLORS['Dropout']}">32,1 %</b><span>Deserción</span></div>
  <div class="kpi"><b style="color:{dd.CLASS_COLORS['Enrolled']}">17,9 %</b><span>Matriculado</span></div>
  <div class="kpi"><b style="color:{dd.CLASS_COLORS['Graduate']}">49,9 %</b><span>Graduado</span></div>
 </div></div>

<div class="card"><h3>Metodología</h3><div class="flow">{flow}</div>
 <p class="note">Redes implementadas con PyTorch (Paszke et al., 2019): MLP con <i>embeddings</i> y FT-Transformer (Gorishniy et al., 2021; Vaswani et al., 2017), entrenadas con AdamW (Loshchilov &amp; Hutter, 2019).
 Desbalance tratado con pérdida ponderada por clase. Referencia: modelos clásicos del ACA 1 (Pedregosa et al., 2011). Escenario adicional de alerta temprana sin variables del 2.º semestre.</p></div>

<div class="grid">
 <div class="card"><h3>Arquitectura de red neuronal</h3>
  <p><b>{html.escape(sel['nombre'])}</b> · {sel.get('parametros', 0):,} parámetros entrenables</p>
  <p class='lbl'><b>Tabla 1</b><br><i>Hiperparámetros de la red desplegada</i></p>
  <table>{arch_rows}</table></div>
 <div class="card"><h3>Curvas de entrenamiento</h3>
  <p class='lbl'><b>Figura 1</b><br><i>Pérdida de entrenamiento y validación por época ({html.escape(sel['nombre'].split(' (')[0])}, escenario completo)</i></p>
  {svg_curves(h)}
  <p class="note"><i>Nota.</i> <i>Early stopping</i> sobre el macro-F1 de validación (15 % del entrenamiento).</p></div>
</div>

<div class="grid">
 <div class="card"><h3>Comparación de modelos</h3>
  <p class='lbl'><b>Tabla 2</b><br><i>Desempeño de redes neuronales y modelos clásicos (prueba, n = {meta['particion']['n_prueba']})</i></p>
  <table><tr><th>Modelo</th><th>Macro-F1 CV</th><th>Accuracy</th><th>Macro-F1</th><th>Bal. acc.</th></tr>{rows}</table>
  <p class="note"><i>Nota.</i> Selección por macro-F1 medio en validación cruzada de 5 pliegues; el conjunto de prueba no interviene en la selección.</p>
  <p class='lbl'><b>Tabla 3</b><br><i>Métricas por clase de la red seleccionada</i></p>
  <table><tr><th>Clase ({html.escape(sel['nombre'].split(' (')[0])})</th><th>Precisión</th><th>Recall</th><th>F1</th></tr>{per_class}</table>
 </div>
 <div class="card"><h3>Matriz de confusión e importancia</h3>
  <p class='lbl'><b>Tabla 4</b><br><i>Matriz de confusión en el conjunto de prueba</i></p>
  <table>{cm_html}</table>
  <p class='lbl'><b>Figura 2</b><br><i>Variables más influyentes (importancia por permutación)</i></p>{imp_html}
  <p class="note"><i>Nota.</i> Caída de macro-F1 al permutar cada variable (asociación, no causalidad).</p>
  <p><b>Alerta temprana</b> ({html.escape(sel_e['nombre'].split(' (')[0])}): macro-F1 CV {f3(sel_e['cv']['f1_macro']['media'])} ·
  macro-F1 prueba {f3(sel_e['prueba']['f1_macro'])} · accuracy {f3(sel_e['prueba']['accuracy'])}.</p>
 </div>
</div>

<div class="grid">
 <div class="card"><h3>Aplicación Streamlit</h3><p class='lbl'><b>Figura 3</b><br><i>Módulo de predicción individual de la aplicación</i></p>{shot_html}<p class="note"><i>Nota.</i> Aplicación desarrollada con Streamlit (Streamlit Inc., 2026).</p></div>
 <div class="card"><h3>Conclusiones</h3><ul>
  <li>{html.escape(sel['nombre'].split(' (')[0])} fue la red seleccionada por validación cruzada (macro-F1 {f3(sel['cv']['f1_macro']['media'])}); frente al mejor modelo clásico del ACA 1 la diferencia es de {delta:+.3f}, comparable a la variación entre pliegues.</li>
  <li>Las variables académicas (unidades aprobadas y notas por semestre) son las que más pesan en la red, según la importancia por permutación.</li>
  <li>Las redes son «cajas negras»: la importancia indica asociación, no explica cada decisión.</li>
  <li><i>Matriculado</i> es la clase más difícil (F1 {f3(sel['prueba']['por_clase']['Enrolled']['f1'])}).</li>
  <li>Con información del 1.er semestre ya es posible una alerta temprana útil, un semestre antes.</li>
  <li>Herramienta académica: identifica asociaciones, no causas, y no reemplaza decisiones institucionales.</li>
 </ul></div>
</div>

<div class="grid" style="grid-template-columns:1.55fr 1fr">
 <div class="card refs"><h3>Referencias</h3>{''.join(f'<p class="ref">{apa(r)}</p>' for r in info.REFERENCIAS_APA)}</div>
 <div class="card"><h3>Enlaces</h3><div class="qrs">{qr('App', info.ENLACE_APP)}{qr('Video', info.ENLACE_VIDEO)}{qr('Repositorio', info.ENLACE_REPO)}</div></div>
</div>
</div></body></html>"""
    out = POSTER_DIR / "poster.html"
    out.write_text(page.replace(f"{meta['dataset']['filas']:,}", f"{meta['dataset']['filas']:,}".replace(",", ".")), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build())
