# -*- coding: utf-8 -*-
"""
test_joinpoint_health.py — Suite de verificación de JoinPoint-Health v2.1.0

Cada prueba comprueba una propiedad formal del método joinpoint, no
simplemente que el código se ejecute sin error.

Uso:  python test_joinpoint_health.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joinpoint_health import JoinpointAnalyzer, DataError, __version__

OK, FALLOS = 0, []


def check(nombre, condicion, detalle=""):
    global OK
    if condicion:
        OK += 1
        print(f"  [OK]    {nombre}")
    else:
        FALLOS.append(nombre)
        print(f"  [FALLA] {nombre}  {detalle}")


def serie_ejemplo():
    y = np.arange(2008, 2023)
    r = np.array([30.2, 29.6, 28.9, 28.1, 27.6, 27.0, 26.5, 26.0,
                  25.8, 25.4, 25.2, 25.1, 64.3, 58.7, 32.1])
    return pd.DataFrame({"Anio": y, "Tasa": r})


print(f"JoinPoint-Health v{__version__} — verificación\n")

# ── 1. Continuidad en los quiebres ─────────────────────────────────────────
print("1. Continuidad del modelo en los puntos de unión")
df = serie_ejemplo()
az = JoinpointAnalyzer(df, "Anio", "Tasa", breakpoint_years=[2019]).run()
cont = az.check_continuity(verbose=False)
check("un quiebre: los segmentos se unen", cont["continuous"].all(),
      f"salto máximo {cont['jump'].max():.2e}")

az3 = JoinpointAnalyzer(df, "Anio", "Tasa", breakpoint_years=[2013, 2019]).run()
c3 = az3.check_continuity(verbose=False)
check("dos quiebres: los segmentos se unen", c3["continuous"].all(),
      f"salto máximo {c3['jump'].max():.2e}")

# ── 2. Cada observación se usa una sola vez ────────────────────────────────
print("\n2. Uso de las observaciones")
tot = sum(s["n_years"] for s in az.results_["segments"])
check("el año del quiebre pertenece a un solo tramo de ajuste",
      az.results_["model"].nobs == len(df),
      f"nobs={az.results_['model'].nobs}, serie={len(df)}")
check("los segmentos cubren la serie completa", tot >= len(df))

# ── 3. Recuperación de parámetros conocidos ────────────────────────────────
print("\n3. Recuperación de una tendencia conocida")
t = np.arange(2000, 2021)
b1, b2, tau = -0.03, 0.06, 2012
lr = np.where(t < tau, 4.0 + b1 * (t - 2000),
              4.0 + b1 * (tau - 2000) + b2 * (t - tau))
d = pd.DataFrame({"Anio": t, "Tasa": np.exp(lr)})
a = JoinpointAnalyzer(d, "Anio", "Tasa", breakpoint_years=[tau]).run()
s = a.results_["segments"]
apc1_esp, apc2_esp = (np.exp(b1) - 1) * 100, (np.exp(b2) - 1) * 100
check("APC del primer segmento recuperado", abs(s[0]["apc"] - apc1_esp) < 1e-3,
      f"{s[0]['apc']:.4f} frente a {apc1_esp:.4f}")
check("APC del segundo segmento recuperado", abs(s[1]["apc"] - apc2_esp) < 1e-3,
      f"{s[1]['apc']:.4f} frente a {apc2_esp:.4f}")

# ── 4. Detección del quiebre por BIC ───────────────────────────────────────
print("\n4. Detección automática del quiebre")
rng = np.random.default_rng(11)
aciertos = 0
for _ in range(30):
    tau_v = int(rng.integers(2006, 2016))
    lrr = np.where(t < tau_v, 4.0 - 0.03 * (t - 2000),
                   4.0 - 0.03 * (tau_v - 2000) + 0.07 * (t - tau_v))
    dd = pd.DataFrame({"Anio": t, "Tasa": np.exp(lrr + rng.normal(0, 0.02, len(t)))})
    aa = JoinpointAnalyzer(dd, "Anio", "Tasa", max_breakpoints=2).run()
    if aa.breakpoints_ and abs(aa.breakpoints_[0] - tau_v) <= 1:
        aciertos += 1
check("el quiebre se localiza con error de a lo sumo un año en 30 series",
      aciertos >= 27, f"{aciertos}/30")

# ── 5. Sin quiebre cuando la tendencia es lineal ───────────────────────────
print("\n5. Ausencia de quiebres espurios")
lin = pd.DataFrame({"Anio": t, "Tasa": np.exp(4.0 - 0.03 * (t - 2000)
                                              + rng.normal(0, 0.01, len(t)))})
al = JoinpointAnalyzer(lin, "Anio", "Tasa", max_breakpoints=3).run()
check("una serie log-lineal no genera quiebres", len(al.breakpoints_) == 0,
      f"detectó {al.breakpoints_}")

# ── 6. Coherencia entre APC, IC y valor p ──────────────────────────────────
print("\n6. Coherencia interna de las estimaciones")
tab = az.summary_table()
coh = True
for _, f in tab.iterrows():
    lo, hi = [float(x) for x in f["IC 95%"].strip("[]").split(",")]
    if not (lo <= f["APC (%)"] <= hi):
        coh = False
    sig = f["p-value"] < 0.05
    excl = (lo > 0) or (hi < 0)
    if sig != excl:
        coh = False
check("el APC cae dentro de su IC y la significación concuerda con el IC", coh)
check("se reporta el error estándar del APC", "SE" in tab.columns and tab["SE"].notna().all())

# ── 7. Ponderación por tamaño poblacional ──────────────────────────────────
print("\n7. Agregación cuando hay varias filas por año")
filas = []
for yy in range(2010, 2021):
    filas.append(dict(Anio=yy, Region="A", Tasa=10.0, Pob=1000))
    filas.append(dict(Anio=yy, Region="B", Tasa=90.0, Pob=10))
dw = pd.DataFrame(filas)
sin_p = JoinpointAnalyzer(dw, "Anio", "Tasa").run().fitted_values()["observed"].iloc[0]
con_p = JoinpointAnalyzer(dw, "Anio", "Tasa", weight_col="Pob").run() \
    .fitted_values()["observed"].iloc[0]
esperado = (10 * 1000 + 90 * 10) / 1010
check("weight_col produce el promedio ponderado correcto",
      abs(con_p - esperado) < 1e-9, f"{con_p:.4f} frente a {esperado:.4f}")
check("sin weight_col se conserva el promedio simple", abs(sin_p - 50.0) < 1e-9)

# ── 8. Validación de entradas ──────────────────────────────────────────────
print("\n8. Validación de entradas")
try:
    JoinpointAnalyzer(df, "Anio", "Tasa", breakpoint_years=[2050]).run()
    check("rechaza quiebres fuera del rango", False)
except DataError:
    check("rechaza quiebres fuera del rango", True)

try:
    JoinpointAnalyzer(df, "Anio", "Tasa", breakpoint_years=[2009]).run()
    check("rechaza quiebres que dejan segmentos demasiado cortos", False)
except DataError:
    check("rechaza quiebres que dejan segmentos demasiado cortos", True)

neg = df.copy(); neg.loc[0, "Tasa"] = -5.0
try:
    JoinpointAnalyzer(neg, "Anio", "Tasa")
    check("rechaza tasas negativas", False)
except DataError:
    check("rechaza tasas negativas", True)

# ── 9. Análisis estratificado ──────────────────────────────────────────────
print("\n9. Análisis estratificado")
est = pd.concat([df.assign(Region="LIMA"), df.assign(Region="PUNO", Tasa=df.Tasa * 1.4)])
ae = JoinpointAnalyzer(est, "Anio", "Tasa", group_col="Region",
                       breakpoint_years=[2019]).run()
ce = ae.check_continuity(verbose=False)
check("continuidad verificada en todos los estratos", ce["continuous"].all())
check("la tabla resumen incluye ambos estratos",
      set(ae.summary_table()["Group"]) == {"LIMA", "PUNO"})

print("\n" + "=" * 60)
print(f"{OK} pruebas superadas, {len(FALLOS)} fallidas")
if FALLOS:
    for f in FALLOS:
        print("  - " + f)
    sys.exit(1)
print("Todas las propiedades formales se verifican correctamente.")
