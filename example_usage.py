# -*- coding: utf-8 -*-
"""
JoinPoint-Health v2.1.0 — Ejemplo de uso con datos sintéticos
=============================================================
Este archivo se ejecuta tal cual, sin datos externos:

    python example_usage.py

Construye series de tiempo ficticias con un quiebre conocido y recorre las
funciones principales del módulo: ajuste con quiebre fijado, detección
automática por BIC, verificación de continuidad, análisis estratificado,
ciclo de vida, agregación ponderada y exportación.
"""
import numpy as np
import pandas as pd

from joinpoint_health import (JoinpointAnalyzer, LifecycleAnalyzer,
                              analyze_health_trend, __version__)

print(f"JoinPoint-Health v{__version__}\n")

# ── 1. Serie con un quiebre conocido ──────────────────────────────────────
anios = np.arange(2008, 2023)
tasas = np.array([30.2, 29.6, 28.9, 28.1, 27.6, 27.0, 26.5, 26.0,
                  25.8, 25.4, 25.2, 25.1, 64.3, 58.7, 32.1])
df = pd.DataFrame({"Anio": anios, "Tasa": tasas})

# ── 2. Ajuste con el quiebre fijado por el investigador ───────────────────
az = JoinpointAnalyzer(df, "Anio", "Tasa", breakpoint_years=[2019]).run()

print("── Tabla resumen (quiebre fijado en 2019) ──")
print(az.summary_table().to_string(index=False), "\n")

# ── 3. Verificación de continuidad ────────────────────────────────────────
# Es la comprobación que distingue a la v2.1.0: los segmentos deben unirse
# exactamente en cada quiebre.
print("── Continuidad en los quiebres ──")
print(az.check_continuity().to_string(index=False), "\n")

# ── 4. Valores ajustados por el modelo continuo ───────────────────────────
print("── Primeras filas de fitted_values() ──")
print(az.fitted_values().head().to_string(index=False), "\n")

# ── 5. Detección automática del quiebre por BIC ───────────────────────────
auto = JoinpointAnalyzer(df, "Anio", "Tasa", max_breakpoints=3).run()
print(f"── Detección automática: quiebres en {auto.breakpoints_} ──")
print(auto.bic_table_.to_string(index=False), "\n")

# ── 6. Recuperación de una tendencia conocida ─────────────────────────────
# Serie construida con pendientes exactas; el software debe recuperarlas.
t = np.arange(2000, 2021)
b1, b2, tau = -0.03, 0.06, 2012
log_r = np.where(t < tau, 4.0 + b1 * (t - 2000),
                 4.0 + b1 * (tau - 2000) + b2 * (t - tau))
conocida = pd.DataFrame({"Anio": t, "Tasa": np.exp(log_r)})
rec = JoinpointAnalyzer(conocida, "Anio", "Tasa", breakpoint_years=[tau]).run()
esperado = [(np.exp(b1) - 1) * 100, (np.exp(b2) - 1) * 100]
obtenido = [s["apc"] for s in rec.results_["segments"]]
print("── Recuperación de parámetros conocidos ──")
for i, (e, o) in enumerate(zip(esperado, obtenido), 1):
    print(f"  segmento {i}: esperado {e:+.4f} %   obtenido {o:+.4f} %")
print()

# ── 7. Agregación ponderada por población ─────────────────────────────────
# Dos subgrupos de tamaño muy distinto dentro de cada año.
filas = []
for anio in range(2010, 2021):
    filas.append(dict(Anio=anio, Region="A", Tasa=10.0, Poblacion=1000))
    filas.append(dict(Anio=anio, Region="B", Tasa=90.0, Poblacion=10))
dw = pd.DataFrame(filas)

simple = JoinpointAnalyzer(dw, "Anio", "Tasa").run()
ponder = JoinpointAnalyzer(dw, "Anio", "Tasa", weight_col="Poblacion").run()
print("── Efecto de la ponderación por exposición ──")
print(f"  promedio simple      : {simple.fitted_values()['observed'].iloc[0]:.4f}")
print(f"  promedio ponderado   : {ponder.fitted_values()['observed'].iloc[0]:.4f}")
print(f"  valor correcto       : {(10*1000 + 90*10) / 1010:.4f}\n")

# ── 8. Análisis estratificado ─────────────────────────────────────────────
estrat = pd.concat([df.assign(Region="LIMA"),
                    df.assign(Region="PUNO", Tasa=df.Tasa * 1.4)])
ae = JoinpointAnalyzer(estrat, "Anio", "Tasa", group_col="Region",
                       breakpoint_years=[2019]).run()
print("── Análisis estratificado por región ──")
print(ae.summary_table().to_string(index=False), "\n")

# ── 9. Análisis por ciclo de vida ─────────────────────────────────────────
etapas = []
for etapa, factor in [("Children", 0.6), ("Adolescents", 0.8),
                      ("Adults", 1.0), ("Older Adults", 1.7)]:
    etapas.append(df.assign(Etapa=etapa, Tasa=df.Tasa * factor))
lc = LifecycleAnalyzer(pd.concat(etapas), "Anio", "Tasa", "Etapa",
                       breakpoint_years=[2019]).run()
print("── Análisis por ciclo de vida ──")
print(lc.summary_table().to_string(index=False), "\n")

# ── 10. Gráficos y exportación ────────────────────────────────────────────
# Descomente para generar las figuras:
# az.plot_trend()                  # curva continua ajustada y quiebres
# auto.plot_bic()                  # curva de selección del modelo
# lc.plot_lifecycle_bars()         # APC por grupo etario, con IC 95 %
# ae.plot_map()                    # mapa coroplético (requiere geopandas)

az.export_results("resultados_ejemplo.xlsx")
print("Exportado a resultados_ejemplo.xlsx "
      "(hojas: Resumen, BIC, Continuidad)\n")

# ── 11. La misma tubería en una sola llamada ──────────────────────────────
tabla = analyze_health_trend(df, "Anio", "Tasa", breakpoint_years=[2019])
print("── analyze_health_trend() devuelve la tabla directamente ──")
print(tabla.to_string(index=False))
