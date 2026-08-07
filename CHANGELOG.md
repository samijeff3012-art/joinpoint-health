# JoinPoint-Health — Registro de cambios

Este proyecto sigue [Versionado Semántico](https://semver.org/lang/es/).

---

## [2.1.0] — 2026-08

### ADVERTENCIA DE REPRODUCIBILIDAD

**Esta versión cambia resultados numéricos, no solo la implementación.**
Los APC estimados con la v2.0.0 difieren de los de la v2.1.0 porque el modelo
subyacente era incorrecto. En la serie de ejemplo del manual:

| Segmento | v2.0.0 | v2.1.0 |
|---|---|---|
| 2008-2019 | −1,74 % | +0,29 % |
| 2019-2022 | +6,68 % | +20,11 % |

**Los análisis realizados con la v2.0.0 deben reejecutarse.** No se conserva un
modo de compatibilidad, porque la especificación anterior no corresponde a una
regresión joinpoint y reproducirla no tendría uso legítimo.

### Corregido

- **Ajuste continuo mediante splines lineales.** La v2.0.0 ajustaba cada
  segmento con regresiones OLS independientes, lo que produce un modelo
  **discontinuo** en los puntos de quiebre y por tanto no corresponde a la
  regresión joinpoint descrita por Kim et al. (2000), que exige continuidad en
  las uniones. Ahora se estima un único modelo

      log(tasa) = b0 + b1·t + Σ_k d_k · max(0, t − τ_k)

  de modo que los segmentos se unen por construcción en cada quiebre. La
  pendiente del segmento *i* es b1 + Σ_{j<i} d_j. Sobre la serie de ejemplo del
  manual (mortalidad 2008-2022, quiebre en 2019), la v2.0.0 producía un salto de
  13,3 unidades en el quiebre —el 54 % del valor ajustado—; la v2.1.0 verifica
  continuidad con un salto máximo del orden de 10⁻¹⁴.

- **Cada observación se emplea una sola vez.** En la v2.0.0 el año del quiebre
  pertenecía a los dos segmentos adyacentes, de modo que con *k* quiebres se
  ajustaban *n + k* puntos mientras el BIC empleaba *n*.

- **Recuento de parámetros del BIC.** Ahora es 2 + 2k, acorde al modelo
  continuo: dos parámetros de regresión, *k* cambios de pendiente y *k*
  localizaciones de quiebre.

- **Errores estándar del APC.** Se derivan de la matriz de covarianzas conjunta
  mediante el método delta, y no de regresiones separadas. La tabla resumen
  incorpora la columna `SE`.

- **Agregación ponderada.** El nuevo parámetro `weight_col` permite promediar
  las tasas ponderando por población o número de casos cuando hay varias filas
  por año. Sin él, el promedio simple sesga el resultado si los subgrupos tienen
  tamaños distintos; el software ahora emite una advertencia en ese caso.

- **Validación de entradas.** Se rechazan los quiebres fuera del rango de la
  serie, los que dejan segmentos con menos de `min_segment_years` observaciones,
  y las tasas negativas, incompatibles con la transformación logarítmica.

- **`plot_trend()` dibuja el modelo realmente estimado.** En la v2.0.0 la figura
  reajustaba cada segmento por separado para trazarlo, de modo que el gráfico
  podía no coincidir con los APC reportados y mostraba saltos en los quiebres.
  Ahora la curva se evalúa sobre el modelo continuo ajustado.

### Añadido

- `check_continuity()` — verifica analíticamente que los segmentos se unen en
  cada quiebre. Se incluye como hoja adicional en `export_results()`.
- `fitted_values()` — devuelve los valores ajustados por el modelo continuo en
  la escala original.
- `test_joinpoint_health.py` — suite de 17 pruebas que verifican propiedades
  formales del método: continuidad, uso de las observaciones, recuperación de
  parámetros conocidos, localización del quiebre, ausencia de quiebres
  espurios, coherencia entre APC, IC y valor p, ponderación y validación de
  entradas.
- `example_usage.py` — ejemplo ejecutable con datos sintéticos, sin
  dependencias externas.
- `CITATION.cff`, `.zenodo.json`, `LICENSE.txt` y este registro de cambios.

### Sin cambios

- La detección automática de quiebres por BIC conserva su comportamiento: sobre
  las series de prueba selecciona los mismos años que un ajuste continuo con el
  BIC correcto.
- La API pública se mantiene íntegra. `plot_trend()`, `plot_bic()`,
  `plot_map()` y `plot_lifecycle_bars()` siguen disponibles con la misma firma,
  de modo que los scripts existentes no requieren modificación.

---

## [2.0.0] — 2026-05

- Detección automática de quiebres estructurales por BIC.
- Cálculo del APC con intervalo de confianza al 95 % y valor p por segmento.
- Análisis estratificado por cualquier variable categórica y módulo de ciclo
  de vida (`LifecycleAnalyzer`).
- Gráficos de tendencia, curva de selección BIC, barras por grupo etario y
  mapa coroplético mediante GeoJSON.
- Exportación a Excel.
- Registro de Derechos de Autor de Software, INDECOPI (Perú).
