# JoinPoint-Health v2.1.0

> **Herramienta de regresión joinpoint para el análisis de tendencias en indicadores cuantitativos de salud.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20151357.svg)](https://doi.org/10.5281/zenodo.20151357)

---

## ¿Qué es JoinPoint-Health?

**JoinPoint-Health** es un módulo Python de código abierto que implementa la regresión joinpoint (Kim et al., 2000) para detectar cambios de tendencia en series de tiempo de indicadores de salud, y estimar el **cambio porcentual anual (APC)** de cada tramo con su medida de incertidumbre.

Responde a preguntas del tipo: ¿en qué año cambió la tendencia de la mortalidad?, ¿la caída previa se detuvo o se revirtió?, ¿el cambio es estadísticamente distinguible del ruido?

- Detecta **automáticamente** los quiebres estructurales mediante BIC, o los acepta fijados por el investigador cuando existe una justificación teórica o epidemiológica.
- Estima un **modelo continuo** por splines lineales: los segmentos se unen exactamente en cada quiebre, como exige el método.
- Reporta **APC, error estándar, intervalo de confianza al 95 % y valor p** por segmento, derivados de la matriz de covarianzas conjunta mediante el método delta.
- Permite **análisis estratificado** por cualquier variable categórica: región, sexo, grupo etario, tipo de establecimiento.
- Incluye **módulo de ciclo de vida**, **verificación de continuidad** y **mapa coroplético** por GeoJSON.

---

## Novedades de la v2.1.0

Versión correctiva. La v2.0.0 ajustaba cada segmento mediante regresiones OLS independientes, lo que produce un modelo **discontinuo** en los puntos de quiebre y por tanto no corresponde a la regresión joinpoint descrita por Kim et al. (2000), que exige continuidad en las uniones.

- **Ajuste continuo mediante splines lineales.** Se estima un único modelo `log(tasa) = b0 + b1·t + Σ_k d_k · max(0, t − τ_k)`, de modo que los segmentos se unen por construcción. Sobre la serie de ejemplo del manual, la v2.0.0 producía un salto de 13,3 unidades en el quiebre —el 54 % del valor ajustado—; la v2.1.0 verifica continuidad con un salto máximo del orden de 10⁻¹⁴.
- **Cada observación se emplea una sola vez.** Antes, el año del quiebre pertenecía a los dos segmentos adyacentes.
- **Recuento de parámetros del BIC** acorde al modelo continuo: 2 + 2k.
- **Errores estándar del APC** por método delta sobre la matriz de covarianzas conjunta. Nueva columna `SE` en la tabla resumen.
- **Agregación ponderada** con `weight_col`, para cuando hay varias filas por año.
- **Validación de entradas:** quiebres fuera de rango, segmentos demasiado cortos y tasas negativas se rechazan con mensaje explícito.
- **Nuevos métodos:** `check_continuity()` y `fitted_values()`.
- **Suite de 17 pruebas** de propiedades formales del método.

> ### Aviso de reproducibilidad
>
> Esta versión **cambia los APC estimados**. En la serie de ejemplo del manual, el segmento 2008-2019 pasa de −1,74 % a +0,29 % y el segmento 2019-2022 de +6,68 % a +20,11 %. **Los análisis realizados con la v2.0.0 deben reejecutarse.**
>
> No se conserva un modo de compatibilidad: la especificación anterior no corresponde a una regresión joinpoint y reproducirla no tendría uso legítimo. Detalles en [CHANGELOG.md](CHANGELOG.md).

La API pública se mantiene íntegra: `plot_trend()`, `plot_bic()`, `plot_map()` y `plot_lifecycle_bars()` siguen disponibles con la misma firma, de modo que los scripts existentes no requieren modificación.

---

## Autores

| Nombre | Institución |
|--------|-------------|
| Cesar Jefferson Samillan Vasquez | Universidad Nacional Toribio Rodríguez de Mendoza de Amazonas |
| Mercedes Acosta Román | Universidad Nacional Autónoma de Tayacaja Daniel Hernández Morillo |
| Gladys Bernardita León Montoya | Universidad Nacional Toribio Rodríguez de Mendoza de Amazonas |
| Rosa Ysabel Bazán Valque | Universidad Nacional Toribio Rodríguez de Mendoza de Amazonas |

**Contacto:** cesar.samillan@untrm.edu.pe

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/samijeff3012-art/joinpoint-health.git
cd joinpoint-health

# Instalar dependencias
pip install -r requirements.txt

# Dependencia opcional para mapas geográficos
pip install geopandas
```

En **Google Colab**:

```python
!pip install pandas numpy statsmodels matplotlib openpyxl
```

Para ver el módulo en funcionamiento sin datos propios:

```bash
python example_usage.py
```

---

## Inicio rápido

```python
import pandas as pd
from joinpoint_health import JoinpointAnalyzer

# Cargar datos: una columna de años y una del indicador
df = pd.read_excel("mis_datos.xlsx")

# Crear el analizador y ejecutar
az = JoinpointAnalyzer(
    data            = df,
    year_col        = "Anio",
    rate_col        = "Tasa",
    max_breakpoints = 3,      # el BIC elige cuántos usar
)
az.run()

# Resultados
print(az.summary_table())
print(f"Quiebres detectados: {az.breakpoints_}")

# Verificar que el modelo es continuo en los quiebres
az.check_continuity()

# Gráficos
az.plot_trend()
az.plot_bic()

# Exportar
az.export_results("resultados.xlsx")
```

**Salida esperada:**

```
  Group Breakpoints   Segment    Period  APC (%)      SE            IC 95%  p-value  n years
Overall        2019 Segment 1 2008-2019   0.2888  2.0569  [-4.0941, 4.872] 0.890517       12
Overall        2019 Segment 2 2019-2022  20.1079 10.6200 [-0.9387, 45.626] 0.060457        4
```

---

## Quiebres fijados por el investigador

Cuando existe una razón teórica o epidemiológica para situar el quiebre en un año determinado —un cambio normativo, el inicio de una pandemia— puede imponerse en lugar de estimarlo:

```python
az = JoinpointAnalyzer(df, "Anio", "Tasa", breakpoint_years=[2019]).run()
```

El software valida que el quiebre caiga dentro del rango de la serie y que deje segmentos con suficientes observaciones; de lo contrario lanza `DataError` con un mensaje explícito.

---

## Verificación de continuidad

Es la comprobación que distingue a esta versión. Verifica analíticamente que los segmentos se unen en cada quiebre:

```python
az.check_continuity()
```

```
Continuidad verificada en 1 punto(s) de unión (salto máximo 1.42e-14).
  group  breakpoint         jump  continuous
Overall        2019 1.421085e-14        True
```

El resultado se incluye además como hoja propia en `export_results()`.

---

## Agregación ponderada

Cuando el conjunto de datos tiene varias filas por año —subregiones, subgrupos— el promedio simple sobrepondera los grupos pequeños. `weight_col` corrige eso ponderando por la población o el número de casos:

```python
az = JoinpointAnalyzer(df, "Anio", "Tasa", weight_col="Poblacion").run()
```

Sin ese parámetro, el software emite una advertencia si detecta varias filas por año.

---

## Análisis estratificado

```python
az = JoinpointAnalyzer(
    data      = df,
    year_col  = "Anio",
    rate_col  = "Tasa",
    group_col = "Region",     # un análisis independiente por región
).run()

print(az.summary_table())
az.plot_trend()               # un panel por región
```

---

## Análisis por ciclo de vida

```python
from joinpoint_health import LifecycleAnalyzer

lc = LifecycleAnalyzer(
    data          = df,
    year_col      = "Anio",
    rate_col      = "Tasa",
    lifecycle_col = "GrupoEtario",
).run()

lc.plot_lifecycle_bars()      # APC por grupo etario, con IC 95 %
```

---

## Mapa geográfico

```python
# Después de un análisis estratificado por región:
az.plot_map(
    metric  = "apc",
    segment = -1,             # último segmento (posterior al quiebre)
    title   = "APC por departamento",
)
```

Requiere `geopandas`. El GeoJSON predeterminado cubre el Perú a nivel departamental; puede proveerse cualquier otro con `geojson_url`.

---

## Función de conveniencia en una línea

```python
from joinpoint_health import analyze_health_trend

tabla = analyze_health_trend(df, "Anio", "Tasa", breakpoint_years=[2019])
```

---

## Estructura del repositorio

```
joinpoint-health/
├── joinpoint_health.py          ← Módulo principal
├── example_usage.py             ← Ejemplo ejecutable con datos sintéticos
├── test_joinpoint_health.py     ← Suite de 17 pruebas de propiedades
├── requirements.txt             ← Dependencias Python
├── CHANGELOG.md                 ← Registro de cambios entre versiones
├── CITATION.cff                 ← Metadatos de citación
├── .zenodo.json                 ← Metadatos del depósito en Zenodo
├── LICENSE.txt                  ← Licencia MIT
└── README.md                    ← Este archivo
```

---

## Pruebas

```bash
python test_joinpoint_health.py
```

Ejecuta 17 pruebas que verifican **propiedades formales del método**, no simplemente que el código corra: continuidad del modelo en los quiebres, uso de cada observación una sola vez, recuperación de pendientes conocidas, localización del quiebre en series simuladas, ausencia de quiebres espurios en series log-lineales, coherencia entre APC, intervalo y valor p, ponderación y validación de entradas.

---

## Componentes principales

| Componente | Tipo | Descripción |
|---|---|---|
| `JoinpointAnalyzer` | Clase principal | Ajuste continuo, detección de quiebres, APC e inferencia. |
| `LifecycleAnalyzer` | Subclase | Análisis por grupos del ciclo de vida, con orden configurable. |
| `analyze_health_trend()` | Función | Pipeline completa en una línea; devuelve la tabla resumen. |
| `DataError` | Excepción | Error en la estructura o el contenido de los datos de entrada. |

### Métodos de `JoinpointAnalyzer`

| Método | Descripción |
|---|---|
| `run()` | Ejecuta el análisis y llena `results_`, `breakpoints_` y `bic_table_`. |
| `summary_table()` | APC, SE, IC 95 %, valor p y número de años por segmento. |
| `check_continuity()` | Verifica que los segmentos se unan en cada quiebre. |
| `fitted_values()` | Valores ajustados por el modelo continuo, en la escala original. |
| `export_results()` | Exporta a Excel: Resumen, BIC y Continuidad. |
| `plot_trend()` | Tasas observadas, curva continua ajustada y quiebres. |
| `plot_bic()` | Curva de selección del modelo por BIC. |
| `plot_map()` | Mapa coroplético del APC por región. |

---

## Parámetros principales

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `data` | DataFrame | Requerido | Datos de entrada. |
| `year_col` | str | Requerido | Columna de años. |
| `rate_col` | str | Requerido | Columna del indicador. Debe ser positiva. |
| `weight_col` | str | None | Columna de población o casos, para agregar varias filas por año. |
| `breakpoint_years` | list | None | Quiebres fijados por el investigador. Si es None, se detectan por BIC. |
| `max_breakpoints` | int | 4 | Número máximo de quiebres a evaluar en la búsqueda. |
| `min_segment_years` | int | 3 | Observaciones mínimas por segmento. |
| `group_col` | str | None | Variable de estratificación. |
| `lifecycle_col` | str | None | Variable de grupo etario (usada por `LifecycleAnalyzer`). |
| `log_transform` | bool | True | Ajustar sobre el logaritmo de la tasa, como exige el cálculo del APC. |

---

## Cómo citar

```bibtex
@software{joinpoint_health_2026,
  author    = {Samillan Vasquez, Cesar Jefferson and
               Acosta Román, Mercedes and
               León Montoya, Gladys Bernardita and
               Bazán Valque, Rosa Ysabel},
  title     = {JoinPoint-Health: regresión joinpoint para indicadores de salud},
  year      = {2026},
  version   = {2.1.0},
  doi       = {10.5281/zenodo.20151357},
  url       = {https://doi.org/10.5281/zenodo.20151357}
}
```

El DOI anterior es el **DOI concepto**, que siempre resuelve a la última versión publicada. En un artículo o donde importe la reproducibilidad exacta, cite el **DOI de la versión** que figura en la página del depósito correspondiente en Zenodo.

El archivo [`CITATION.cff`](CITATION.cff) permite a GitHub y a los gestores bibliográficos generar la cita automáticamente.

---

## Dependencias

| Biblioteca | Versión mínima | Función |
|---|---|---|
| pandas | ≥ 1.3.0 | Manipulación de DataFrames y exportación a Excel |
| numpy | ≥ 1.21.0 | Operaciones numéricas y construcción de la matriz de diseño |
| statsmodels | ≥ 0.13.0 | Ajuste OLS/WLS, matriz de covarianzas e inferencia |
| matplotlib | ≥ 3.4.0 | Gráficos de tendencia, BIC y barras comparativas |
| openpyxl | ≥ 3.0.0 | Exportación a .xlsx |
| geopandas | ≥ 0.10.0 | Opcional — mapas coropléticos |

---

## Referencias

Kim, H. J., Fay, M. P., Feuer, E. J., & Midthune, D. N. (2000). Permutation tests for joinpoint regression with applications to cancer rates. *Statistics in Medicine*, 19(3), 335–351.

Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461–464.

Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. *Proceedings of the 9th Python in Science Conference.*

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95.

---

## Licencia

Este proyecto está bajo la licencia [MIT](LICENSE.txt).
Copyright © 2026 Samillan Vasquez, Acosta Román, León Montoya, Bazán Valque.
