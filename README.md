# JoinPoint-Health v2.0.0

> **Herramienta generalizada de regresión joinpoint para indicadores cuantitativos de salud en series de tiempo.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## ¿Qué es JoinPoint-Health?

**JoinPoint-Health** es un módulo Python de código abierto que implementa regresión joinpoint generalizada para el análisis de tendencias temporales en indicadores cuantitativos de salud pública (tasas de mortalidad, incidencia, prevalencia, cobertura, entre otros).

A diferencia de herramientas existentes, JoinPoint-Health:

- **Detecta automáticamente** el número óptimo de quiebres estructurales usando el Criterio de Información Bayesiana (BIC).
- **Generaliza** a cualquier indicador cuantitativo, no solo mortalidad por cáncer.
- **Incluye un módulo de ciclo de vida** para análisis simultáneo por grupos etarios.
- **Genera mapas coropléticos** geográficos del APC por región para cualquier país.
- Produce **tablas y gráficos listos para publicación científica**.

---

## Autores

| Nombre | Institución |
|--------|-------------|
| Cesar Jefferson Samillan Vasquez | Universidad Nacional Toribio Rodríguez de Mendoza |
| Mercedes Acosta Román | Universidad Nacional Autónoma de Tayacaja Daniel Hernández Morillo |
| Gladys Bernardita León Montoya | Universidad Nacional Autónoma de Tayacaja Daniel Hernández Morillo |

**Contacto:** cjeffry.30@gmail.com

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/joinpoint-health.git
cd joinpoint-health

# Instalar dependencias
pip install -r requirements.txt

# Dependencia opcional para mapas geográficos
pip install geopandas
```

---

## Inicio rápido

```python
import pandas as pd
from joinpoint_health import JoinpointAnalyzer

# Cargar datos (columna de años + columna del indicador)
df = pd.read_excel("mis_datos.xlsx")

# Crear el analizador y ejecutar
az = JoinpointAnalyzer(
    data            = df,
    year_col        = "Anio",
    rate_col        = "TasaAjustada",
    max_breakpoints = 4          # BIC elige el número óptimo automáticamente
)
az.run()

# Ver resultados
print(az.summary_table())

# Gráfico de tendencia
az.plot_trend(title="Tendencia Nacional de Mortalidad")

# Exportar a Excel
az.export_results("resultados.xlsx")
```

**Salida esperada:**

```
   Group  Breakpoints   Segment      Period  APC (%)              IC 95%  p-value  n years
 Overall         2019  Segment 1   2008-2019    -1.47  [-2.01,  -0.93]   0.0001       12
 Overall         2019  Segment 2   2019-2022   +13.81  [ 8.34,  19.28]   0.0052        4
```

---

## Análisis por ciclo de vida

```python
from joinpoint_health import LifecycleAnalyzer

lc = LifecycleAnalyzer(
    data          = df,
    year_col      = "Anio",
    rate_col      = "TasaAjustada",
    lifecycle_col = "GrupoEtario",      # columna con grupos etarios
    breakpoint_years = [2019]
)
lc.run()
lc.plot_trend(title="Mortalidad por Ciclo de Vida")
lc.plot_lifecycle_bars()               # barras comparativas de APC por grupo
```

---

## Mapa geográfico

```python
# Después de un análisis estratificado por región:
az.plot_map(
    region_col  = "Region",
    country_iso = "PER",              # Perú por defecto; acepta cualquier GeoJSON
    metric      = "apc",              # o "p_value"
    segment     = -1,                 # último segmento (post-quiebre)
    title       = "APC por Departamento 2019-2022"
)
```

---

## Estructura del repositorio

```
joinpoint-health/
├── joinpoint_health.py     ← Módulo principal (JoinpointAnalyzer, LifecycleAnalyzer)
├── example_usage.py        ← Ejemplos de uso con datos sintéticos
├── requirements.txt        ← Dependencias Python
├── LICENSE.txt             ← Licencia MIT
└── README.md               ← Este archivo
```

---

## Componentes principales

| Componente | Descripción |
|---|---|
| `JoinpointAnalyzer` | Clase principal. Detección automática de quiebres por BIC o forzados. |
| `LifecycleAnalyzer` | Subclase para análisis por grupos del ciclo de vida (edad). |
| `analyze_health_trend()` | Función de conveniencia de una sola línea. |
| `.plot_trend()` | Gráfico de tendencias con líneas de regresión por segmento. |
| `.plot_bic()` | Gráfico de selección de modelo BIC. |
| `.plot_lifecycle_bars()` | Barras comparativas de APC por grupo etario. |
| `.plot_map()` | Mapa coroplético geográfico de APC o p-valor. |
| `.summary_table()` | Tabla con APC, IC 95% y p-valor por segmento. |
| `.export_results()` | Exportación a Excel (.xlsx). |

---

## Parámetros principales

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `data` | DataFrame | Requerido | Datos de entrada. |
| `year_col` | str | Requerido | Columna de años. |
| `rate_col` | str | Requerido | Columna del indicador de salud. |
| `breakpoint_years` | list[int] | None | Quiebres forzados. Si None, se detectan por BIC. |
| `max_breakpoints` | int | 4 | Máximo número de quiebres a evaluar. |
| `group_col` | str | None | Columna de estratificación (región, sexo, etc.). |
| `lifecycle_col` | str | None | Columna de grupos etarios. |
| `min_segment_years` | int | 3 | Mínimo de años por segmento. |
| `log_transform` | bool | True | Aplica log antes de la regresión (estándar para tasas). |

---

## Cómo citar

Si utilizas JoinPoint-Health en tu investigación, por favor cita:

```bibtex
@software{joinpoint_health_2026,
  author    = {Samillan Vasquez, Cesar Jefferson and
               Acosta Román, Mercedes and
               León Montoya, Gladys Bernardita},
  title     = {JoinPoint-Health: A generalized joinpoint regression
               tool for quantitative health indicators},
  year      = {2026},
  version   = {2.0.0},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

---

## Dependencias

- [pandas](https://pandas.pydata.org/) ≥ 1.3.0
- [numpy](https://numpy.org/) ≥ 1.21.0
- [statsmodels](https://www.statsmodels.org/) ≥ 0.13.0
- [matplotlib](https://matplotlib.org/) ≥ 3.4.0
- [openpyxl](https://openpyxl.readthedocs.io/) ≥ 3.0.0
- [geopandas](https://geopandas.org/) ≥ 0.10.0 *(opcional, para mapas)*

---

## Referencias

- Kim HJ, Fay MP, Feuer EJ, Midthune DN. (2000). Permutation tests for joinpoint regression with applications to cancer rates. *Statistics in Medicine*, 19(3), 335–351.
- Schwarz G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461–464.

---

## Licencia

Este proyecto está bajo la licencia [MIT](LICENSE.txt).  
Copyright © 2026 Samillan Vasquez, Acosta Román, León Montoya.
