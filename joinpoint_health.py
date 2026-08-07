# -*- coding: utf-8 -*-
"""
joinpoint_health.py — JoinPoint-Health v2.1.0
=============================================
Herramienta generalizada de regresión joinpoint para indicadores
cuantitativos de salud en series de tiempo.

Novedades de la versión 2.1.0
-----------------------------
La versión 2.0.0 ajustaba cada segmento mediante regresiones OLS
independientes. Ese procedimiento produce un modelo discontinuo en los
puntos de quiebre y, por tanto, no corresponde a la regresión joinpoint
descrita por Kim et al. (2000), que exige continuidad en las uniones.
La versión 2.1.0 corrige ese comportamiento:

1. Ajuste continuo mediante splines lineales. Se estima un único modelo
   log(tasa) = b0 + b1*t + sum_k d_k * max(0, t - tau_k)
   de modo que los segmentos se unen por construcción en cada quiebre.
2. Cada observación se emplea una sola vez. En la versión anterior el año
   del quiebre pertenecía a los dos segmentos adyacentes.
3. Recuento de parámetros del BIC acorde al modelo continuo: 2 + 2k.
4. Errores estándar e intervalos de confianza del APC derivados de la
   matriz de covarianzas conjunta, mediante el método delta.
5. Agregación ponderada por el tamaño poblacional cuando existen varias
   filas por año, con `weight_col`.
6. Prueba interna de continuidad, `check_continuity()`, que verifica que
   los segmentos se unen en los quiebres dentro de la tolerancia numérica.
7. Los gráficos emplean el modelo estimado y no un reajuste posterior:
   plot_trend() dibuja la curva continua efectivamente ajustada.

Autores  : Cesar Jefferson Samillan Vasquez, Mercedes Acosta Román,
           Gladys Bernardita León Montoya, Rosa Ysabel Bazán Valque
Contacto : cesar.samillan@untrm.edu.pe
Versión  : 2.1.0
Licencia : MIT
DOI      : 10.5281/zenodo.20151357
Repositorio : github.com/samijeff3012-art/joinpoint-health

Referencias
-----------
Kim HJ, Fay MP, Feuer EJ, Midthune DN (2000). Permutation tests for joinpoint
regression with applications to cancer rates. Statistics in Medicine, 19(3),
335-351.
Schwarz G (1978). Estimating the dimension of a model. Annals of Statistics,
6(2), 461-464.
"""
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

__version__ = "2.1.0"


class DataError(Exception):
    """Error en la estructura o el contenido de los datos de entrada."""


class JoinpointAnalyzer:
    """Analizador de regresión joinpoint con segmentos continuos.

    Parameters
    ----------
    data : pd.DataFrame
        Datos de entrada.
    year_col : str
        Columna del año, con valores enteros.
    rate_col : str
        Columna del indicador cuantitativo.
    weight_col : str or None
        Columna de tamaño poblacional o número de casos. Si se indica y hay
        varias filas por año, las tasas se promedian ponderando por ella.
        Si es None se emplea el promedio simple, con una advertencia.
    breakpoint_years : list[int] or None
        Quiebres forzados. Si es None se detectan por BIC.
    max_breakpoints : int
        Número máximo de quiebres a evaluar.
    group_col, lifecycle_col : str or None
        Columna de estratificación.
    min_segment_years : int
        Número mínimo de observaciones por segmento.
    log_transform : bool
        Aplica logaritmo natural antes de la regresión.
    replace_zeros : float
        Valor de reemplazo de los ceros antes del logaritmo.
    """

    def __init__(self, data, year_col, rate_col, weight_col=None,
                 breakpoint_years=None, max_breakpoints=4, group_col=None,
                 lifecycle_col=None, min_segment_years=3, log_transform=True,
                 replace_zeros=0.01):
        self._validate(data, year_col, rate_col, weight_col, group_col, lifecycle_col)
        self.data = data.copy()
        self.year_col = year_col
        self.rate_col = rate_col
        self.weight_col = weight_col
        self.breakpoint_years = breakpoint_years
        self.max_breakpoints = max_breakpoints
        self.group_col = group_col or lifecycle_col
        self.min_segment_years = min_segment_years
        self.log_transform = log_transform
        self.replace_zeros = replace_zeros
        self.results_ = {}
        self.breakpoints_ = []
        self.bic_table_ = pd.DataFrame()

    # ── Validación ─────────────────────────────────────────────────────────
    @staticmethod
    def _validate(data, year_col, rate_col, weight_col, group_col, lifecycle_col):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("'data' debe ser un DataFrame de pandas.")
        for col in [year_col, rate_col]:
            if col not in data.columns:
                raise ValueError(f"No se encontró la columna '{col}'.")
        for col in [weight_col, group_col, lifecycle_col]:
            if col is not None and col not in data.columns:
                raise ValueError(f"No se encontró la columna '{col}'.")
        if data[rate_col].isnull().all():
            raise ValueError(f"La columna '{rate_col}' solo contiene valores nulos.")
        if (pd.to_numeric(data[rate_col], errors="coerce") < 0).any():
            raise DataError(f"La columna '{rate_col}' contiene valores negativos; "
                            "la transformación logarítmica no está definida.")

    # ── Preparación de la serie ────────────────────────────────────────────
    def _prepare_series(self, subset):
        if self.weight_col is not None:
            def _pond(g):
                w = g[self.weight_col].astype(float)
                if w.sum() <= 0:
                    return g[self.rate_col].mean()
                return np.average(g[self.rate_col].astype(float), weights=w)
            agg = (subset.groupby(self.year_col).apply(_pond)
                   .reset_index(name=self.rate_col))
        else:
            porano = subset.groupby(self.year_col).size()
            if (porano > 1).any():
                warnings.warn(
                    "Hay varias filas por año y no se indicó weight_col: las tasas se "
                    "promedian sin ponderar, lo que sesga el resultado cuando los "
                    "subgrupos tienen tamaños distintos.", UserWarning)
            agg = subset.groupby(self.year_col)[self.rate_col].mean().reset_index()
        agg = agg.sort_values(self.year_col).reset_index(drop=True)
        agg["_y"] = (np.log(agg[self.rate_col].replace(0, self.replace_zeros))
                     if self.log_transform else agg[self.rate_col])
        return agg

    # ── Núcleo: modelo continuo por splines lineales ───────────────────────
    def _design(self, years, taus):
        """Matriz de diseño del spline lineal continuo."""
        cols = [np.ones_like(years, dtype=float), years.astype(float)]
        for t in taus:
            cols.append(np.maximum(0.0, years.astype(float) - float(t)))
        return np.column_stack(cols)

    def _fit_model(self, series, taus):
        """Ajusta un único modelo continuo para todos los segmentos."""
        years = series[self.year_col].values.astype(float)
        y = series["_y"].values.astype(float)
        X = self._design(years, taus)
        if np.linalg.matrix_rank(X) < X.shape[1] or len(y) <= X.shape[1]:
            return None
        return sm.OLS(y, X).fit()

    def _segments_from_model(self, model, series, taus, alpha=0.05):
        """Deriva APC, error estándar e IC por segmento desde el modelo conjunto."""
        from scipy import stats
        years = series[self.year_col].values.astype(float)
        beta = model.params
        V = model.cov_params()
        gl = int(model.df_resid)
        tcrit = stats.t.ppf(1 - alpha / 2, gl)
        limites = [years[0]] + [float(t) for t in taus] + [years[-1]]
        segs = []
        for i in range(len(limites) - 1):
            # la pendiente del segmento i acumula beta1 y los deltas anteriores
            c = np.zeros(len(beta)); c[1] = 1.0
            for j in range(i):
                c[2 + j] = 1.0
            pend = float(c @ beta)
            var = float(c @ V @ c)
            ee = float(np.sqrt(max(var, 0.0)))
            lo_b, hi_b = pend - tcrit * ee, pend + tcrit * ee
            tval = pend / ee if ee > 0 else np.nan
            pval = float(2 * stats.t.sf(abs(tval), gl)) if ee > 0 else np.nan
            if self.log_transform:
                apc = (np.exp(pend) - 1) * 100
                lo = (np.exp(lo_b) - 1) * 100
                hi = (np.exp(hi_b) - 1) * 100
                ee_apc = float(abs(np.exp(pend)) * ee * 100)   # método delta
            else:
                apc, lo, hi, ee_apc = pend, lo_b, hi_b, ee
            # recta del segmento en la escala transformada: y = a_i + s_i * t
            a_i = float(beta[0] - sum(beta[2 + j] * limites[1 + j] for j in range(i)))
            n_obs = int(((years >= limites[i]) & (years <= limites[i + 1])).sum())
            segs.append(dict(
                intercept=a_i,
                phase=f"Segment {i + 1}",
                period=f"{int(limites[i])}-{int(limites[i + 1])}",
                apc=round(float(apc), 4),
                se_apc=round(ee_apc, 4),
                ic_95_lower=round(float(lo), 4),
                ic_95_upper=round(float(hi), 4),
                p_value=round(float(pval), 6) if np.isfinite(pval) else np.nan,
                n_years=n_obs,
                slope=pend,
            ))
        return segs

    def _bic(self, model, n_obs, n_bp):
        """BIC del modelo continuo: 2 parámetros de regresión + k pendientes + k quiebres."""
        rss = float(model.ssr)
        if rss <= 0 or n_obs <= 0:
            return np.inf
        n_par = 2 + 2 * n_bp
        return n_obs * np.log(rss / n_obs) + n_par * np.log(n_obs)

    # ── Búsqueda de quiebres ───────────────────────────────────────────────
    def _candidates(self, years):
        return [int(y) for y in years[self.min_segment_years - 1:
                                      len(years) - self.min_segment_years + 1]]

    def _valid(self, years, taus):
        limites = [years[0]] + sorted(taus) + [years[-1]]
        for i in range(len(limites) - 1):
            n = ((years >= limites[i]) & (years <= limites[i + 1])).sum()
            if n < self.min_segment_years:
                return False
        return True

    def _search(self, series):
        years = series[self.year_col].values
        n = len(years)
        registros = []
        m0 = self._fit_model(series, [])
        b0 = self._bic(m0, n, 0) if m0 is not None else np.inf
        registros.append(dict(n_breakpoints=0, breakpoints=[], BIC=round(b0, 4)))
        mejor_bic, mejor = b0, []
        cand = self._candidates(years)
        for k in range(1, self.max_breakpoints + 1):
            if len(cand) < k:
                break
            lb, lc = np.inf, None
            for combo in combinations(cand, k):
                combo = sorted(combo)
                if not self._valid(years, combo):
                    continue
                m = self._fit_model(series, combo)
                if m is None:
                    continue
                v = self._bic(m, n, k)
                if v < lb:
                    lb, lc = v, list(combo)
            if lc is not None:
                registros.append(dict(n_breakpoints=k, breakpoints=lc, BIC=round(lb, 4)))
                if lb < mejor_bic:
                    mejor_bic, mejor = lb, lc
        return mejor, registros

    # ── Análisis ───────────────────────────────────────────────────────────
    def _analyze_single(self, subset, label="Overall"):
        series = self._prepare_series(subset)
        years = series[self.year_col].values
        if self.breakpoint_years is not None:
            bps = sorted(int(b) for b in self.breakpoint_years)
            fuera = [b for b in bps if not (years[0] < b < years[-1])]
            if fuera:
                raise DataError(f"Quiebres fuera del rango de la serie: {fuera}")
            if not self._valid(years, bps):
                raise DataError(
                    f"Los quiebres {bps} generan segmentos con menos de "
                    f"{self.min_segment_years} observaciones.")
            registros = [dict(n_breakpoints=len(bps), breakpoints=bps, BIC=None)]
        else:
            bps, registros = self._search(series)
        modelo = self._fit_model(series, bps)
        if modelo is None:
            raise DataError(f"No fue posible ajustar el modelo para '{label}'.")
        segs = self._segments_from_model(modelo, series, bps)
        return dict(label=label, breakpoints=bps, segments=segs,
                    bic_table=pd.DataFrame(registros), model=modelo,
                    _series=series, _taus=bps)

    def run(self):
        if self.group_col is None:
            self.results_ = self._analyze_single(self.data)
            self.breakpoints_ = self.results_["breakpoints"]
            self.bic_table_ = self.results_["bic_table"]
        else:
            self.results_ = {}
            bics = []
            for g in sorted(self.data[self.group_col].dropna().unique()):
                res = self._analyze_single(self.data[self.data[self.group_col] == g], str(g))
                self.results_[g] = res
                bics.append(res["bic_table"].assign(group=g))
            if bics:
                self.bic_table_ = pd.concat(bics, ignore_index=True)
        return self

    # ── Verificación de continuidad ────────────────────────────────────────
    def check_continuity(self, tol=1e-8, verbose=True):
        """Verifica que los segmentos se unen en cada quiebre.

        Devuelve un DataFrame con el salto observado en cada punto de unión.
        En un modelo joinpoint correcto todos los saltos deben ser nulos
        dentro de la tolerancia numérica.
        """
        resultados = (self.results_ if isinstance(self.results_, dict)
                      and "label" in self.results_ else None)
        conjuntos = [resultados] if resultados else list(self.results_.values())
        filas = []
        for res in conjuntos:
            if not res:
                continue
            segs, taus = res["segments"], res["_taus"]
            for i, t in enumerate(taus):
                s_izq, s_der = segs[i], segs[i + 1]
                izq = s_izq["intercept"] + s_izq["slope"] * float(t)
                der = s_der["intercept"] + s_der["slope"] * float(t)
                salto = abs(der - izq)
                filas.append(dict(group=res["label"], breakpoint=int(t),
                                  jump=salto, continuous=salto < tol))
        out = pd.DataFrame(filas)
        if verbose:
            if out.empty:
                print("Sin quiebres que verificar: el modelo tiene un solo segmento.")
            elif out["continuous"].all():
                print(f"Continuidad verificada en {len(out)} punto(s) de unión "
                      f"(salto máximo {out['jump'].max():.2e}).")
            else:
                print("ATENCIÓN: se detectaron discontinuidades.")
                print(out.to_string(index=False))
        return out

    # ── Salidas ────────────────────────────────────────────────────────────
    def summary_table(self):
        filas = []

        def _add(res):
            bps = ", ".join(str(b) for b in res["breakpoints"]) or "None"
            for s in res["segments"]:
                filas.append({
                    "Group": res["label"], "Breakpoints": bps,
                    "Segment": s["phase"], "Period": s["period"],
                    "APC (%)": s["apc"], "SE": s["se_apc"],
                    "IC 95%": f"[{s['ic_95_lower']}, {s['ic_95_upper']}]",
                    "p-value": s["p_value"], "n years": s["n_years"],
                })

        if isinstance(self.results_, dict) and "label" in self.results_:
            _add(self.results_)
        else:
            for res in self.results_.values():
                if res:
                    _add(res)
        return pd.DataFrame(filas)

    def fitted_values(self, group=None):
        """Valores ajustados por el modelo continuo, en la escala original."""
        res = (self.results_ if isinstance(self.results_, dict) and "label" in self.results_
               else self.results_[group])
        series, m, taus = res["_series"], res["model"], res["_taus"]
        years = series[self.year_col].values.astype(float)
        pred = self._design(years, taus) @ m.params
        return pd.DataFrame({
            self.year_col: years.astype(int),
            "observed": series[self.rate_col].values,
            "fitted": np.exp(pred) if self.log_transform else pred,
        })


    # ── Visualización ──────────────────────────────────────────────────────
    def plot_trend(
        self,
        title="JoinPoint-Health: análisis de tendencia",
        ylabel="Tasa por 100 000 habitantes",
        xlabel="Año",
        style="grayscale",
        figsize=(10, 6),
        save_path=None,
    ):
        """
        Grafica las tasas observadas, la curva ajustada y los quiebres.

        A diferencia de la versión 2.0.0, la curva proviene del modelo
        continuo efectivamente estimado y no de un reajuste posterior por
        segmentos: por construcción no presenta saltos en los quiebres.

        Un panel por grupo cuando se ha indicado group_col o lifecycle_col.

        Parameters
        ----------
        title, ylabel, xlabel : str
            Título y etiquetas de los ejes.
        style : str
            Estilo de matplotlib (por defecto 'grayscale', apto para
            publicaciones impresas).
        figsize : tuple
            Tamaño base de cada panel, en pulgadas.
        save_path : str or None
            Ruta donde guardar la figura a 300 ppp.
        """
        import matplotlib.pyplot as plt

        if not self.results_:
            print("Ejecute .run() primero.")
            return

        plt.style.use(style)
        linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]

        def _plot_one(ax, res):
            series = res["_series"]
            years = series[self.year_col].values.astype(float)
            rates = series[self.rate_col].values
            taus = res["_taus"]

            ax.scatter(years, rates, color="black", zorder=5,
                       label="Tasa observada", s=38)

            # Curva del modelo continuo, evaluada en una malla fina
            malla = np.linspace(years.min(), years.max(), 400)
            pred = self._design(malla, taus) @ res["model"].params
            ajust = np.exp(pred) if self.log_transform else pred
            ax.plot(malla, ajust, color="black", linewidth=1.9, zorder=4,
                    label="Modelo continuo ajustado")

            # Sombreado y etiqueta por segmento
            for i, seg in enumerate(res["segments"]):
                y0, y1 = (int(v) for v in seg["period"].split("-"))
                if i % 2 == 1:
                    ax.axvspan(y0, y1, color="black", alpha=0.04, zorder=0)
                xm = (y0 + y1) / 2
                ym = np.exp(self._design(np.array([xm]), taus) @ res["model"].params)[0] \
                    if self.log_transform else \
                    (self._design(np.array([xm]), taus) @ res["model"].params)[0]
                signo = "*" if seg["p_value"] < 0.05 else ""
                ax.annotate(f"APC {seg['apc']:+.2f}%{signo}",
                            xy=(xm, ym), xytext=(0, 12),
                            textcoords="offset points", ha="center",
                            fontsize=7.5, color="#303030")

            for bp in res["breakpoints"]:
                ax.axvline(x=bp, color="gray", linestyle=":", alpha=0.7,
                           linewidth=1.2)
                ax.annotate(f"quiebre {bp}", xy=(bp, ax.get_ylim()[1]),
                            xytext=(3, -12), textcoords="offset points",
                            fontsize=7, color="gray", rotation=90, va="top")

            ax.set_title(res["label"], fontsize=11, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(fontsize=7.5, loc="best")
            ax.grid(True, alpha=0.2)

        singular = isinstance(self.results_, dict) and "label" in self.results_
        if singular:
            fig, ax = plt.subplots(figsize=figsize)
            fig.suptitle(title, fontsize=13, fontweight="bold")
            _plot_one(ax, self.results_)
        else:
            validos = {k: v for k, v in self.results_.items() if v and "_series" in v}
            n = len(validos)
            if n == 0:
                print("No hay resultados que graficar.")
                return
            cols = min(3, n)
            filas = (n + cols - 1) // cols
            fig, axes = plt.subplots(filas, cols,
                                     figsize=(figsize[0] * cols, figsize[1] * filas))
            fig.suptitle(title, fontsize=13, fontweight="bold")
            planos = np.array(axes).flatten() if n > 1 else [axes]
            for ax, (_, res) in zip(planos, validos.items()):
                _plot_one(ax, res)
            for ax in planos[n:]:
                ax.set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figura de tendencia guardada: {save_path}")
        plt.show()

    def plot_bic(self, save_path=None, figsize=(6, 4)):
        """
        Grafica el BIC frente al número de quiebres evaluados.

        Muestra por qué el algoritmo eligió determinada configuración. El
        mínimo se marca con una línea vertical discontinua. Incluir esta
        curva en una publicación aporta transparencia sobre la decisión del
        modelo.

        Parameters
        ----------
        save_path : str or None
            Ruta donde guardar la figura a 300 ppp.
        figsize : tuple
            Tamaño del panel, en pulgadas.
        """
        import matplotlib.pyplot as plt

        if self.bic_table_ is None or self.bic_table_.empty:
            print("Ejecute .run() primero.")
            return
        tabla = self.bic_table_.dropna(subset=["BIC"]).copy()
        if tabla.empty:
            print("No hay curva BIC: los quiebres se fijaron manualmente.")
            return

        plt.style.use("grayscale")

        def _uno(ax, sub, titulo):
            mejor = sub.loc[sub["BIC"].idxmin()]
            ax.plot(sub["n_breakpoints"], sub["BIC"], marker="o", color="black")
            ax.axvline(x=mejor["n_breakpoints"], color="gray", linestyle="--",
                       label=f"Elegido: {int(mejor['n_breakpoints'])} quiebre(s)")
            ax.set_xlabel("Número de quiebres", fontsize=9)
            ax.set_ylabel("BIC", fontsize=9)
            ax.set_title(titulo, fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
            ax.set_xticks(sorted(sub["n_breakpoints"].unique()))

        if "group" in tabla.columns:
            grupos = list(tabla["group"].unique())
            fig, axes = plt.subplots(1, len(grupos),
                                     figsize=(figsize[0] * len(grupos), figsize[1]))
            axes = [axes] if len(grupos) == 1 else list(np.array(axes).flatten())
            for ax, g in zip(axes, grupos):
                _uno(ax, tabla[tabla["group"] == g], str(g))
        else:
            fig, ax = plt.subplots(figsize=figsize)
            _uno(ax, tabla, "Selección del modelo por BIC")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figura BIC guardada: {save_path}")
        plt.show()

    def plot_map(
        self,
        region_col=None,
        geojson_url=None,
        region_name_field=None,
        country_iso="PER",
        metric="apc",
        segment=-1,
        title="JoinPoint-Health: distribución geográfica",
        cmap="RdYlGn_r",
        figsize=(12, 14),
        save_path=None,
    ):
        """
        Mapa coroplético del APC (o del valor p) por región.

        Requiere geopandas y un archivo GeoJSON de límites. El predeterminado
        cubre el Perú a nivel departamental. El análisis debe haberse
        ejecutado con group_col igual a la columna de región.

        Parameters
        ----------
        region_col : str or None
            Se conserva por compatibilidad; las regiones se toman de las
            etiquetas de los resultados, es decir, de group_col.
        geojson_url : str or None
            URL de un GeoJSON de límites. Si es None se usa el del Perú.
        region_name_field : str or None
            Propiedad del GeoJSON que contiene el nombre de la región. Se
            detecta automáticamente si es None.
        country_iso : str
            Código ISO 3166-1 alfa-3 para elegir el GeoJSON por defecto.
        metric : str
            'apc' o 'p_value'.
        segment : int
            Índice del segmento a mapear. -1 es el último (posterior al
            quiebre); 0 es el primero.
        title : str
            Título del mapa.
        cmap : str
            Mapa de color de matplotlib.
        figsize : tuple
            Tamaño de la figura, en pulgadas.
        save_path : str or None
            Ruta donde guardar el mapa a 300 ppp.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "plot_map() requiere geopandas. Instálelo con: pip install geopandas")

        filas = []

        def _extraer(res):
            segs = res.get("segments", [])
            if not segs:
                return
            idx = segment if segment >= 0 else len(segs) + segment
            idx = max(0, min(idx, len(segs) - 1))
            s = segs[idx]
            filas.append({"region": res["label"], "apc": s["apc"],
                          "p_value": s["p_value"]})

        if isinstance(self.results_, dict) and "label" in self.results_:
            _extraer(self.results_)
        else:
            for res in self.results_.values():
                if res:
                    _extraer(res)

        if not filas:
            print("No hay resultados. Ejecute .run() antes de .plot_map().")
            return
        if len(filas) == 1:
            warnings.warn(
                "Solo hay una región en los resultados. Ejecute el análisis con "
                "group_col igual a la columna de región para obtener un mapa.")

        df_metric = pd.DataFrame(filas)

        URLS = {"PER": ("https://raw.githubusercontent.com/juaneladio/"
                        "peru-geojson/master/peru_departamental_simple.geojson")}
        url = geojson_url or URLS.get(country_iso.upper(), URLS["PER"])
        try:
            gdf = gpd.read_file(url)
        except Exception as e:
            print(f"No se pudo cargar el GeoJSON desde {url}.\nError: {e}")
            return

        if region_name_field is None:
            cand = [c for c in gdf.columns
                    if any(k in c.upper()
                           for k in ["NAME", "NOMBRE", "NOMB", "REGION", "DEP"])]
            region_name_field = cand[0] if cand else gdf.columns[0]

        gdf["_key"] = gdf[region_name_field].astype(str).str.upper().str.strip()
        df_metric["_key"] = df_metric["region"].astype(str).str.upper().str.strip()
        gdf = gdf.merge(df_metric[["_key", metric]], on="_key", how="left")

        sin_datos = int(gdf[metric].isna().sum())
        if sin_datos == len(gdf):
            print("Ninguna región del GeoJSON coincidió con las etiquetas de los "
                  "resultados. Revise la ortografía de la columna de región.")
            return

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        gdf[gdf[metric].isna()].plot(ax=ax, color="#e0e0e0",
                                     edgecolor="white", linewidth=0.5)
        gdf[gdf[metric].notna()].plot(
            ax=ax, column=metric, cmap=cmap, edgecolor="white", linewidth=0.5,
            legend=True,
            legend_kwds={"label": "APC (%)" if metric == "apc" else "valor p",
                         "orientation": "horizontal", "shrink": 0.5})

        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
        ax.set_axis_off()
        ax.legend(handles=[mpatches.Patch(color="#e0e0e0", label="Sin datos")],
                  loc="lower left", frameon=False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Mapa guardado: {save_path}")
        plt.show()

    def export_results(self, path="joinpoint_results.xlsx"):
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            self.summary_table().to_excel(w, sheet_name="Resumen", index=False)
            self.bic_table_.to_excel(w, sheet_name="BIC", index=False)
            self.check_continuity(verbose=False).to_excel(
                w, sheet_name="Continuidad", index=False)
        print(f"Resultados exportados a: {path}")


class LifecycleAnalyzer(JoinpointAnalyzer):
    """Subclase para el análisis por grupos del ciclo de vida."""

    DEFAULT_ORDER = ["Children", "Adolescents", "Young Adults", "Adults", "Older Adults"]

    def __init__(self, data, year_col, rate_col, lifecycle_col, group_order=None, **kw):
        super().__init__(data=data, year_col=year_col, rate_col=rate_col,
                         lifecycle_col=lifecycle_col, **kw)
        self.group_order = group_order or self.DEFAULT_ORDER

    def plot_lifecycle_bars(
        self,
        title="APC por grupo del ciclo de vida y segmento",
        ylabel="Cambio porcentual anual (%)",
        figsize=(10, 6),
        save_path=None,
    ):
        """
        Barras agrupadas del APC por grupo del ciclo de vida y segmento.

        Las barras por encima de cero indican tasas crecientes; por debajo,
        decrecientes. Las barras de error representan el intervalo de
        confianza al 95 %, derivado de la matriz de covarianzas conjunta.

        Parameters
        ----------
        title, ylabel : str
            Título y etiqueta del eje vertical.
        figsize : tuple
            Tamaño de la figura, en pulgadas.
        save_path : str or None
            Ruta donde guardar la figura a 300 ppp.
        """
        import matplotlib.pyplot as plt

        if not self.results_:
            print("Ejecute .run() antes de .plot_lifecycle_bars().")
            return
        resumen = self.summary_table()
        if resumen.empty:
            print("No hay resultados que graficar.")
            return

        presentes = list(resumen["Group"].unique())
        orden = [g for g in self.group_order if g in presentes]
        orden += [g for g in presentes if g not in orden]

        segmentos = list(resumen["Segment"].unique())
        n_seg = len(segmentos)
        x = np.arange(len(orden))
        ancho = 0.8 / max(n_seg, 1)
        tonos = ["black", "gray", "darkgray", "silver", "lightgray"]

        plt.style.use("grayscale")
        fig, ax = plt.subplots(figsize=figsize)

        for i, seg in enumerate(segmentos):
            sub = resumen[resumen["Segment"] == seg].set_index("Group")
            apcs, lo, hi = [], [], []
            for g in orden:
                if g in sub.index:
                    apcs.append(float(sub.loc[g, "APC (%)"]))
                    crudo = str(sub.loc[g, "IC 95%"]).strip("[]").split(",")
                    lo.append(float(crudo[0]))
                    hi.append(float(crudo[1]))
                else:
                    apcs.append(0.0)
                    lo.append(0.0)
                    hi.append(0.0)
            err = np.vstack([np.array(apcs) - np.array(lo),
                             np.array(hi) - np.array(apcs)])
            err = np.clip(err, 0, None)
            ax.bar(x + i * ancho - 0.4 + ancho / 2, apcs, ancho,
                   yerr=err, capsize=3, label=str(seg),
                   color=tonos[i % len(tonos)], alpha=0.85,
                   error_kw={"linewidth": 0.9})

        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(orden, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, title="Segmento", title_fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figura de ciclo de vida guardada: {save_path}")
        plt.show()


def analyze_health_trend(data, year_col, rate_col, weight_col=None,
                         breakpoint_years=None, max_breakpoints=4, group_col=None,
                         lifecycle_col=None, **kw):
    """Ejecuta el análisis completo y devuelve la tabla resumen."""
    cls = LifecycleAnalyzer if lifecycle_col else JoinpointAnalyzer
    if lifecycle_col:
        az = cls(data=data, year_col=year_col, rate_col=rate_col,
                 lifecycle_col=lifecycle_col, weight_col=weight_col,
                 breakpoint_years=breakpoint_years,
                 max_breakpoints=max_breakpoints, **kw)
    else:
        az = cls(data=data, year_col=year_col, rate_col=rate_col,
                 weight_col=weight_col, breakpoint_years=breakpoint_years,
                 max_breakpoints=max_breakpoints, group_col=group_col, **kw)
    az.run()
    return az.summary_table()
