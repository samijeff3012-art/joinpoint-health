# -*- coding: utf-8 -*-
"""
joinpoint_health.py  —  JoinPoint-Health v2.0.0
================================================
A generalized tool for joinpoint regression analysis of quantitative
health indicators over time.

Automatically detects the optimal number of structural breakpoints using
the Bayesian Information Criterion (BIC), calculates Annual Percent Change
(APC) per segment with 95% confidence intervals and p-values, supports
stratified and life-cycle analysis, and produces publication-ready trend
plots, BIC model-selection charts, and geographic choropleth maps.

Author  : Cesar Jefferson Samillan Vasquez, Mercedes Acosta Román, Gladys Bernardita León Montoya
Version : 2.0.0
License : MIT
DOI     : [To be assigned via Zenodo]

References
----------
Kim HJ, et al. (2000). Permutation tests for joinpoint regression with
applications to cancer rates. Statistics in Medicine, 19(3), 335-351.

Schwarz G. (1978). Estimating the dimension of a model. Annals of
Statistics, 6(2), 461-464.

Usage
-----
>>> from joinpoint_health import JoinpointAnalyzer, LifecycleAnalyzer
>>> from joinpoint_health import analyze_health_trend

# — National trend, auto-detect breakpoints
>>> az = JoinpointAnalyzer(df, year_col='Year', rate_col='Rate')
>>> az.run()
>>> az.plot_trend()
>>> print(az.summary_table())

# — Life-cycle analysis
>>> lc = LifecycleAnalyzer(df, year_col='Year', rate_col='Rate',
...                         lifecycle_col='AgeGroup')
>>> lc.run()
>>> lc.plot_trend()
>>> lc.plot_lifecycle_bars()

# — Geographic map
>>> az.plot_map(region_col='Region', country_iso='PER')
"""

import warnings
from itertools import combinations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CORE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class JoinpointAnalyzer:
    """
    Joinpoint regression analyzer for quantitative health time-series.

    Automatically detects the optimal number and position of structural
    breakpoints using BIC minimisation, then calculates APC with 95 %
    confidence intervals for each resulting segment.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with at least a year column and a rate column.
    year_col : str
        Name of the year column (integer values expected).
    rate_col : str
        Name of the quantitative health indicator column
        (e.g. mortality rate, incidence, prevalence).
    breakpoint_years : list[int] or None
        Force specific breakpoints instead of searching. Example: [2014, 2019].
        Default None (auto-detect via BIC).
    max_breakpoints : int
        Maximum number of breakpoints to search for (default 4).
    group_col : str or None
        Column for stratified analysis (e.g. 'Region', 'Sex').
    lifecycle_col : str or None
        Alias for group_col, used semantically for age-group analyses.
        If both supplied, group_col takes precedence.
    min_segment_years : int
        Minimum observations required per segment (default 3).
    log_transform : bool
        Apply log transformation before OLS — standard for rate data
        (default True).
    replace_zeros : float
        Replacement value for zeros before log transform (default 0.01).

    Attributes
    ----------
    results_ : dict
        Full results after calling run().
    breakpoints_ : list[int]
        Optimal breakpoint years (global or per group).
    bic_table_ : pd.DataFrame
        BIC scores per candidate number of breakpoints.
    """

    def __init__(
        self,
        data,
        year_col,
        rate_col,
        breakpoint_years=None,
        max_breakpoints=4,
        group_col=None,
        lifecycle_col=None,
        min_segment_years=3,
        log_transform=True,
        replace_zeros=0.01,
    ):
        self._validate_inputs(data, year_col, rate_col, group_col, lifecycle_col)
        self.data              = data.copy()
        self.year_col          = year_col
        self.rate_col          = rate_col
        self.breakpoint_years  = breakpoint_years
        self.max_breakpoints   = max_breakpoints
        self.group_col         = group_col or lifecycle_col
        self.min_segment_years = min_segment_years
        self.log_transform     = log_transform
        self.replace_zeros     = replace_zeros
        self.results_          = {}
        self.breakpoints_      = []
        self.bic_table_        = pd.DataFrame()

    # ── Validation ───────────────────────────────────────────────────────────

    def _validate_inputs(self, data, year_col, rate_col, group_col, lifecycle_col):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("'data' must be a pandas DataFrame.")
        for col in [year_col, rate_col]:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
        for col in [group_col, lifecycle_col]:
            if col is not None and col not in data.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
        if data[rate_col].isnull().all():
            raise ValueError(f"Column '{rate_col}' contains only null values.")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _prepare_series(self, subset):
        agg = subset.groupby(self.year_col)[self.rate_col].mean().reset_index()
        agg = agg.sort_values(self.year_col).reset_index(drop=True)
        agg['_y'] = (
            np.log(agg[self.rate_col].replace(0, self.replace_zeros))
            if self.log_transform else agg[self.rate_col]
        )
        return agg

    def _fit_segment(self, series, start, end):
        seg = series[
            (series[self.year_col] >= start) &
            (series[self.year_col] <= end)
        ].copy()
        if len(seg) < 2:
            return None

        X     = sm.add_constant(seg[self.year_col].astype(float))
        model = sm.OLS(seg['_y'], X).fit()
        slope = model.params[self.year_col]
        p_val = model.pvalues[self.year_col]
        ci    = model.conf_int().loc[self.year_col]

        if self.log_transform:
            apc      = (np.exp(slope) - 1) * 100
            ic_lower = (np.exp(ci[0]) - 1) * 100
            ic_upper = (np.exp(ci[1]) - 1) * 100
        else:
            apc, ic_lower, ic_upper = slope, ci[0], ci[1]

        return {
            'apc'        : round(float(apc), 4),
            'ic_95_lower': round(float(ic_lower), 4),
            'ic_95_upper': round(float(ic_upper), 4),
            'p_value'    : round(float(p_val), 4),
            'n_years'    : int(len(seg)),
            'rss'        : float(model.ssr),
            'n_params'   : int(model.df_model + 1),
        }

    def _bic(self, total_rss, n_obs, n_params):
        if total_rss <= 0 or n_obs <= 0:
            return np.inf
        return n_obs * np.log(total_rss / n_obs) + n_params * np.log(n_obs)

    def _candidate_breakpoints(self, years):
        min_y = years[self.min_segment_years - 1]
        max_y = years[-self.min_segment_years]
        return [int(y) for y in years if min_y <= y <= max_y]

    def _search_breakpoints(self, series):
        """Search 0..max_breakpoints using BIC; return best list and BIC table."""
        years = series[self.year_col].values
        n_obs = len(years)
        records = []

        seg0 = self._fit_segment(series, years[0], years[-1])
        bic0 = self._bic(seg0['rss'], n_obs, seg0['n_params']) if seg0 else np.inf
        records.append({'n_breakpoints': 0, 'breakpoints': [], 'BIC': round(bic0, 4)})

        best_bic = bic0
        best_bps = []
        candidates = self._candidate_breakpoints(years)

        for n_bp in range(1, self.max_breakpoints + 1):
            if len(candidates) < n_bp:
                break
            local_best_bic   = np.inf
            local_best_combo = None

            for combo in combinations(candidates, n_bp):
                combo      = sorted(combo)
                boundaries = [years[0]] + combo + [years[-1]]
                valid = all(
                    boundaries[i + 1] - boundaries[i] >= self.min_segment_years - 1
                    for i in range(len(boundaries) - 1)
                )
                if not valid:
                    continue

                total_rss  = 0.0
                total_pars = 0
                ok = True
                for i in range(len(boundaries) - 1):
                    seg = self._fit_segment(series, boundaries[i], boundaries[i + 1])
                    if seg is None:
                        ok = False
                        break
                    total_rss  += seg['rss']
                    total_pars += seg['n_params']

                if not ok:
                    continue

                bic_val = self._bic(total_rss, n_obs, total_pars + n_bp)
                if bic_val < local_best_bic:
                    local_best_bic   = bic_val
                    local_best_combo = list(combo)

            if local_best_combo is not None:
                records.append({
                    'n_breakpoints': n_bp,
                    'breakpoints'  : local_best_combo,
                    'BIC'          : round(local_best_bic, 4),
                })
                if local_best_bic < best_bic:
                    best_bic = local_best_bic
                    best_bps = local_best_combo

        return best_bps, records

    def _build_segments(self, series, bps):
        years      = series[self.year_col].values
        boundaries = [years[0]] + sorted(bps) + [years[-1]]
        segments   = []
        for i in range(len(boundaries) - 1):
            start = int(boundaries[i])
            end   = int(boundaries[i + 1])
            seg   = self._fit_segment(series, start, end)
            if seg:
                seg['period'] = f"{start}-{end}"
                seg['phase']  = f"Segment {i + 1}"
                segments.append(seg)
        return segments

    def _analyze_single(self, subset, label='Overall'):
        series = self._prepare_series(subset)

        if self.breakpoint_years is not None:
            bps     = sorted(self.breakpoint_years)
            bic_rec = [{'n_breakpoints': len(bps), 'breakpoints': bps, 'BIC': None}]
        else:
            bps, bic_rec = self._search_breakpoints(series)

        segments = self._build_segments(series, bps)
        return {
            'label'      : label,
            'breakpoints': bps,
            'segments'   : segments,
            'bic_table'  : pd.DataFrame(bic_rec),
            '_series'    : series,
        }

    # ── Public API ───────────────────────────────────────────────────────────

    def run(self):
        """
        Execute the joinpoint analysis.

        Returns self for chaining. Results stored in self.results_.
        """
        if self.group_col is None:
            self.results_     = self._analyze_single(self.data, label='Overall')
            self.breakpoints_ = self.results_['breakpoints']
            self.bic_table_   = self.results_['bic_table']
        else:
            groups = self.data[self.group_col].dropna().unique()
            self.results_ = {}
            bic_all = []
            for grp in sorted(groups):
                subset = self.data[self.data[self.group_col] == grp]
                res    = self._analyze_single(subset, label=str(grp))
                self.results_[grp] = res
                bic_all.append(res['bic_table'].assign(group=grp))
            if bic_all:
                self.bic_table_ = pd.concat(bic_all, ignore_index=True)
        return self

    def summary_table(self):
        """
        Return a publication-ready summary DataFrame.

        Columns: Group, Breakpoints, Segment, Period, APC (%), IC 95%,
                 p-value, n years.
        """
        rows    = []
        results = self.results_

        def _add(res):
            bps_str = ', '.join(str(b) for b in res['breakpoints']) or 'None'
            for seg in res['segments']:
                rows.append({
                    'Group'      : res['label'],
                    'Breakpoints': bps_str,
                    'Segment'    : seg['phase'],
                    'Period'     : seg['period'],
                    'APC (%)'    : seg['apc'],
                    'IC 95%'     : f"[{seg['ic_95_lower']}, {seg['ic_95_upper']}]",
                    'p-value'    : seg['p_value'],
                    'n years'    : seg['n_years'],
                })

        if isinstance(results, dict) and 'label' in results:
            _add(results)
        else:
            for res in results.values():
                if res:
                    _add(res)

        return pd.DataFrame(rows)

    def export_results(self, path='joinpoint_results.xlsx'):
        """Export summary table to Excel."""
        self.summary_table().to_excel(path, index=False)
        print(f"Results exported to: {path}")

    # ── Plotting: trend ──────────────────────────────────────────────────────

    def plot_trend(
        self,
        title='JoinPoint-Health: Trend Analysis',
        ylabel='Rate per 100,000 inhabitants',
        xlabel='Year',
        style='grayscale',
        figsize=(10, 6),
        save_path=None,
    ):
        """
        Plot observed rates, fitted segment lines and breakpoint markers.

        One panel per group when group_col / lifecycle_col is provided.

        Parameters
        ----------
        title : str
            Figure title.
        ylabel : str
            Y-axis label.
        xlabel : str
            X-axis label.
        style : str
            Matplotlib style (default 'grayscale' for publications).
        figsize : tuple
            Base panel size in inches.
        save_path : str or None
            Save figure at 300 dpi if provided.
        """
        plt.style.use(style)
        linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]
        shades     = ['black', 'dimgray', 'gray', 'darkgray', 'silver']

        def _plot_one(ax, res):
            series = res['_series']
            years  = series[self.year_col].values
            rates  = series[self.rate_col].values

            ax.scatter(years, rates, color='black', zorder=5,
                       label='Observed rates', s=40)

            for i, seg in enumerate(res['segments']):
                y0, y1 = map(int, seg['period'].split('-'))
                mask   = (years >= y0) & (years <= y1)
                sx, sy = years[mask].astype(float), rates[mask]
                if len(sx) < 2:
                    continue
                z      = np.polyfit(sx,
                                    np.log(sy + 1e-9) if self.log_transform else sy, 1)
                fitted = (np.exp(np.poly1d(z)(sx))
                          if self.log_transform else np.poly1d(z)(sx))
                lbl = (f"{seg['phase']} ({seg['period']}): "
                       f"APC={seg['apc']:+.2f}%  p={seg['p_value']:.3f}")
                ax.plot(sx, fitted,
                        linestyle=linestyles[i % len(linestyles)],
                        color=shades[i % len(shades)],
                        linewidth=1.8, label=lbl)

            for bp in res['breakpoints']:
                ax.axvline(x=bp, color='gray', linestyle=':', alpha=0.5,
                           label=f'Breakpoint ({bp})')

            ax.set_title(res['label'], fontsize=11, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.2)

        results  = self.results_
        singular = isinstance(results, dict) and 'label' in results

        if singular:
            fig, ax = plt.subplots(figsize=figsize)
            fig.suptitle(title, fontsize=13, fontweight='bold')
            _plot_one(ax, results)
        else:
            valid = {k: v for k, v in results.items() if v and '_series' in v}
            n     = len(valid)
            cols  = min(3, n)
            rows  = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols,
                                     figsize=(figsize[0] * cols, figsize[1] * rows))
            fig.suptitle(title, fontsize=13, fontweight='bold')
            flat = np.array(axes).flatten() if n > 1 else [axes]
            for ax, (_, res) in zip(flat, valid.items()):
                _plot_one(ax, res)
            for ax in flat[n:]:
                ax.set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trend figure saved: {save_path}")
        plt.show()

    # ── Plotting: BIC selection ───────────────────────────────────────────────

    def plot_bic(self, save_path=None):
        """
        Plot BIC score vs. number of breakpoints (model selection chart).

        Shows why the algorithm chose a specific number of breakpoints.
        The minimum BIC point is marked with a dashed vertical line.

        Parameters
        ----------
        save_path : str or None
            Save figure at 300 dpi if provided.
        """
        if self.bic_table_.empty:
            print("Run .run() first.")
            return

        plt.style.use('grayscale')
        bic_df = self.bic_table_.copy()

        if 'group' in bic_df.columns:
            groups = bic_df['group'].unique()
            fig, axes = plt.subplots(1, len(groups),
                                     figsize=(5 * len(groups), 4))
            axes = [axes] if len(groups) == 1 else list(np.array(axes).flatten())
            for ax, grp in zip(axes, groups):
                sub  = bic_df[bic_df['group'] == grp].dropna(subset=['BIC'])
                best = sub.loc[sub['BIC'].idxmin()]
                ax.plot(sub['n_breakpoints'], sub['BIC'], marker='o', color='black')
                ax.axvline(x=best['n_breakpoints'], color='gray', linestyle='--',
                           label=f"Best: {int(best['n_breakpoints'])} bp")
                ax.set_title(str(grp), fontsize=10)
                ax.set_xlabel('Number of breakpoints')
                ax.set_ylabel('BIC')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.2)
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            sub  = bic_df.dropna(subset=['BIC'])
            best = sub.loc[sub['BIC'].idxmin()]
            ax.plot(sub['n_breakpoints'], sub['BIC'], marker='o', color='black')
            ax.axvline(x=best['n_breakpoints'], color='gray', linestyle='--',
                       label=f"Best: {int(best['n_breakpoints'])} bp")
            ax.set_xlabel('Number of breakpoints')
            ax.set_ylabel('BIC')
            ax.set_title('BIC Model Selection', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.2)

        plt.suptitle('JoinPoint-Health — BIC Model Selection', fontsize=11)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"BIC figure saved: {save_path}")
        plt.show()

    # ── Plotting: geographic map ──────────────────────────────────────────────

    def plot_map(
        self,
        region_col,
        geojson_url=None,
        region_name_field=None,
        country_iso='PER',
        metric='apc',
        segment=-1,
        title='JoinPoint-Health: Geographic Distribution',
        cmap='RdYlGn_r',
        figsize=(12, 14),
        save_path=None,
    ):
        """
        Choropleth map of APC (or p-value) by region.

        Requires geopandas. Any GeoJSON boundary file can be supplied;
        the default covers Peru at the departmental level.

        Parameters
        ----------
        region_col : str
            Column in data whose values match region names in the GeoJSON.
        geojson_url : str or None
            URL to a GeoJSON boundary file. Defaults to Peru if None.
        region_name_field : str or None
            Property in the GeoJSON containing region names.
            Auto-detected from common field names if None.
        country_iso : str
            ISO 3166-1 alpha-3 code for default GeoJSON selection
            (default 'PER').
        metric : str
            'apc' or 'p_value' (default 'apc').
        segment : int
            Which segment index to map. -1 = last (post-breakpoint),
            0 = first (pre-breakpoint).
        title : str
            Map title.
        cmap : str
            Matplotlib colormap (default 'RdYlGn_r').
        figsize : tuple
            Figure size in inches.
        save_path : str or None
            Save map at 300 dpi if provided.
        """
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas required. Install with: pip install geopandas"
            )

        # Build per-region metric table
        results = self.results_
        rows    = []

        def _extract(res):
            segs = res.get('segments', [])
            if not segs:
                return
            idx = segment if segment >= 0 else len(segs) + segment
            idx = max(0, min(idx, len(segs) - 1))
            seg = segs[idx]
            rows.append({
                'region' : res['label'],
                'apc'    : seg['apc'],
                'p_value': seg['p_value'],
            })

        if isinstance(results, dict) and 'label' in results:
            _extract(results)
        else:
            for res in results.values():
                if res:
                    _extract(res)

        if not rows:
            print("No results. Run .run() before .plot_map().")
            return

        df_metric = pd.DataFrame(rows)

        # Load GeoJSON
        DEFAULT_URLS = {
            'PER': ('https://raw.githubusercontent.com/juaneladio/'
                    'peru-geojson/master/peru_departamental_simple.geojson'),
        }
        url = geojson_url or DEFAULT_URLS.get(country_iso.upper(),
                                               DEFAULT_URLS['PER'])
        try:
            gdf = gpd.read_file(url)
        except Exception as e:
            print(f"Could not load GeoJSON from {url}.\nError: {e}")
            return

        # Auto-detect name field
        if region_name_field is None:
            candidates = [c for c in gdf.columns
                          if any(k in c.upper()
                                 for k in ['NAME', 'NOMBRE', 'NOMB', 'REGION', 'DEP'])]
            region_name_field = candidates[0] if candidates else gdf.columns[0]

        # Merge
        gdf['_key']        = gdf[region_name_field].str.upper().str.strip()
        df_metric['_key']  = df_metric['region'].str.upper().str.strip()
        gdf = gdf.merge(df_metric[['_key', metric]], on='_key', how='left')

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        gdf[gdf[metric].isna()].plot(ax=ax, color='#e0e0e0',
                                     edgecolor='white', linewidth=0.5)
        if gdf[metric].notna().any():
            gdf[gdf[metric].notna()].plot(
                ax=ax, column=metric, cmap=cmap,
                edgecolor='white', linewidth=0.5, legend=True,
                legend_kwds={
                    'label'      : 'APC (%)' if metric == 'apc' else 'p-value',
                    'orientation': 'horizontal',
                    'shrink'     : 0.5,
                },
            )

        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.set_axis_off()
        no_data = mpatches.Patch(color='#e0e0e0', label='No data')
        ax.legend(handles=[no_data], loc='lower left', frameon=False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Map saved: {save_path}")
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — LIFE-CYCLE SUBCLASS
# ═══════════════════════════════════════════════════════════════════════════════

class LifecycleAnalyzer(JoinpointAnalyzer):
    """
    Specialised subclass for life-cycle (age-group) joinpoint analysis.

    Adds ordered group display and a comparative APC bar chart across
    life-cycle stages, complementing the inherited trend and map plots.

    Standard life-cycle groups (customisable via group_order):
        Children · Adolescents · Young Adults · Adults · Older Adults

    Parameters
    ----------
    data : pd.DataFrame
        Must contain year_col, rate_col, and lifecycle_col.
    year_col : str
        Year column name.
    rate_col : str
        Rate/indicator column name.
    lifecycle_col : str
        Column containing age-group labels.
    group_order : list[str] or None
        Display order for age groups in plots and tables.
        Defaults to the five standard groups above.
    **kwargs
        Additional arguments forwarded to JoinpointAnalyzer.

    Examples
    --------
    >>> lc = LifecycleAnalyzer(
    ...     df, year_col='Year', rate_col='MortalityRate',
    ...     lifecycle_col='AgeGroup'
    ... )
    >>> lc.run()
    >>> lc.plot_trend(title='Mortality by Life-Cycle Group')
    >>> lc.plot_lifecycle_bars()
    >>> lc.plot_map(region_col='Region')
    """

    DEFAULT_ORDER = [
        'Children', 'Adolescents', 'Young Adults', 'Adults', 'Older Adults',
    ]

    def __init__(self, data, year_col, rate_col, lifecycle_col,
                 group_order=None, **kwargs):
        super().__init__(
            data=data, year_col=year_col, rate_col=rate_col,
            lifecycle_col=lifecycle_col, **kwargs,
        )
        self.group_order = group_order or self.DEFAULT_ORDER

    def plot_lifecycle_bars(
        self,
        title='APC by Life-Cycle Group and Segment',
        ylabel='Annual Percent Change (%)',
        figsize=(10, 6),
        save_path=None,
    ):
        """
        Grouped bar chart: APC per life-cycle group, coloured by segment.

        Bars above zero indicate increasing rates; below zero indicate
        decreasing rates. Error bars represent 95 % CI.

        Parameters
        ----------
        title : str
            Chart title.
        ylabel : str
            Y-axis label.
        figsize : tuple
            Figure size in inches.
        save_path : str or None
            Save figure at 300 dpi if provided.
        """
        if not self.results_:
            print("Run .run() before .plot_lifecycle_bars().")
            return

        summary = self.summary_table()
        if summary.empty:
            print("No results to plot.")
            return

        # Apply group ordering
        all_groups = list(summary['Group'].unique())
        ordered    = [g for g in self.group_order if g in all_groups]
        ordered   += [g for g in all_groups if g not in ordered]

        segments = list(summary['Segment'].unique())
        n_segs   = len(segments)
        n_groups = len(ordered)
        x        = np.arange(n_groups)
        width    = 0.8 / n_segs
        shades   = ['black', 'gray', 'darkgray', 'silver', 'lightgray']

        plt.style.use('grayscale')
        fig, ax = plt.subplots(figsize=figsize)

        for i, seg in enumerate(segments):
            sub    = summary[summary['Segment'] == seg].set_index('Group')
            apcs   = [sub.loc[g, 'APC (%)'] if g in sub.index else 0.0
                      for g in ordered]
            ci_lo, ci_hi = [], []
            for g in ordered:
                if g in sub.index:
                    raw = sub.loc[g, 'IC 95%'].strip('[]').split(',')
                    ci_lo.append(float(raw[0]))
                    ci_hi.append(float(raw[1]))
                else:
                    ci_lo.append(0.0); ci_hi.append(0.0)

            err_lo = [a - l for a, l in zip(apcs, ci_lo)]
            err_hi = [h - a for a, h in zip(apcs, ci_hi)]

            ax.bar(
                x + i * width - (n_segs - 1) * width / 2,
                apcs, width * 0.9,
                label=seg,
                color=shades[i % len(shades)],
                yerr=[err_lo, err_hi],
                capsize=3, error_kw={'linewidth': 0.8},
            )

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(ordered, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend(title='Segment', fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Lifecycle bar chart saved: {save_path}")
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_health_trend(
    data,
    year_col,
    rate_col,
    breakpoint_years=None,
    max_breakpoints=4,
    group_col=None,
    lifecycle_col=None,
    plot=True,
    plot_bic=False,
    **kwargs,
):
    """
    One-line wrapper: run analysis and return the summary table.

    Automatically selects LifecycleAnalyzer when lifecycle_col is provided,
    otherwise uses JoinpointAnalyzer.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    year_col : str
        Year column name.
    rate_col : str
        Rate/indicator column name.
    breakpoint_years : list[int] or None
        Force specific breakpoints (optional).
    max_breakpoints : int
        Maximum breakpoints to search for (default 4).
    group_col : str or None
        Stratification column (optional).
    lifecycle_col : str or None
        Life-cycle group column; triggers LifecycleAnalyzer.
    plot : bool
        Show trend plot (default True).
    plot_bic : bool
        Show BIC selection chart (default False).
    **kwargs
        Forwarded to the analyzer class.

    Returns
    -------
    pd.DataFrame
        Publication-ready summary table.

    Examples
    --------
    >>> summary = analyze_health_trend(
    ...     df, year_col='Year', rate_col='IncidenceRate',
    ...     lifecycle_col='AgeGroup', max_breakpoints=3
    ... )
    >>> print(summary)
    """
    if lifecycle_col:
        az = LifecycleAnalyzer(
            data=data, year_col=year_col, rate_col=rate_col,
            lifecycle_col=lifecycle_col,
            breakpoint_years=breakpoint_years,
            max_breakpoints=max_breakpoints, **kwargs,
        )
    else:
        az = JoinpointAnalyzer(
            data=data, year_col=year_col, rate_col=rate_col,
            breakpoint_years=breakpoint_years,
            max_breakpoints=max_breakpoints,
            group_col=group_col, **kwargs,
        )

    az.run()
    if plot:
        az.plot_trend()
    if plot_bic:
        az.plot_bic()
    return az.summary_table()
