#!/usr/bin/env python3
"""
Compute and plot 3D well trajectory and a planned trajectory (plan) from XLSX/CSV or JSON API.

This variant supports:
- A trajectory file (--traj) containing MD, INC, AZI (or a JSON API that contains a trajectory list)
- An optional planned trajectory file (--plan) which can be either:
    * MD, INC, AZI (same format as trajectory) OR
    * explicit coordinates (columns X & Y or EAST & NORTH)
- Input files can be CSV or XLSX. For XLSX you can supply a sheet name with --traj-sheet / --plan-sheet.
- Units: meters (m) or feet (ft). Computations are unit-agnostic; labels and DLS default follow --units.

Outputs:
- CSVs with computed X,Y,TVD for each of trajectory and plan.
- Matplotlib PNG composite showing 3D view + plan view with BOTH traces.
- Plotly interactive HTML with BOTH traces.

Example:
  pip install -r requirements.txt
  python example_well_profile_usage.py --traj trajectory.csv --plan trajectoryplan.csv --units ft

"""
from __future__ import annotations
import sys
import os
import argparse
import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests

# ---------------------------
# core geometry
# ---------------------------
def min_curvature(md, inc_deg, azi_deg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    inc = np.deg2rad(inc_deg)
    azi = np.deg2rad(azi_deg)
    n = len(md)
    x = np.zeros(n, dtype=float)
    y = np.zeros(n, dtype=float)
    tvd = np.zeros(n, dtype=float)

    for i in range(n - 1):
        dl = md[i+1] - md[i]
        i1, i2 = inc[i], inc[i+1]
        a1, a2 = azi[i], azi[i+1]

        cos_dog = np.sin(i1)*np.sin(i2)*np.cos(a2-a1) + np.cos(i1)*np.cos(i2)
        cos_dog = np.clip(cos_dog, -1.0, 1.0)
        dogleg = np.arccos(cos_dog)

        if dogleg == 0.0:
            rf = 1.0
        else:
            rf = (2.0 / dogleg) * np.tan(dogleg / 2.0)

        dx = dl/2.0 * (np.sin(i1)*np.cos(a1) + np.sin(i2)*np.cos(a2)) * rf
        dy = dl/2.0 * (np.sin(i1)*np.sin(a1) + np.sin(i2)*np.sin(a2)) * rf
        dz = dl/2.0 * (np.cos(i1) + np.cos(i2)) * rf

        x[i+1] = x[i] + dx
        y[i+1] = y[i] + dy
        tvd[i+1] = tvd[i] + dz

    return x, y, tvd

def dogleg_and_dls(md, inc_deg, azi_deg, dls_unit='deg_per_30m') -> Tuple[np.ndarray, np.ndarray]:
    inc = np.deg2rad(inc_deg)
    azi = np.deg2rad(azi_deg)
    dogs = []
    dls = []
    for i in range(len(md)-1):
        cos_dog = np.sin(inc[i])*np.sin(inc[i+1])*np.cos(azi[i+1]-azi[i]) + np.cos(inc[i])*np.cos(inc[i+1])
        cos_dog = np.clip(cos_dog, -1.0, 1.0)
        dog = np.arccos(cos_dog)
        dogs.append(dog)
        dl = md[i+1] - md[i]
        if dl == 0:
            dls.append(0.0)
        else:
            deg_dog = np.degrees(dog)
            if dls_unit == 'deg_per_30m':
                dls.append(deg_dog * (30.0 / dl))
            elif dls_unit == 'deg_per_100ft':
                dls.append(deg_dog * (100.0 / dl))
            elif dls_unit == 'deg_per_m':
                dls.append(deg_dog / dl)
            else:
                try:
                    if isinstance(dls_unit, str) and dls_unit.startswith('deg_per_'):
                        x = float(dls_unit.split('_')[-1])
                        dls.append(deg_dog * (x / dl))
                    else:
                        dls.append(deg_dog / dl)
                except Exception:
                    dls.append(deg_dog / dl)
    return np.array(dogs), np.array(dls)

# ---------------------------
# API / JSON helpers
# ---------------------------
def fetch_json(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None, timeout: int = 15) -> Any:
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Response from {url} is not valid JSON: {e}")

def _looks_like_trajectory_list(obj: Any) -> bool:
    if not isinstance(obj, list) or len(obj) == 0:
        return False
    keys = set()
    for it in obj:
        if not isinstance(it, dict):
            return False
        keys.update(k.lower() for k in it.keys())
    return {'md', 'inc', 'azi'}.issubset(keys)

def find_trajectory_list_in_json(obj: Any) -> Optional[List[Dict[str, Any]]]:
    if _looks_like_trajectory_list(obj):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            found = find_trajectory_list_in_json(v)
            if found is not None:
                return found
    if isinstance(obj, list):
        for item in obj:
            found = find_trajectory_list_in_json(item)
            if found is not None:
                return found
    return None

def json_to_dataframe(json_obj: Any) -> pd.DataFrame:
    traj_list = find_trajectory_list_in_json(json_obj)
    if traj_list is None:
        raise RuntimeError("Could not find a trajectory list with MD/INC/AZI in the JSON response.")
    rows = []
    for rec in traj_list:
        if not isinstance(rec, dict):
            continue
        mapped = {k.upper(): v for k, v in rec.items()}
        rows.append(mapped)
    df = pd.DataFrame(rows)
    expected = ['MD','INC','AZI']
    cols_lower = {c.lower(): c for c in df.columns}
    for ex in expected:
        if ex not in df.columns:
            if ex.lower() in cols_lower:
                df[ex] = df[cols_lower[ex.lower()]]
            else:
                for alt in ['measured_depth','md','depth','dept','inc','inclination','azi','azimuth','az']:
                    if alt in df.columns:
                        df[ex] = df[alt]
                        break
    if not set(expected).issubset(set(df.columns)):
        raise RuntimeError(f"JSON parsed but missing required columns. Found: {list(df.columns)}")
    df = df[expected].apply(pd.to_numeric, errors='coerce')
    if df[['MD','INC','AZI']].isnull().any().any():
        raise RuntimeError("Some MD/INC/AZI values could not be converted to numbers.")
    return df

# ---------------------------
# input loaders (trajectory & plan)
# ---------------------------
def load_table_from_file(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    if path.lower().startswith(('http://','https://')):
        js = fetch_json(path)
        # JSON may be a trajectory or a wrapped object
        try:
            return json_to_dataframe(js)
        except Exception:
            # try to normalize raw json to DataFrame directly
            if isinstance(js, dict):
                return pd.DataFrame([js])
            return pd.DataFrame(js)
    if path.lower().endswith(('.xls', '.xlsx')):
        if sheet:
            df = pd.read_excel(path, sheet_name=sheet)
        else:
            df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df

def normalize_traj_df(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ('md','measured_depth','measureddepth','depth','dept'):
            col_map[c] = 'MD'
        elif cl in ('inc','inclination'):
            col_map[c] = 'INC'
        elif cl in ('azi','azimuth','az'):
            col_map[c] = 'AZI'
    df = df.rename(columns=col_map)
    if not {'MD','INC','AZI'}.issubset(set(df.columns)):
        raise RuntimeError("Trajectory must include MD, INC, AZI (case-insensitive) columns.")
    df = df[['MD','INC','AZI']].apply(pd.to_numeric, errors='coerce')
    if df.isnull().any().any():
        raise RuntimeError("Some MD/INC/AZI values could not be converted to numbers.")
    return df

def normalize_plan_df(df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    cols = [c.strip().lower() for c in df.columns]
    if any(c in ('md','measured_depth','measureddepth','depth','dept') for c in cols) and any(c in ('inc','inclination') for c in cols):
        df_traj = normalize_traj_df(df)
        return 'traj', df_traj
    mapping = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ('x','east','e'):
            mapping[c] = 'X'
        if cl in ('y','north','n'):
            mapping[c] = 'Y'
    if {'X','Y'}.issubset(set(mapping.values())):
        df_xy = df.rename(columns=mapping)
        df_xy = df_xy[['X','Y']].apply(pd.to_numeric, errors='coerce')
        if df_xy.isnull().any().any():
            raise RuntimeError("Some plan X/Y values could not be converted to numbers.")
        df_xy['TVD'] = 0.0
        return 'xy', df_xy
    raise RuntimeError("Plan file must contain either MD/INC/AZI (trajectory) or X/Y (coordinates).")

# ---------------------------
# plotting: both traces
# ---------------------------
def plot_both_matplotlib(traj, plan, labels: Tuple[str,str], color_vals_traj=None, color_vals_plan=None, filename=None, units='m'):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        print("Matplotlib not available; skipping matplotlib plot")
        return

    x_t, y_t, z_t = traj
    x_p, y_p, z_p = plan

    cmap = 'viridis'
    fig = plt.figure(figsize=(14, 6))
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')

    zt = -z_t
    zp = -z_p

    if color_vals_traj is None:
        ax3.plot(x_t, y_t, zt, '-o', color='tab:blue', label=labels[0])
    else:
        sc_t = ax3.scatter(x_t, y_t, zt, c=color_vals_traj, cmap=cmap, s=30, label=labels[0])
        cb = fig.colorbar(sc_t, ax=ax3, pad=0.05)
        cb.set_label('DLS (traj)')

    if color_vals_plan is None:
        ax3.plot(x_p, y_p, zp, '-o', color='tab:orange', label=labels[1])
    else:
        sc_p = ax3.scatter(x_p, y_p, zp, c=color_vals_plan, cmap='plasma', s=30, label=labels[1])
        cb2 = fig.colorbar(sc_p, ax=ax3, pad=0.1)
        cb2.set_label('DLS (plan)')

    ax3.plot(x_t, y_t, np.zeros_like(x_t), linestyle='--', color='tab:blue', alpha=0.4)
    ax3.plot(x_p, y_p, np.zeros_like(x_p), linestyle='--', color='tab:orange', alpha=0.4)

    unit_label = 'ft' if units == 'ft' else 'm'
    ax3.set_xlabel(f'East / X ({unit_label})')
    ax3.set_ylabel(f'North / Y ({unit_label})')
    ax3.set_zlabel(f'TVD ({unit_label}) (downwards)')
    ax3.view_init(elev=20, azim=-60)
    ax3.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x_t, y_t, '-o', color='tab:blue', label=labels[0])
    ax2.plot(x_p, y_p, '-o', color='tab:orange', label=labels[1])
    ax2.set_xlabel(f'East / X ({unit_label})')
    ax2.set_ylabel(f'North / Y ({unit_label})')
    ax2.set_title('Plan view (top-down)')
    ax2.grid(True)
    ax2.axis('equal')
    ax2.legend()
    plt.tight_layout()
    if filename:
        fig.savefig(filename, dpi=200)
        print("Saved matplotlib composite plot to", filename)
    else:
        plt.show()

def plot_both_plotly(traj, plan, labels: Tuple[str,str], color_vals_traj=None, color_vals_plan=None, filename="well_both_3d.html", units='m'):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        print("Plotly not available; skipping Plotly plot")
        return

    x_t, y_t, z_t = traj
    x_p, y_p, z_p = plan
    zt = -z_t
    zp = -z_p

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]],
                        subplot_titles=("3D Trajectories + Plan Projection", "Plan view (top-down)"))

    fig.add_trace(go.Scatter3d(x=x_t, y=y_t, z=zt, mode='lines+markers',
                               marker=dict(size=4, color=color_vals_traj, colorscale='Viridis', showscale=False),
                               line=dict(width=3), name=labels[0]), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=x_p, y=y_p, z=zp, mode='lines+markers',
                               marker=dict(size=4, color=color_vals_plan, colorscale='Plasma', showscale=False),
                               line=dict(width=3), name=labels[1]), row=1, col=1)

    fig.add_trace(go.Scatter3d(x=x_t, y=y_t, z=[0.0]*len(x_t), mode='lines', line=dict(width=2, dash='dash', color='lightblue'), name=f'{labels[0]} plan'), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=x_p, y=y_p, z=[0.0]*len(x_p), mode='lines', line=dict(width=2, dash='dash', color='lightcoral'), name=f'{labels[1]} plan'), row=1, col=1)

    fig.add_trace(go.Scatter(x=x_t, y=y_t, mode='lines+markers', marker=dict(size=6, color='blue'), name=labels[0]), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_p, y=y_p, mode='lines+markers', marker=dict(size=6, color='orange'), name=labels[1]), row=1, col=2)

    unit_label = 'ft' if units == 'ft' else 'm'
    fig.update_layout(scene=dict(xaxis_title=f'East / X ({unit_label})', yaxis_title=f'North / Y ({unit_label})', zaxis_title=f'TVD ({unit_label}) (downwards)'),
                      xaxis2=dict(title=f'East / X ({unit_label})'),
                      yaxis2=dict(title=f'North / Y ({unit_label})'),
                      margin=dict(l=0, r=0, t=40, b=0),
                      height=650, width=1300)
    fig.write_html(filename, auto_open=False)
    print("Saved interactive Plotly HTML to", filename)

# ---------------------------
# main CLI
# ---------------------------
def load_table_from_file(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    if path.lower().startswith(('http://','https://')):
        js = fetch_json(path)
        try:
            return json_to_dataframe(js)
        except Exception:
            if isinstance(js, dict):
                return pd.DataFrame([js])
            return pd.DataFrame(js)
    if path.lower().endswith(('.xls', '.xlsx')):
        if sheet:
            df = pd.read_excel(path, sheet_name=sheet)
        else:
            df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description="Plot trajectory and planned trajectory in 3D (X,Y,TVD) from XLSX/CSV or JSON.")
    p.add_argument('--traj', required=True, help="Trajectory file (CSV/XLSX) or JSON URL")
    p.add_argument('--plan', required=True, help="Planned trajectory file (CSV/XLSX) or JSON URL")
    p.add_argument('--traj-sheet', help="Sheet name for trajectory XLSX (optional)", default=None)
    p.add_argument('--plan-sheet', help="Sheet name for plan XLSX (optional)", default=None)
    p.add_argument('--out-prefix', help='Output prefix for CSV/plots (default: well)', default='well')
    p.add_argument('--units', choices=['m','ft'], default='m', help="Units for MD/TVD: 'm' or 'ft' (default: m)")
    p.add_argument('--no-plotly', action='store_true', help="Disable Plotly HTML output")
    p.add_argument('--dls-unit', help="DLS output unit. Default: deg_per_100ft for ft, deg_per_30m for m", default=None)
    p.add_argument('--token', help="API token for HTTP sources (Bearer by default). Can also set API_TOKEN env var.", default=os.environ.get('API_TOKEN'))
    p.add_argument('--token-header', help="Header name for token (default Authorization)", default='Authorization')
    args = p.parse_args(argv)

    df_traj_raw = load_table_from_file(args.traj, sheet=args.traj_sheet)
    df_traj = normalize_traj_df(df_traj_raw)

    df_plan_raw = load_table_from_file(args.plan, sheet=args.plan_sheet)
    plan_type, df_plan = normalize_plan_df(df_plan_raw)

    dls_unit = args.dls_unit or ('deg_per_100ft' if args.units == 'ft' else 'deg_per_30m')

    md_t = df_traj['MD'].to_numpy()
    inc_t = df_traj['INC'].to_numpy()
    azi_t = df_traj['AZI'].to_numpy()
    x_t, y_t, tvd_t = min_curvature(md_t, inc_t, azi_t)
    dogs_t, dls_t = dogleg_and_dls(md_t, inc_t, azi_t, dls_unit=dls_unit)

    if plan_type == 'traj':
        md_p = df_plan['MD'].to_numpy()
        inc_p = df_plan['INC'].to_numpy()
        azi_p = df_plan['AZI'].to_numpy()
        x_p, y_p, tvd_p = min_curvature(md_p, inc_p, azi_p)
        dogs_p, dls_p = dogleg_and_dls(md_p, inc_p, azi_p, dls_unit=dls_unit)
    else:
        x_p = df_plan['X'].to_numpy()
        y_p = df_plan['Y'].to_numpy()
        tvd_p = df_plan['TVD'].to_numpy() if 'TVD' in df_plan.columns else np.zeros_like(x_p)
        dls_p = np.zeros(max(0, len(x_p)-1))

    unit_label = 'ft' if args.units == 'ft' else 'm'
    out_traj_csv = f"{args.out_prefix}_traj_xyz_{unit_label}.csv"
    out_plan_csv = f"{args.out_prefix}_plan_xyz_{unit_label}.csv"

    df_out_traj = df_traj.copy()
    df_out_traj['X'] = x_t; df_out_traj['Y'] = y_t; df_out_traj['TVD'] = tvd_t
    df_out_traj['DLS'] = np.append(dls_t, dls_t[-1] if len(dls_t)>0 else 0.0)
    df_out_traj.to_csv(out_traj_csv, index=False)
    print("Wrote", out_traj_csv)

    df_out_plan = pd.DataFrame({'X': x_p, 'Y': y_p, 'TVD': tvd_p})
    df_out_plan['DLS'] = np.append(dls_p, dls_p[-1] if len(dls_p)>0 else 0.0)
    df_out_plan.to_csv(out_plan_csv, index=False)
    print("Wrote", out_plan_csv)

    labels = ("Trajectory", "Planned trajectory")
    color_vals_traj = df_out_traj['DLS'].to_numpy()
    color_vals_plan = df_out_plan['DLS'].to_numpy() if len(df_out_plan)>0 else None

    mpl_fn = f"{args.out_prefix}_both_3d_plan_{unit_label}.png"
    plot_both_matplotlib((x_t, y_t, tvd_t), (x_p, y_p, tvd_p), labels,
                         color_vals_traj=color_vals_traj, color_vals_plan=color_vals_plan,
                         filename=mpl_fn, units=args.units)

    if not args.no_plotly:
        plotly_fn = f"{args.out_prefix}_both_3d_plan_{unit_label}.html"
        plot_both_plotly((x_t, y_t, tvd_t), (x_p, y_p, tvd_p), labels,
                         color_vals_traj=color_vals_traj, color_vals_plan=color_vals_plan,
                         filename=plotly_fn, units=args.units)

if __name__ == '__main__':
    main()
