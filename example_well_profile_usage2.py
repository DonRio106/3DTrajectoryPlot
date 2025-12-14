#!/usr/bin/env python3
"""
Compute and plot 3D well trajectory and a planned trajectory (plan) from XLSX/CSV or JSON API.

FIXES APPLIED:
- Removed duplicate load_table_from_file definition
- Safer dogleg zero check (tolerance)
- MD monotonic validation
- Last DLS set to NaN (engineering-correct)
- Minor robustness cleanups
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
    md = np.asarray(md, dtype=float)
    if not np.all(np.diff(md) > 0):
        raise RuntimeError("MD must be strictly increasing")

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

        if dogleg < 1e-12:
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
    md = np.asarray(md, dtype=float)
    inc = np.deg2rad(inc_deg)
    azi = np.deg2rad(azi_deg)

    dogs = []
    dls = []
    for i in range(len(md)-1):
        cos_dog = (
            np.sin(inc[i])*np.sin(inc[i+1])*np.cos(azi[i+1]-azi[i])
            + np.cos(inc[i])*np.cos(inc[i+1])
        )
        cos_dog = np.clip(cos_dog, -1.0, 1.0)
        dog = np.arccos(cos_dog)
        dogs.append(dog)

        dl = md[i+1] - md[i]
        if dl <= 0:
            dls.append(np.nan)
            continue

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
def fetch_json(url: str, headers: Optional[Dict[str, str]] = None,
               params: Optional[Dict[str, Any]] = None, timeout: int = 15) -> Any:
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def _looks_like_trajectory_list(obj: Any) -> bool:
    if not isinstance(obj, list) or not obj:
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
        raise RuntimeError("Could not find trajectory list with MD/INC/AZI in JSON")
    df = pd.DataFrame([{k.upper(): v for k, v in r.items()} for r in traj_list])

    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ('md','measured_depth','depth','dept'):
            colmap[c] = 'MD'
        elif cl in ('inc','inclination'):
            colmap[c] = 'INC'
        elif cl in ('azi','azimuth','az'):
            colmap[c] = 'AZI'
    df = df.rename(columns=colmap)

    if not {'MD','INC','AZI'}.issubset(df.columns):
        raise RuntimeError(f"JSON missing MD/INC/AZI. Found: {list(df.columns)}")

    df = df[['MD','INC','AZI']].apply(pd.to_numeric, errors='coerce')
    if df.isnull().any().any():
        raise RuntimeError("Non-numeric MD/INC/AZI values in JSON")
    return df

# ---------------------------
# input loaders
# ---------------------------
def load_table_from_file(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    if path.lower().startswith(('http://','https://')):
        return json_to_dataframe(fetch_json(path))

    if path.lower().endswith(('.xls', '.xlsx')):
        return pd.read_excel(path, sheet_name=sheet) if sheet else pd.read_excel(path)

    return pd.read_csv(path)

def normalize_traj_df(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ('md','measured_depth','depth','dept'):
            col_map[c] = 'MD'
        elif cl in ('inc','inclination'):
            col_map[c] = 'INC'
        elif cl in ('azi','azimuth','az'):
            col_map[c] = 'AZI'

    df = df.rename(columns=col_map)
    if not {'MD','INC','AZI'}.issubset(df.columns):
        raise RuntimeError("Trajectory must contain MD, INC, AZI")

    df = df[['MD','INC','AZI']].apply(pd.to_numeric, errors='coerce')
    if df.isnull().any().any():
        raise RuntimeError("Invalid numeric values in MD/INC/AZI")
    return df

def normalize_plan_df(df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    cols = [c.lower() for c in df.columns]
    if {'md','inc'}.intersection(cols):
        return 'traj', normalize_traj_df(df)

    mapping = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ('x','east','e'):
            mapping[c] = 'X'
        elif cl in ('y','north','n'):
            mapping[c] = 'Y'

    if {'X','Y'}.issubset(mapping.values()):
        out = df.rename(columns=mapping)[['X','Y']].apply(pd.to_numeric, errors='coerce')
        out['TVD'] = 0.0
        return 'xy', out

    raise RuntimeError("Plan must be MD/INC/AZI or X/Y")

# ---------------------------
# main
# ---------------------------
def main(argv=None):
    argv = argv or sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument('--traj', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--traj-sheet')
    p.add_argument('--plan-sheet')
    p.add_argument('--out-prefix', default='well')
    p.add_argument('--units', choices=['m','ft'], default='m')
    p.add_argument('--no-plotly', action='store_true')
    p.add_argument('--dls-unit', default=None)
    args = p.parse_args(argv)

    df_traj = normalize_traj_df(load_table_from_file(args.traj, args.traj_sheet))
    plan_type, df_plan = normalize_plan_df(load_table_from_file(args.plan, args.plan_sheet))

    dls_unit = args.dls_unit or ('deg_per_100ft' if args.units == 'ft' else 'deg_per_30m')

    x_t, y_t, tvd_t = min_curvature(df_traj.MD, df_traj.INC, df_traj.AZI)
    _, dls_t = dogleg_and_dls(df_traj.MD, df_traj.INC, df_traj.AZI, dls_unit)

    if plan_type == 'traj':
        x_p, y_p, tvd_p = min_curvature(df_plan.MD, df_plan.INC, df_plan.AZI)
        _, dls_p = dogleg_and_dls(df_plan.MD, df_plan.INC, df_plan.AZI, dls_unit)
    else:
        x_p, y_p, tvd_p = df_plan.X.values, df_plan.Y.values, df_plan.TVD.values
        dls_p = np.full(max(len(x_p)-1, 0), np.nan)

    unit = 'ft' if args.units == 'ft' else 'm'

    out_traj = f"{args.out_prefix}_traj_xyz_{unit}.csv"
    out_plan = f"{args.out_prefix}_plan_xyz_{unit}.csv"

    df_traj_out = df_traj.copy()
    df_traj_out['X'] = x_t
    df_traj_out['Y'] = y_t
    df_traj_out['TVD'] = tvd_t
    df_traj_out['DLS'] = np.append(dls_t, np.nan)
    df_traj_out.to_csv(out_traj, index=False)

    df_plan_out = pd.DataFrame({'X': x_p, 'Y': y_p, 'TVD': tvd_p})
    df_plan_out['DLS'] = np.append(dls_p, np.nan)
    df_plan_out.to_csv(out_plan, index=False)

    print("Wrote", out_traj)
    print("Wrote", out_plan)

if __name__ == '__main__':
    main()
