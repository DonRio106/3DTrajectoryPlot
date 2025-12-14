```markdown
# Well trajectory + plan plotting (3D)

Small repo that reads a well trajectory and a planned trajectory (X,Y or MD/INC/AZI),
computes 3D coordinates (X,Y,TVD) using minimum-curvature and produces plots.

Quick start:
1. Create & activate venv:
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1

2. Install deps:
   pip install -r requirements.txt

3. Run (example using provided sample CSVs, units in feet):
   python example_well_profile_usage2.py --traj trajectory.xlsx --plan trajectoryplan.xlsx --units ft

Outputs:
- well_traj_xyz_<units>.csv
- well_plan_xyz_<units>.csv
- well_both_3d_plan_<units>.png
- well_both_3d_plan_<units>.html

Notes:
- Input XLSX files: you can specify sheet names with --traj-sheet / --plan-sheet.
- If your plan file contains MD/INC/AZI, it will be treated as a trajectory and will be processed similarly.
- For JSON API sources, pass a URL (and --token if authentication needed).
```
