# How to edit

- Change gains: edit `CBFBoxConfig.alpha`, `alpha1`, `alpha2`.
- Change solver: set `solver` to `closed_form`, `scipy`, or `cvxpy`.
- Add bounds: set `control_lower_bound` and `control_upper_bound`.
- Add new constraints: create a file under `constraints/` and register it in `builders.py`.
- Add a new solver: create a file under `optimization/` and dispatch from `api.py`.
