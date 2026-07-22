# Run guide

```bash
pip install -r requirements.txt
pip install -e .
python examples/run_velocity_cbf_basic.py
python examples/run_acceleration_hocbf_basic.py
python examples/run_backstepping_basic.py
python examples/run_compare_qp_solvers.py
pytest -q
```
