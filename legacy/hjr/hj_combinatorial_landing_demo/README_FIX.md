# Poisson-CBF contingency boxes patch v2

The earlier import failure occurs when the new contingency runtime is placed
beside the **original baseline** `cbf_safety_box`.  That baseline does not export
`AffineCertificate` and does not contain the multi-row contingency QP API.

This patch includes the complete corrected sources for both mandatory boxes,
the commented runtime, and an executable verification script.

## Install into an existing Helicopter workspace

```bash
cd /path/to/extracted/poisson_cbf_contingency_patch_v2
./apply_patch_and_verify.sh ~/ATMOS/Docker/workspace/Helicopter
```

The installer backs up the existing CBF box as
`cbf_safety_box_before_contingency_patch`, overlays the corrected boxes, clears
stale Python bytecode, and executes `verify_box_integration.py`.

## Manual installation

From the Helicopter directory:

```bash
unzip -o poisson_cbf_contingency_patch_v2.zip -d /tmp/poisson_patch_v2
rsync -a /tmp/poisson_patch_v2/cbf_safety_box/ ./cbf_safety_box/
rsync -a /tmp/poisson_patch_v2/poisson_safety_box/ ./poisson_safety_box/
cp /tmp/poisson_patch_v2/run_contingency_study_with_boxes.py .
cp /tmp/poisson_patch_v2/verify_box_integration.py .
find cbf_safety_box poisson_safety_box -type d -name __pycache__ -prune -exec rm -rf {} +
python verify_box_integration.py
```

Expected final line:

```text
PASS: PoissonSafetyBox and the extended CBFBox are correctly installed and executed.
```
