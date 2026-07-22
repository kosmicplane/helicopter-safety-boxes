# Connection with poisson_safety_box

The Poisson box should provide local samples:

```text
h(p)
grad_h(p)
hessian_h(p)
```

Create a `SafetySample`:

```python
from cbf_safety_box import SafetySample
safety = SafetySample(h=h_value, grad_h=grad_h_value, hessian_h=H_value)
```

Then filter a nominal command:

```python
result = cbf_box.filter_control(state, safety, u_nom)
u_safe = result.u_safe
```

This box intentionally uses duck typing for Poisson adapters so that both boxes remain independent.
