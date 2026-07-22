# Four-Zone Hamilton–Jacobi + Poisson-CBF Landing Demo

Este proyecto implementa un ejemplo reproducible de aterrizaje contingente con:

- **4 zonas de aterrizaje** (`p = 4`).
- Requisito de conservar **al menos 2 zonas alcanzables** (`r = 2`).
- Un target activo inicial (`LZ-1`).
- Descubrimiento durante el vuelo de un obstáculo que ocupa `LZ-1`.
- Rechazo de `LZ-1`, cambio automático a una alternativa alcanzable y reinicio del horizonte activo de acuerdo con la lógica del paper.
- Campo de seguridad 3-D creado con el paquete real `poisson_safety_box`.
- Restricción CBF creada con el paquete real `cbf_safety_box`.
- QP unificado que impone simultáneamente seguridad Poisson y las restricciones reach-avoid combinatoriales.
- Exportación de matrices, CSV, diagnósticos y figuras completas.

## Estructura esperada

```text
workspace/
├── poisson_safety_box/
├── cbf_safety_box/
└── hj_combinatorial_landing_demo/
```

## Ejecución

```bash
cd hj_combinatorial_landing_demo
python3 -m pip install -r requirements.txt
python3 run_demo.py
```

También se pueden pasar rutas explícitas:

```bash
python3 run_demo.py \
  --poisson-box ../poisson_safety_box \
  --cbf-box ../cbf_safety_box \
  --output-dir outputs/four_zone_demo
```

## Modelo Hamilton–Jacobi implementado

Para hacer el ejemplo verificable y computacionalmente razonable, reachability se calcula para el modelo reducido

\[
\dot p = u, \qquad \|u\|_2\le v_{\max},
\]

con posición tridimensional `p` y comando de velocidad `u`. Para este modelo isotrópico y obstáculos estáticos, la función reach-avoid es

\[
V_j(p,\tau)=v_{\max}(-\tau)-D_j(p),
\]

donde `D_j` es la distancia geodésica mínima hasta la zona `j` sin atravesar la ocupación. La zona `j` es alcanzable cuando `V_j >= 0`.

Esto constituye una realización HJ/Eikonal exacta para el **modelo de integrador simple**, pero **no** es todavía HJR de 6-DOF del dron. En una integración con PX4, `u_safe` se interpretaría como un setpoint de velocidad que los controladores internos de PX4 deben seguir.

## Filtro combinatorial del paper

El QP utiliza:

1. Una restricción para mantener alcanzable el target activo con horizonte decreciente `tau1`.
2. Cuatro restricciones, una por landing zone, que preservan el conjunto `r-out-of-p` con `r=2`.
3. Una restricción Poisson-CBF obtenida directamente desde `cbf_safety_box`.
4. Límites de velocidad.

La función pivote es el segundo valor más grande:

\[
\widetilde h(p,\tau_2)=\max^{(2)}\{V_1,V_2,V_3,V_4\}.
\]

`tilde h >= 0` significa que al menos dos zonas siguen alcanzables.

## Salidas

La carpeta `outputs/four_zone_demo/figures` incluye:

1. Arquitectura completa.
2. Mundo y trayectoria 3-D.
3. Matrices de ocupación antes/después.
4. Matrices de frontera Dirichlet.
5. Matrices del forcing de Poisson.
6. Matrices completas de `h` en cortes 3-D.
7. Gradiente, Hessiano y Laplaciano.
8. Isosuperficies 3-D de `h`.
9. Las cuatro matrices de valor HJ antes/después.
10. Distancias geodésicas, función pivote y número de zonas alcanzables.
11. Historial temporal de `V1...V4`, pivote y switching.
12. `h(t)`, residuales CBF/HJ y controles.
13. Variables auxiliares y tiempo del QP.
14. Dashboard integrado.
15. Isosuperficies 3-D de los cuatro conjuntos reach-avoid antes y después de la detección.

La carpeta `data` contiene los arrays completos `.npz`, el log `.csv` y los resúmenes `.json`.

## Nota científica

El obstáculo de `LZ-1` no se incluye en el mapa previo. Cuando se descubre, el programa:

1. actualiza el occupancy;
2. recalcula Poisson;
3. recalcula los cuatro campos de reachability;
4. marca `LZ-1` como rechazada;
5. selecciona una alternativa con `V_j >= 0`;
6. reinicia el horizonte activo usando el horizonte de contingencia;
7. continúa manteniendo `r=2` alternativas cuando es factible.

## Resultado de la ejecución verificada

Con la configuración incluida, la ejecución de referencia produjo:

- `landed: true`
- `collision: false`
- cambio de `LZ-1` a `LZ-3` después de detectar el obstáculo
- mínimo de **3 zonas alcanzables**, superior al requisito `r=2`
- función pivote mínima positiva
- 100 % de QP resueltos satisfactoriamente
- tiempo medio del QP cercano a 1.1 ms en este entorno de prueba

Los números exactos se guardan en `outputs/four_zone_demo/summary.json`.
