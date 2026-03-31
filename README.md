# Mahalanobis Distance Cross-Platform Investigation Summary

## Background
A simple outlier detection script using Mahalanobis distance produced **different results across platforms**. The investigation traced the root cause through OS, package manager, CPU architecture, and BLAS backend.

## Root Cause
The dataset used for initial testing had **perfectly collinear data** (`y = x`), making the covariance matrix singular. Inverting a singular matrix via `np.linalg.inv()` is mathematically undefined — and different BLAS backends handle this differently, exposing platform inconsistencies.

```python
X = np.array([[10, 10], [12, 12], [11, 11], [14, 14], [100, 100], [14, 14]])
# All points on y=x → singular covariance matrix → unstable inversion
```

## Main Finding: Linux vs macOS

| Environment | OS | BLAS | OpenBLAS Kernel | inv_cov [0,0] | Distance | Outlier Detected? |
|---|---|---|---|---|---|---|
| Intel Mac conda | macOS | OpenBLAS 0.3.21 | CORE2 | 4.398e+12 | 4.572 | ❌ Borderline |
| Intel Mac uv | macOS | OpenBLAS64 0.3.23 | SANDYBRIDGE | 4.398e+12 | 4.572 | ❌ Borderline |
| Intel Mac container | Linux | OpenBLAS 0.3.21 | PRESCOTT | 4.398e+12 | 2.614 | ❌ No |
| Intel Bootcamp WSL | Linux | OpenBLAS 0.3.21 | PRESCOTT | 4.398e+12 | 2.614 | ❌ No |

### Why macOS and Linux diverge
Despite **same NumPy version, same CPU, same OpenBLAS version** — the results differ because `DYNAMIC_ARCH=1` selects the CPU kernel at runtime based on OS:

```
macOS  → selects CORE2 / SANDYBRIDGE  → distance ~4.572
Linux  → selects PRESCOTT             → distance ~2.614
```

PRESCOTT uses an older x87 FPU instruction path. CORE2 and SANDYBRIDGE use SSE2-optimised paths. On a singular matrix, these two paths accumulate floating point errors differently, producing divergent results.

Notably, **package manager did not matter** — Intel Mac conda and uv gave identical results (4.572), confirming the OS-driven kernel selection is the true differentiator.

## Side Notes

### Apple M4 Mac
| Environment | BLAS | Kernel | Distance |
|---|---|---|---|
| M4 Mac conda | OpenBLAS 0.3.21 | VORTEX (ARM) | 5.227 |

M4 uses the **VORTEX** kernel — ARM's NEON/ASIMD instruction set — which takes a completely different floating point path from any x86 kernel, producing the most divergent result (5.227). This is purely an ARM vs x86 architecture difference, not an OS issue.

### Windows
| Environment | BLAS | Behavior |
|---|---|---|
| Windows conda | **MKL 2023.1** | ✅ Raises `LinAlgError: Singular matrix` |

Windows conda defaults to **Intel MKL** instead of OpenBLAS. MKL performs strict pivot checking during inversion and **correctly refuses** to invert the singular matrix. Ironically, Windows was the only platform that exposed the real bug — all other platforms silently returned meaningless results.

## Key Takeaways

| Factor | Role |
|---|---|
| NumPy version | ❌ Not the cause — all platforms used 1.26.0 |
| Package manager (conda vs uv) | ❌ Not the cause — same results on same OS |
| CPU chipset | ❌ Not the cause for Intel platforms |
| **OS** | ✅ Determines which BLAS backend and OpenBLAS kernel is selected |
| **BLAS backend (MKL vs OpenBLAS)** | ✅ Primary cause of behavioral difference |
| **OpenBLAS CPU kernel** | ✅ Secondary cause within OpenBLAS platforms |
| **Singular matrix (the data)** | ✅ The true root cause — all platform differences vanish with well-conditioned data |

## Resolution
Replacing the collinear dataset with well-conditioned data produced **identical results across all platforms**:

```python
# New dataset — full rank, no collinearity
X = np.array([
    [10.0, 10.2], [10.2, 9.8], [9.9, 10.1],
    [10.1, 10.0], [10.0, 9.9],
    [14.0, 18.0],  # Outlier A
    [25.0, 5.0]    # Outlier B
], dtype=float)
# All platforms → distance 5.13 and 5.14 → both outliers correctly detected ✅
```

## Reference: Original Code to Compute the Mahalanobis Distance

```python
# Outlier detection using Mahalanobis distance
from scipy.stats import chi2
X = np.array([[10, 10], [12, 12], [11, 11], [14, 14], [100, 100], [14, 14]])
# Calculate mean and covariance matrix

mean = np.mean(X, axis=0)
cov_matrix = np.cov(X.T)
inv_cov_matrix = np.linalg.inv(cov_matrix)
mahalanobis_distances = np.array([np.dot(np.dot((x - mean), inv_cov_matrix), (x - mean).T) for x in X])
critical_value = chi2.ppf((1-0.1), df=2)  # df is the number of dimensions
outliers = np.where(mahalanobis_distances > critical_value)

```
