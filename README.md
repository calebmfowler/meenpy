# meenpy

`meenpy` is a library of utilities for mechanical engineering problem solving.

`meenpy.numerics` enables the succinct articulation equations, efficient assembly of systems, and provides a transparent solving interface.

## Installation

Check out the package manager [uv](https://docs.astral.sh/uv/) and add `meenpy` to your project or workspace.

```bash
uv add meenpy
```

Alternatively, install `meenpy` directly with [pip](https://pip.pypa.io/en/stable/).

```bash
pip install meenpy
```

## Usage

```python
from meenpy.numerics import ScalarEquation, MatrixEquation, TabularEquation, System
import sympy as sym, pandas as pd

## Variable Definitions
T, x, p, v = sym.symbols("T, x, p, v")
T_amb, h, Qd = sym.symbols("T_amb, h, Qd")
column_map = {T: "Temperature", x: "Quality", p: "Pressure", v: "Specific Volume"}

## Equation Definitions
water = TabularEquation(pd.read_csv("test/water.csv"), ["Temperature", "Quality"], residual_type="all_column_differential")
convection = ScalarEquation(Qd, h * (T - T_amb))

## System Composition and Solution
water_system = System([water, convection], column_map)
water_system_solution = water_system.solve({T_amb: 25, h: 1, Qd: 100}, {T: 115, x: 1})
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)