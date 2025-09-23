# Welcome to `meenpy`!

This library contains a few classes enabling 

## Installation and Usage Instructions

# MEENPY

`meenpy` is a Python library of utilities for mechanical engineering.

`meenpy.numerics` enables the succinct articulation equations, efficient assembly of systems, and transparent solving interface.

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

# Mechanical System
## Variable Definitions
f, f1, f2, f3, m, a1, a2, a3 = symb("f, f1, f2, f3, m, a1, a2, a3")
F = Mat([f1, f2, f3])
A = Mat([a1, a2, a3])

## Equation Definitions
newtons_second_law = meqn(F, m * A)
force_magnitude = seqn(f, sym.sqrt(f1**2 + f2**2 + f3**2))

## System Composition and Solution
mechanical_system = sys([newtons_second_law, force_magnitude])
sol = mechanical_system.solve({a1: 1, a2: 1, a3: 1, m: 1})

# Thermodynamic System
## Variable Definitions
T, x, p, v = symb("T, x, p, v")
T_amb, h, Qd = symb("T_amb, h, Qd")
column_map = {T: "Temperature", x: "Quality", p: "Pressure", v: "Specific Volume"}

## Equation Definitions
water = teqn(read_csv("test/water.csv"), ["Temperature", "Quality"], residual_type="all_column_differential")
convection = seqn(Qd, h * (T - T_amb))

## System Composition and Solution
water_system = sys([water, convection], column_map)
sol = water_system.solve({T_amb: 25, h: 1, Qd: 100}, {T: 115, x: 1}, verbosity=0)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)