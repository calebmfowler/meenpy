from meenpy.numerics import ScalarEquation, MatrixEquation, TabularEquation, System
import pandas as pd
import sympy as sym

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