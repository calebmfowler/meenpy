import unittest
from meenpy.numerics import ScalarEquation as seqn, MatrixEquation as meqn, System as sys, TabularEquation as teqn
from meenpy.numerics.utils import *

class MEENPyTest(unittest.TestCase):
    def test_scalar_equation(self):
        y, m, x, b = symb("y, m, x, b")
        line = seqn(y, m * x + b)

        sol = line.solve({y: 0, m: 1, b: -1}, guess=2, verbosity=0)
        self.assertEqual(sol, dict({x: 1}))
        
        pass

    def test_size_1_matrix_equation(self):
        f, f1, f2, f3 = symb("f, f1, f2, f3")
        F = Mat([f1, f2, f3])
        vector_magnitude = meqn(Mat([f]), sym.sqrt(F.T @ F))

        sol = vector_magnitude.solve({f1: 1, f2: 1, f3: 1}, guess_dict={f: 2}, verbosity=0)
        result = [sol[variable] for variable in [f]]
        expected = [np.sqrt(3)]
        self.assertEqual(result, expected)

        pass

    def test_vector_matrix_equation(self):
        a1, a2, a3, a4, x1, x2, b1, b2 = symb("a1, a2, a3, a4, x1, x2, b1, b2")
        A = Mat([
            [a1, a2],
            [a3, a4]
        ])
        X = Mat([x1, x2])
        B = Mat([b1, b2])
        line = meqn(A @ X, B)

        sol = line.solve({a1: 1, a2: -1, a3: 1, a4: 1, b1: 0, b2: 2}, guess_dict={x1: 2, x2: 2}, verbosity=0)
        result = [sol[variable] for variable in [x1, x2]]
        expected = [npfloat(1), npfloat(1)]
        self.assertEqual(result, expected)

        pass

    def test_square_matrix_equation(self):
        a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4 = symb("a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4")
        A = Mat([
            [a1, a2],
            [a3, a4]
        ])
        B = Mat([
            [b1, b2],
            [b3, b4]
        ])
        C = Mat([
            [c1, c2],
            [c3, c4]
        ])

        eqn = meqn(A @ B, C, residual_type="right_inversion")

        sol = eqn.solve({a1: 1, a2: 0, a3: 1, a4: 1, c1: 1, c2: 0, c3: 1, c4: 1}, verbosity=0)
        result = [round(sol[variable], 6) for variable in [b1, b2, b3, b4]]
        expected = [npfloat(1), npfloat(0), npfloat(0), npfloat(1)]
        self.assertEqual(result, expected)

        pass

    def test_scalar_matrix_system(self):
        f, f1, f2, f3, m, a1, a2, a3 = symb("f, f1, f2, f3, m, a1, a2, a3")
        F = Mat([f1, f2, f3])
        A = Mat([a1, a2, a3])
        newtons_second_law = meqn(F, m * A)
        force_magnitude = seqn(f, sym.sqrt(f1**2 + f2**2 + f3**2))

        mechanical_system = sys([newtons_second_law, force_magnitude])

        sol = mechanical_system.solve({a1: 1, a2: 1, a3: 1, m: 1}, guess_dict={f: 2, f1: 2, f2: 2, f3: 2}, verbosity=0)
        result = [round(sol[variable], 6) for variable in [f, f1, f2, f3]]
        expected = [round(np.sqrt(3), 6), npfloat(1), npfloat(1), npfloat(1)]
        self.assertEqual(result, expected)

        pass

    def test_tabular_equation(self):
        water_table = teqn(read_csv("test/water.csv"), ["Temperature", "Quality"])
        sol = water_table.solve({"Specific Volume": 0.7505}, verbosity=0, xtol=1e-25)
        result = [round(sol[variable], 6) for variable in ["Temperature", "Quality", "Pressure", "Specific Volume"]]
        expected = [npfloat(100), npfloat(0.5), npfloat(1), npfloat(0.7505)]
        self.assertEqual(result, expected)

        pass

    def test_scalar_tabular_system(self):
        T, x, p, v = symb("T, x, p, v")
        column_map = {T: "Temperature", x: "Quality", p: "Pressure", v: "Specific Volume"}
        water = teqn(read_csv("test/water.csv"), ["Temperature", "Quality"], residual_type="all_column_differential")
        
        T_amb, h, Qd = symb("T_amb, h, Qd")
        convection = seqn(Qd, h * (T - T_amb))

        water_system = sys([water, convection], column_map)
        sol = water_system.solve({T_amb: 25, h: 1, v: 0.7505}, verbosity=0)
        result = [round(sol[variable], 6) for variable in [Qd, T, x, p]]
        expected = [npfloat(75), npfloat(100), npfloat(0.5), npfloat(1)]
        self.assertEqual(result, expected)

    pass

if __name__ == "__main__":
    unittest.main()
