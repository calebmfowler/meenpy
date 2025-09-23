from scipy.optimize import fsolve
from sympy import sympify, lambdify, Basic, Expr, Matrix, Identity
from numpy import array as nparr, ndarray, float64 as npfloat, concatenate, prod, sum, argmin, shape, vectorize, ones
from typing import Callable, get_type_hints
from pandas import Series, DataFrame, concat

usernum = int | float | npfloat

class Equation:
    def __init__(self):
        self._init_size()

    def _init_size(self):
        self.size: int = -1

    def get_subbed(self, subs: dict) -> "Equation":
        return self
    
    def get_lambda_residual(self, subs: dict = {}) -> tuple[Callable, list]:
        return (lambda *args, **kwargs: None, [])
    
    def solve(self, subs: dict = {}) -> dict:
        return {}


class AlgebraicEquation(Equation):
    def __init__(self, lhs, rhs, residual_type = "differential") -> None:
        self._init_expression(lhs, rhs)
        self._init_shape()
        self._init_size()
        self._init_variables()
        self._init_residual_type(residual_type)
       
    def _init_expression(self, lhs, rhs) -> None:
        self.lhs = sympify(lhs)
        self.rhs = sympify(rhs)
    
    def _init_shape(self) -> None:
        self.shape: tuple[int, int] = (-1, -1)

    def _init_variables(self) -> None:
        self.variables: list[Basic] = []
    
    def _init_residual_type(self, residual_type):
        self.residual_type = residual_type

    def get_subbed(self, subs: dict[Basic, usernum]) -> "AlgebraicEquation":
        return self

    def get_residual(self, subs: dict[Basic, usernum] = {}) -> Expr | Matrix:
        return Expr()
    
    def get_lambda_residual(self, subs: dict[Basic, usernum] = {}) -> tuple[Callable, list[Basic]]:
        return (lambda *args, **kwargs: None, [])
    
    def solve(self, subs: dict[Basic, usernum] = {}) -> dict[Basic, npfloat]:
        return {}

    def __str__(self) -> str:
        return self.lhs.__str__() + ' = ' + self.rhs.__str__()


class ScalarEquation(AlgebraicEquation):
    def _init_expression(self, lhs, rhs) -> None:
        self.lhs: Expr = sympify(lhs)
        self.rhs: Expr = sympify(rhs)
    
    def _init_shape(self) -> None:
        self.shape: tuple[int, int] = (1, 1)

    def _init_size(self):
        self.size = 1

    def _init_variables(self) -> None:
        self.variables: list[Basic] = list(self.lhs.free_symbols | self.rhs.free_symbols)
        self.variables.sort(key = lambda x: str(x))

    def _init_residual_type(self, residual_type: str):
        if residual_type not in ["differential", "left_rational", "right_rational"]:
            raise ValueError(f"Invalid residual_type = '{residual_type}'")
        self.residual_type = residual_type
    
    def get_subbed(self, subs: dict[Basic, usernum]) -> "ScalarEquation":
        return ScalarEquation(self.lhs.subs(subs), self.rhs.subs(subs))

    def get_residual(self, subs: dict[Basic, usernum] = {}) -> Expr:
        subbed_eqn = self.get_subbed(subs)

        if self.residual_type == "differential":
            return sympify(subbed_eqn.lhs) - sympify(subbed_eqn.rhs)
        
        elif self.residual_type == "left_rational":
            return sympify(subbed_eqn.lhs) / sympify(subbed_eqn.rhs) - 1
        
        elif self.residual_type == "right_rational":
            return sympify(subbed_eqn.rhs) / sympify(subbed_eqn.lhs) - 1
        
        else:
            raise ValueError(f"Invalid residual_type = '{self.residual_type}'")
    
    def get_lambda_residual(self, subs: dict[Basic, usernum] = {}) -> tuple[Callable[[ndarray], usernum], list[Basic]]:
        residual = self.get_residual(subs)
        residual_variables = list(residual.free_symbols)
        residual_variables.sort(key = lambda x: str(x))
        lambda_residual: Callable[[tuple], usernum] = lambdify(residual_variables, residual)

        def unpack_wrapper(func: Callable[[tuple], usernum]):
            return lambda arr: func(*arr)

        return unpack_wrapper(lambda_residual), residual_variables
        
    def solve(self, subs: dict[Basic, usernum] = {}, guess: usernum = 1) -> dict[Basic, npfloat]:
        lambda_residual, residual_variables = self.get_lambda_residual(subs)

        exception_output = f"""
                ScalarEquation:\n    {self.__str__().replace('\n', '\n    ')}\n
                subs:\n    {subs.__str__().replace('\n', '\n    ')}\n
                residual_variables:\n    {residual_variables.__str__().replace('\n', '\n    ')}"""

        if len(residual_variables) == 0:
            raise ValueError(f"Subs provided for solving reduced scalar equation satisfied all variables\n{exception_output}")

        if len(residual_variables) > 1:
            raise ValueError(f"Insufficient subs to solve scalar equation\n{exception_output}")
        
        try:
            solution = fsolve(lambda_residual, guess)
        
        except Exception as e:
            raise Exception(f"Unable to solve ScalarEquation\n{exception_output}\n{e}")

        return {residual_variables[0] : npfloat(solution[0])}


class MatrixEquation(AlgebraicEquation):
    def _init_expression(self, lhs, rhs) -> None:
        self.lhs: Matrix = sympify(lhs)
        self.rhs: Matrix = sympify(rhs)

    def _init_shape(self) -> None:
        lhs_shape: tuple[int, int] = self.lhs.shape
        rhs_shape: tuple[int, int] = self.rhs.shape
        lhs_rows, lhs_cols = lhs_shape
        rhs_rows, rhs_cols = rhs_shape

        exception_output = f"""
                lhs:\n    {str(self.lhs.__str__()).replace('\n', '\n    ')}\n
                rhs:\n    {str(self.rhs.__str__()).replace('\n', '\n    ')}\n
                lhs.shape: {self.lhs.shape}\n
                rhs.shape: {self.rhs.shape}"""

        if lhs_rows == 0 or lhs_cols == 0 or rhs_rows == 0 or rhs_cols == 0:
            raise ValueError(f"Given expression shapes include a zero width dimension\n{exception_output}")
        
        if self.lhs.shape != self.rhs.shape:
            raise ValueError(f"Given matrices have unequal shapes\n{exception_output}")
        
        self.shape: tuple[int, int] = self.lhs.shape
    
    def _init_size(self):
        self.size: int = int(prod(self.shape))

    def _init_variables(self) -> None:
        self.variables: list[Basic] = list(self.lhs.free_symbols | self.rhs.free_symbols)
        self.variables.sort(key = lambda x: str(x))

    def _init_residual_type(self, residual_type: str):
        if residual_type not in ["differential", "left_inversion", "right_inversion"]:
            raise ValueError(f"Invalid residual_type = '{residual_type}'")
        
        if residual_type in ["left_inversion", "right_inversion"] and self.shape[0] != self.shape[1]:
            raise ValueError(f"Invalid residual_type = '{self.residual_type}' for non-square MatrixEquation of shape = {self.shape}")
                
        self.residual_type = residual_type

    def get_subbed(self, subs: dict[Basic, usernum]) -> "MatrixEquation":
        return MatrixEquation(self.lhs.subs(subs), self.rhs.subs(subs))

    def get_residual(self, subs: dict[Basic, usernum] = {}) -> Matrix:
        subbed_eqn = self.get_subbed(subs)

        if self.residual_type == "differential":
            return sympify(subbed_eqn.lhs) - sympify(subbed_eqn.rhs)
        
        if self.residual_type == "left_inversion":
            return subbed_eqn.lhs**-1 @ subbed_eqn.rhs - Identity(self.shape[0])
        
        if self.residual_type == "right_inversion":
            return subbed_eqn.lhs @ subbed_eqn.rhs**-1 - Identity(self.shape[0])
                 
        else:
            raise ValueError(f"Invalid residual_type = '{self.residual_type}'")

    def get_lambda_residual(self, subs: dict[Basic, usernum] = {}) -> tuple[Callable[[ndarray], ndarray], list[Basic]]:
        residual = self.get_residual(subs)
        residual_variables: list[Basic] = list(residual.free_symbols)
        residual_variables.sort(key = lambda x: str(x))
        shaped_lambda_residual: Callable[[tuple], ndarray] = lambdify(residual_variables, residual)

        def unpack_ravel_wrapper(func: Callable[[tuple], ndarray]) -> Callable[[ndarray], ndarray]:
            return lambda args: func(*args).ravel()

        return unpack_ravel_wrapper(shaped_lambda_residual), residual_variables

    def solve(self, subs: dict[Basic, usernum] = {}, guess_dict: dict[Basic, usernum] = {}) -> dict[Basic, npfloat]:
        lambda_residual, residual_variables = self.get_lambda_residual(subs)

        exception_output = f"""
                MatrixEquation:\n    {self.__str__().replace('\n', '\n    ')}\n
                subs:\n    {subs.__str__().replace('\n', '\n    ')}\n
                residual_variables:\n    {residual_variables.__str__().replace('\n', '\n    ')}"""

        if len(residual_variables) > self.size:
            raise ValueError(f"Insufficient subs to solve matrix equation\n{exception_output}")
        
        guess_vect = [guess_dict.get(variable) if variable in guess_dict.keys() else 1 for variable in residual_variables]
        
        try:
            solution = nparr(fsolve(lambda_residual, guess_vect), dtype=npfloat)

        except Exception as e:
            raise Exception(f"{e}\nUnable to solve MatrixEquation\n{exception_output}")

        return dict(zip(residual_variables, solution))


class TabularEquation(Equation):
    def __init__(self, df: DataFrame, indexing_columns: list[str] = [], preformatted: bool = False, residual_type: str = "proper_column_differential") -> None:
        self._init_expression(df, indexing_columns, preformatted)
        self._init_df_shorthand()
        self._init_size()
        self._init_index_adjacency()
        self._init_residual_type(residual_type)
    
    def _init_expression(self, df: DataFrame, indexing_columns: list[str], preformatted: bool):
        if preformatted:
            self.df = df.sort_index()
        else:
            self.df = df.set_index(indexing_columns).sort_index()

    def _init_df_shorthand(self):
        self.at = self.df.at
        self.columns = self.df.columns
        self.index = self.df.index
        self.iloc = self.df.iloc
        self.loc = self.df.loc

    def _init_size(self):
        self.size = self.index.nlevels + len(self.columns)

    def _init_index_adjacency(self):
        self.index_adjacency = DataFrame(
            [[self._how_is_index_adjacent(I, J)[0] for J in self.index] for I in self.index],
            index=self.index,
            columns=self.index
        )

    def _init_residual_type(self, residual_type: str) -> None:
        if residual_type not in ["proper_column_differential", "all_column_differential"]:
            raise ValueError(f"Invalid residual_type = '{residual_type}'")

        self.residual_type = residual_type

    def _how_is_index_adjacent(self, I: tuple, J: tuple) -> tuple[int, bool]:
        index_inequality = [1 if i != j else 0 for i, j in zip(I, J)]
        num_unequal = sum(index_inequality)
        
        if num_unequal == 1:
            index = index_inequality.index(1)
            return (index, I[index] > J[index])

        else:
            return (-1, False)

    def get_subbed(self, subs: dict[str, usernum]) -> "TabularEquation":
        subbed_df = self.df

        for col, val in zip(subs.keys(), subs.values()):
            is_proper_column = col in subbed_df.columns
            is_index_column = col in subbed_df.index.names

            if not is_proper_column and not is_index_column:
                raise ValueError(f"Given column {col} is not a proper column or an index column in Table\n{self.__str__()}")
            
            if is_proper_column:
                equal_df = subbed_df[subbed_df[col] == val]

                lesser_index_candidates = subbed_df[subbed_df[col] < val].index.values
                greater_index_candidates = subbed_df[subbed_df[col] > val].index.values

            else:
                equal_df = subbed_df[subbed_df.index.get_level_values(col) == val]
                
                lesser_index_candidates = subbed_df[subbed_df.index.get_level_values(col) < val].index.values
                greater_index_candidates = subbed_df[subbed_df.index.get_level_values(col) > val].index.values

            interpolation_candidates_index_adjacency = self.index_adjacency.loc[
                lesser_index_candidates,
                greater_index_candidates
            ].stack(list(range(self.index.nlevels)), future_stack=True)

            interpolated_df = DataFrame([
                self._get_interpolated_row(adj_index, AB[:self.index.nlevels], AB[self.index.nlevels:], col, val, is_proper_column)
                for AB, adj_index in zip(
                    interpolation_candidates_index_adjacency.index.values,
                    interpolation_candidates_index_adjacency.values
                )
                if adj_index != -1
            ])

            if not interpolated_df.empty:
                interpolated_df.index.names = subbed_df.index.names

            subbed_df = concat([equal_df, interpolated_df])

            if subbed_df.empty:
                raise ValueError("Given subs not satisfied in Table")

        return TabularEquation(subbed_df, preformatted=True, residual_type=self.residual_type)

    def _get_interpolated_row(self, adj_index: int, A: tuple, B: tuple, col: str, val: usernum, is_proper_column: bool) -> Series:
        if is_proper_column:
            val_A, val_B = self.at[A, col], self.at[B, col]
        else:
            val_A, val_B = A[adj_index], B[adj_index]

        row_A, row_B = self.loc[A], self.loc[B]
        x = (val - val_A) / (val_B - val_A)

        N = A[:adj_index] + (A[adj_index] * (1 - x) + B[adj_index] * (x),) + A[adj_index + 1:]
        row_N: Series = row_A * (1 - x) + row_B * (x)
        row_N.name = N

        return row_N

    def get_lambda_residual(self, subs: dict[str, usernum] = {}) -> tuple[Callable[[ndarray], ndarray], list[str]]:
        subbed_table = self.get_subbed(subs)

        def lambda_residual(vals: ndarray) -> ndarray:
            N = tuple(vals[:subbed_table.index.nlevels])
            col_vals_N = vals[subbed_table.index.nlevels:]

            if N in subbed_table.index:
                if subbed_table.residual_type == "proper_column_differential":
                    return concatenate([[0] * subbed_table.index.nlevels, col_vals_N - subbed_table.loc[N].values])
                elif subbed_table.residual_type == "all_column_differential":
                    return concatenate([[0] * subbed_table.index.nlevels, col_vals_N - subbed_table.loc[N].values])
            
            novel_index_adjacency = [subbed_table._how_is_index_adjacent(I, N) for I in subbed_table.index]

            lesser_index_candidates = [
                I for I, (adj_index, is_greater) in zip(subbed_table.index, novel_index_adjacency)
                if adj_index != -1 and not is_greater
            ]

            greater_index_candidates = [
                I for I, (adj_index, is_greater) in zip(subbed_table.index, novel_index_adjacency)
                if adj_index != -1 and is_greater
            ]

            interpolation_candidates_index_adjacency = subbed_table.index_adjacency.loc[
                lesser_index_candidates,
                greater_index_candidates
            ].stack(list(range(subbed_table.index.nlevels)), future_stack=True)

            interpolated_col_vals = [
                subbed_table._get_interpolated_col_vals(adj_index, AB[:subbed_table.index.nlevels], N, AB[subbed_table.index.nlevels:])
                for AB, adj_index in zip(
                    interpolation_candidates_index_adjacency.index.values,
                    interpolation_candidates_index_adjacency.values
                )
                if adj_index != -1
            ]

            if len(interpolated_col_vals) != 0:
                if subbed_table.residual_type == "proper_column_differential":
                    return concatenate([[0] * subbed_table.index.nlevels, col_vals_N - interpolated_col_vals[0]])
                elif subbed_table.residual_type == "all_column_differential":
                    return concatenate([[0] * subbed_table.index.nlevels, col_vals_N - interpolated_col_vals[0]])

            index_seperation = [subbed_table._get_index_seperation(N, I) for I in subbed_table.index]
            i_nearest_index = argmin(index_seperation)
            nearest_index: tuple = subbed_table.index.values[i_nearest_index]
            nearest_index_seperation: npfloat = index_seperation[i_nearest_index]
            nearest_index_residual = lambda_residual(nparr(list(nearest_index) + col_vals_N.tolist()))
            unscaled_residual = nearest_index_residual

            if subbed_table.residual_type == "proper_column_differential":
                pass
            elif subbed_table.residual_type == "all_column_differential":
                unscaled_residual[:subbed_table.index.nlevels] = nparr(N) - nparr(nearest_index)

            return unscaled_residual + vectorize(lambda x: -1 if x < 0 else 1)(unscaled_residual) * nearest_index_seperation

        return lambda_residual, [str(name) for name in subbed_table.index.names] + subbed_table.columns.to_list()

    def _get_index_seperation(self, N: tuple, I: tuple) -> npfloat:
        return sum([abs(n - i) for n, i in zip(N, I)])

    def _get_interpolated_col_vals(self, adj_index: int, A: tuple, N: tuple, B: tuple) -> ndarray:
        adj_index_val_A, adj_index_val_N, adj_index_val_B = A[adj_index], N[adj_index], B[adj_index]
        x: float = (adj_index_val_N - adj_index_val_A) / (adj_index_val_B - adj_index_val_A)
        col_vals_A, col_vals_B = self.loc[A].values, self.loc[B].values

        return col_vals_A * (1 - x) + col_vals_B * (x)

    def __getitem__(self, *args, **kwargs):
        return self.df.__getitem__(*args, **kwargs)

    def __str__(self):
        return self.df.__str__()


class System:
    def __init__(self, eqn_list: list[Equation], column_map: dict[Basic, str] = {}) -> None:
        self.eqn_list = eqn_list
        self.column_map = column_map
        self.symbol_map = dict(zip(column_map.values(), column_map.keys()))
        self.size = sum([eqn.size for eqn in self.eqn_list])

        if any([isinstance(eqn, TabularEquation) for eqn in self.eqn_list]) and column_map == {}:
            raise ValueError(f"TabularEquation included in system but no column_map supplied\neqn_list\n{self.eqn_list}\ncolumn_map\n{self.column_map}")
        
    def get_subbed(self, subs: dict[Basic, usernum] = {}) -> "System":
        if self.column_map != {}:
            table_subs = {self.column_map[key]: value for key, value in subs.items() if key in self.column_map.keys()}
            subbed_eqn_list = [eqn.get_subbed(table_subs) if isinstance(eqn, TabularEquation) else eqn.get_subbed(subs) for eqn in self.eqn_list]

        else:
            subbed_eqn_list = [eqn.get_subbed(subs) for eqn in self.eqn_list]

        return System(subbed_eqn_list, self.column_map)

    def get_lambda_residual(self, subs: dict[Basic, usernum]) -> tuple[Callable[[ndarray], ndarray], list[Basic], int]:
        subbed_system = self.get_subbed(subs)

        lambda_residual_list: list[tuple[Callable[[ndarray], usernum | ndarray], list[Basic]]] = [
            (func, fargs if isinstance(fargs[0], Basic) else [subbed_system.symbol_map[farg] for farg in fargs])
            for func, fargs in (eqn.get_lambda_residual() for eqn in subbed_system.eqn_list)
        ]
        func_list = [lambda_residual[0] for lambda_residual in lambda_residual_list]
        farg_list = [lambda_residual[1] for lambda_residual in lambda_residual_list]
        arg_list: list[Basic] = list(set().union(*[fargs for fargs in farg_list]))
        arg_list.sort(key = lambda x: str(x))
        arg_index_map = {arg: i for i, arg in enumerate(arg_list)}
        farg_indices_list = [[arg_index_map[arg] for arg in farg] for farg in farg_list]

        return_len: int = sum([
            shape(func(nparr([1] * len(farg_indices_list))))[0] if get_type_hints(func).get('return') == ndarray\
            else 1 for func, farg_indices_list in zip(func_list, farg_indices_list)
        ])

        def concatenate_wrapper(func_list: list[Callable[[ndarray], usernum | ndarray]]) -> Callable[[ndarray], ndarray]:
            return lambda args: concatenate([
                func(args[farg_indices_list]) if get_type_hints(func).get('return') == ndarray\
                else nparr([func(args[farg_indices_list])]).ravel()\
                for func, farg_indices_list in zip(func_list, farg_indices_list)
            ])

        return concatenate_wrapper(func_list), arg_list, return_len

    def solve(self, subs: dict[Basic, usernum], guess_dict: dict[Basic, usernum] = {}, verbosity: int = 1)-> dict[Basic, npfloat]:
        lambda_residual, residual_variables, return_len = self.get_lambda_residual(subs)
    
        exception_output = f"""
                System:\n    {self.__str__().replace('\n', '    ')}\n
                subs:\n    {subs.__str__().replace('\n', '    ')}\n
                residual_variables:\n    {residual_variables.__str__().replace('\n', '    ')}"""

        if len(residual_variables) > self.size:
            raise ValueError(f"Insufficient subs to solve matrix equation\n{exception_output}")
        
        guess_list = [guess_dict[variable] if variable in guess_dict.keys() else 1 for variable in residual_variables]
        guess_vect = nparr(guess_list + [1] * (return_len - len(guess_list)))
        solution, infodict, ier, msg = fsolve(lambda_residual, guess_vect, full_output=True)

        if verbosity > 0:
            print("fsolve results:")
            print("    " + dict(zip(residual_variables, nparr(solution, dtype=npfloat))).__str__().replace("\n", "    \n"))
            print("    " + msg)

        if verbosity > 1:
            print(infodict)
            print(ier)

        return dict(zip(residual_variables, nparr(solution, dtype=npfloat)))        

    def __str__(self) -> str:
        return "| " + "\n| ".join([eqn.__str__().replace('\n', '\n| ') for eqn in self.eqn_list])

