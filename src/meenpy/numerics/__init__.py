from numpy import abs, apply_along_axis, argmin, concatenate, linspace, max, meshgrid, newaxis, ndarray, array as nparray, float64 as npfloat, prod, reciprocal, shape, sum, vectorize, where
from numpy.linalg import norm
from pandas import concat, DataFrame, Series
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy.optimize import fsolve
from sympy import Basic, Expr, Identity, lambdify, Matrix, sympify
from typing import Callable, cast, get_type_hints


pio.renderers.default = "browser"
usernum = int | float | npfloat


class Solvable:
    """Solvable provides the structure for user facing system and equation classes and should not be used directly
    """
    def get_lambda_residual(self, subs: dict = {}) -> tuple:
        return tuple()
    
    def visualize_lambda_residual(self, domain_definition: dict[Basic | str, tuple[usernum, usernum, int]], subs: dict[Basic | str, usernum] = {}, html_path: str = "", show: bool = False) -> None:
        """visualize_lambda_residual provides a visualization of the magnitude of the residual function of a Solvable object across a given domain for the putpose of debugging solver failures

        Args:
            domain_definition (dict[Basic  |  str, tuple[usernum, usernum, int]]): Dictionary of domain definitions. Basic | str objects must specify a residual variable in the Solvable object. Tuples must specify a start, stop, and number of values included in the subdomain of the variable, being passed to numpy's linspace
            subs (dict[Basic  |  str, usernum], optional): substitution dictionary. Defaults to {}.
            html_path (str, optional): Path to save *.html file of visualization. Not saved if show=True. Defaults to "".
            show (bool, optional): Whether or not to display visualization in your the default browser at runtime. Note that this gives warnings because of plotly. Defaults to False.

        Raises:
            ValueError: Residual variable exists in Solvable object that is not provided in domain_definition
            ValueError: Uneccesary or duplicate domain_defintion inclusions leading to more domain variables that residual variables
        """
        
        lambda_residual, residual_variables, *_ = self.get_lambda_residual(subs)

        domain_variables = domain_definition.keys()

        if any([variable not in domain_variables for variable in residual_variables]):
            raise ValueError(f"Insufficient substitutions to be visualize AlgebraicEquation by variation in\n{len(domain_variables)} provided domain variables: {domain_variables}\nwith provided subs: {subs}\nleaving {len(residual_variables)} residual_variables: {residual_variables}")

        if len(domain_variables) != len(residual_variables):
            raise ValueError(f"Mismatch between\n{len(domain_variables)} provided domain variables: {domain_variables}\nwith provided subs: {subs}\nleaving {len(residual_variables)} residual_variables: {residual_variables}")

        if len(domain_variables) == 1:
            residual_variable = residual_variables.pop()
            domain = linspace(*domain_definition[residual_variable])
            residual_magnitudes = apply_along_axis(lambda state: norm(lambda_residual(state)), 1, domain[:, newaxis])

            df = DataFrame({residual_variable.__str__(): domain, "residual magnitude": residual_magnitudes})
            fig = px.line(df, template="plotly_dark")

        elif len(domain_variables) == 2:
            subdomains = [linspace(*subdomain_definition) for subdomain_definition in domain_definition.values()]
            value_frames = meshgrid(*subdomains)
            value_permutations = nparray([frame.ravel() for frame in value_frames]).transpose()
            variable_index_mapping = {
                residual_variables.index(domain_variable): domain_variable_index
                for domain_variable_index, domain_variable in enumerate(domain_definition.keys())
            }
            variable_indexing = nparray([variable_index_mapping[i] for i in range(len(domain_variables))])
            residual_magnitudes = apply_along_axis(
                lambda values: norm(lambda_residual(values[variable_indexing])),
                1, value_permutations
            ).reshape(value_frames[0].shape)

            fig = px.imshow(residual_magnitudes, x=subdomains[0], y=subdomains[1], origin='lower', labels={'x': residual_variables[0].__str__(), 'y': residual_variables[1].__str__(), 'color': 'residual magnitude'}, color_continuous_scale='Viridis', template='plotly_dark')

        else:
            subdomains = [linspace(*subdomain_definition) for subdomain_definition in domain_definition.values()]
            value_frames = meshgrid(*subdomains)
            value_permutations = nparray([frame.ravel() for frame in value_frames]).transpose()
            variable_index_mapping = {
                residual_variables.index(domain_variable): domain_variable_index
                for domain_variable_index, domain_variable in enumerate(domain_definition.keys())
            }
            variable_indexing = nparray([variable_index_mapping[i] for i in range(len(domain_variables))])

            residual_magnitudes = apply_along_axis(
                lambda values: norm(lambda_residual(values[variable_indexing])),
                1, value_permutations
            )
            i_min = argmin(residual_magnitudes)
            min_residual_magnitude = residual_magnitudes[i_min]
            min_value_permutation = value_permutations[i_min]
            max_residual_magnitude = max(residual_magnitudes)
            x_len, y_len = len(subdomains[0]), len(subdomains[1])
            z_len = int(len(residual_magnitudes) / (x_len * y_len))
            shaped_residual_magnitudes = residual_magnitudes.reshape((z_len, y_len, x_len)).transpose()

            fig = go.Figure(data=[go.Heatmap(z=shaped_residual_magnitudes[:, :, 0], x=subdomains[0], y=subdomains[1], colorscale="Viridis", zmin=float(min_residual_magnitude), zmax=float(max_residual_magnitude))])
            domain_variable_strings = [domain_variable.__str__() for domain_variable in domain_definition.keys()]

            steps = [{
                "method": "restyle",
                "label": ", ".join([f"{value:.{2}e}" for value in value_permutations[i_z, 2:]]),
                "args": [{"z": [shaped_residual_magnitudes[:, :, i_z]]}, [0]]
            } for i_z in range(z_len)]
            sliders = [{"active": 0, "currentvalue": {"prefix": ", ".join(domain_variable_strings[2:]) + " = "}, "steps": steps}]

            fig.update_layout(title=f"Residual magnitude minimized to {min_residual_magnitude} at {", ".join(domain_variable_strings) + " = "} = {min_value_permutation}", sliders=sliders, xaxis_title=domain_variable_strings[0], yaxis_title=domain_variable_strings[1])

        if show:
            fig.show()
        else:
            if html_path != "" and html_path[-5:] != ".html":
                html_path = html_path + ".html"
            
            fig.write_html(html_path, auto_open=False)
    
    def solve(self, subs: dict = {}) -> dict:
        return {}


class Equation(Solvable):
    """Equation provides the structure for user facing equation classes and should not be used directly
    """

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
    """AlgebraicEquation  provides the structure for user facing algebraic equation classes and should not be used directly

    Args:
        Equation (class): AlgebricEquation inherits from this
    """

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
    """ScalarEquation enables articulation and solution of scalar valued equations

    Args:
        AlgebraicEquation (class): ScalarEquation inherits from this
    """

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
        """get_subbed returns a copy of self with substitutions applied

        Args:
            subs (dict[Basic, usernum]): subsitution dictionary

        Returns:
            ScalarEquation: copy of self with substitutions applied
        """

        return ScalarEquation(self.lhs.subs(subs), self.rhs.subs(subs))

    def get_residual(self, subs: dict[Basic, usernum] = {}) -> Expr:
        """get_residual returns a residual derived from self with subtitutions applied

        Args:
            subs (dict[Basic, usernum], optional): subsitution dictionary. Defaults to {}.

        Raises:
            ValueError: raised on invalid residual type

        Returns:
            Expr: residual derived from self with subtitutions applied
        """

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
        """get_lambda_residual returns a residual function and list of function arguments after apply a substitution

        Args:
            subs (dict[Basic, usernum], optional): substitution dictionary. Defaults to {}.

        Returns:
            tuple[Callable[[ndarray], usernum], list[Basic]]: residual function and its arguments
        """

        residual = self.get_residual(subs)
        residual_variables = list(residual.free_symbols)
        residual_variables.sort(key = lambda x: str(x))
        lambda_residual: Callable[[tuple], usernum] = lambdify(residual_variables, residual)

        def unpack_wrapper(func: Callable[[tuple], usernum]):
            return lambda arr: func(*arr)

        return unpack_wrapper(lambda_residual), residual_variables
        
    def solve(self, subs: dict[Basic, usernum] = {}, guess: usernum = 1, verbosity: int = 1, maxfev: int = 0, xtol: usernum = 1.49012e-8, epsfcn: usernum | None = None) -> dict[Basic, npfloat]:
        """solve returns one solution after applying a substitution, utilizing a guess, and passing optional arguments to scipy's fsolve

        Args:
            subs (dict[Basic, usernum], optional): substitution dictionary. Defaults to {}.
            guess (usernum, optional): guess for fsolve. Defaults to 1.
            verbosity (int, optional): amount of information regarding the solution printed to console according to `{0: 'None', 1: 'Result and message', 2: 'All'}`. Defaults to 1.
            maxfev (int, optional): The maximum number of calls to the function. If zero, then `100*(N+1)` is the maximum where `N` is the number of elements in `guess`. Defaults to 0.
            xtol (usernum, optional): The calculation will terminate if the relative error between two consecutive iterates is at most xtol. Defaults to 1.49012e-8.
            epsfcn (usernum | None, optional): A suitable step length for the forward-difference approximation of the Jacobian (for `fprime=None`). If epsfcn is less than the machine precision, it is assumed that the relative errors in the functions are of the order of the machine precision. Defaults to None.

        Raises:
            ValueError: Substitutions satisfy all variables
            ValueError: Insufficient substitutions to possibly solve
            Exception: fsolve exception

        Returns:
            dict[Basic, npfloat]: solution dictionary
        """
        
        lambda_residual, residual_variables = self.get_lambda_residual(subs)

        exception_output = f"\
                ScalarEquation:\n    {self.__str__().replace('\n', '\n    ')}\n\
                subs:\n    {subs.__str__().replace('\n', '\n    ')}\n\
                residual_variables:\n    {residual_variables.__str__().replace('\n', '\n    ')}"

        if len(residual_variables) == 0:
            raise ValueError(f"Subs provided for solving reduced ScalarEquation satisfied all variables\n{exception_output}")

        if len(residual_variables) > 1:
            raise ValueError(f"Insufficient subs to solve ScalarEquation\n{exception_output}")
        
        try:
            solution, infodict, ier, msg = fsolve(lambda_residual, guess, full_output=True, maxfev=maxfev, xtol=xtol, epsfcn=epsfcn)

            if verbosity > 0:
                print("fsolve results:")
                print("    " + dict(zip(residual_variables, nparray(solution, dtype=npfloat))).__str__().replace("\n", "    \n"))
                print("    " + msg)

            if verbosity > 1:
                print(infodict)
                print(ier)
        
        except Exception as e:
            raise Exception(f"Unable to solve ScalarEquation\n{exception_output}\n{e}")

        return {residual_variables[0] : npfloat(solution[0])}


class MatrixEquation(AlgebraicEquation):
    """MatrixEquation enables articulation and solution of matrix (including vector) valued equations

    Args:
        AlgebraicEquation (class): MatrixEquation inherits from this
    """

    def _init_expression(self, lhs, rhs) -> None:
        if isinstance(lhs, list) and isinstance(rhs, list):
            self.lhs = Matrix(lhs)
            self.rhs = Matrix(rhs)
        else:
            self.lhs: Matrix = sympify(lhs)
            self.rhs: Matrix = sympify(rhs)

    def _init_shape(self) -> None:
        lhs_shape: tuple[int, int] = self.lhs.shape
        rhs_shape: tuple[int, int] = self.rhs.shape
        lhs_rows, lhs_cols = lhs_shape
        rhs_rows, rhs_cols = rhs_shape

        exception_output = f"\
                lhs:\n    {str(self.lhs.__str__()).replace('\n', '\n    ')}\n\
                rhs:\n    {str(self.rhs.__str__()).replace('\n', '\n    ')}\n\
                lhs.shape: {self.lhs.shape}\n\
                rhs.shape: {self.rhs.shape}"

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
        """get_subbed returns a copy of self with substitutions applied

        Args:
            subs (dict[Basic, usernum]): subsitution dictionary

        Returns:
            MatrixEquation: copy of self with substitutions applied
        """

        return MatrixEquation(self.lhs.subs(subs), self.rhs.subs(subs))

    def get_residual(self, subs: dict[Basic, usernum] = {}) -> Matrix:
        """get_residual returns a residual derived from self with subtitutions applied

        Args:
            subs (dict[Basic, usernum], optional): subsitution dictionary. Defaults to {}.

        Raises:
            ValueError: raised on invalid residual type

        Returns:
            Matrix: residual derived from self with subtitutions applied
        """

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
        """get_lambda_residual returns a residual function and list of function arguments after apply a substitution

        Args:
            subs (dict[Basic, usernum], optional): substitution dictionary. Defaults to {}.

        Returns:
            tuple[Callable[[ndarray], ndarray], list[Basic]]: residual function and its arguments
        """

        residual = self.get_residual(subs)
        residual_variables: list[Basic] = list(residual.free_symbols)
        residual_variables.sort(key = lambda x: str(x))
        shaped_lambda_residual: Callable[[tuple], ndarray] = lambdify(residual_variables, residual)

        def unpack_ravel_wrapper(func: Callable[[tuple], ndarray]) -> Callable[[ndarray], ndarray]:
            return lambda args: func(*args).ravel()

        return unpack_ravel_wrapper(shaped_lambda_residual), residual_variables

    def solve(self, subs: dict[Basic, usernum] = {}, guess_dict: dict[Basic, usernum] = {}, verbosity: int = 1, maxfev: int = 0, xtol: usernum = 1.49012e-8, epsfcn: usernum | None = None) -> dict[Basic, npfloat]:
        """solve returns one solution after applying a substitution, utilizing a guess, and passing optional arguments to scipy's fsolve

        Args:
            subs (dict[Basic, usernum], optional): substitution dictionary. Defaults to {}.
            guess_dict (dict[Basic, usernum], optional): guess dictionary, the values of which are passed to fsolve. Defaults to {}.
            verbosity (int, optional): amount of information regarding the solution printed to console according to `{0: 'None', 1: 'Result and message', 2: 'All'}`. Defaults to 1.
            maxfev (int, optional): The maximum number of calls to the function. If zero, then `100*(N+1)` is the maximum where `N` is the number of elements in `guess`. Defaults to 0.
            xtol (usernum, optional): The calculation will terminate if the relative error between two consecutive iterates is at most xtol. Defaults to 1.49012e-8.
            epsfcn (usernum | None, optional): A suitable step length for the forward-difference approximation of the Jacobian (for `fprime=None`). If epsfcn is less than the machine precision, it is assumed that the relative errors in the functions are of the order of the machine precision. Defaults to None.

        Raises:
            ValueError: Insufficient substitutions to possibly solve
            Exception: fsolve exception

        Returns:
            dict[Basic, npfloat]: solution dictionary
        """
        
        lambda_residual, residual_variables = self.get_lambda_residual(subs)

        exception_output = f"\
                MatrixEquation:\n    {self.__str__().replace('\n', '\n    ')}\n\
                subs:\n    {subs.__str__().replace('\n', '\n    ')}\n\
                residual_variables:\n    {residual_variables.__str__().replace('\n', '\n    ')}"

        if len(residual_variables) > self.size:
            raise ValueError(f"Insufficient subs to solve MatrixEquation\n{exception_output}")
        
        guess_vect = [guess_dict.get(variable) if variable in guess_dict.keys() else 1 for variable in residual_variables]
        
        try:
            solution, infodict, ier, msg = fsolve(lambda_residual, guess_vect, full_output=True, maxfev=maxfev, xtol=xtol, epsfcn=epsfcn)

            if verbosity > 0:
                print("fsolve results:")
                print("    " + dict(zip(residual_variables, nparray(solution, dtype=npfloat))).__str__().replace("\n", "    \n"))
                print("    " + msg)

            if verbosity > 1:
                print(infodict)
                print(ier)
        
        except Exception as e:
            raise Exception(f"Unable to solve MatrixEquation\n{exception_output}\n{e}")

        return dict(zip(residual_variables, solution))


class TabularEquation(Equation):
    """TabularEquation enables articulation and solution of equations defined by tables

    Args:
        Equation (class): TabularEquation inherits from this
    """

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
        self.residual_scaling: ndarray = reciprocal(max(abs(nparray(
            [vectorize(lambda I: I[i])(self.index.values) for i in range(self.index.nlevels)] + [series.values for col, series in self.df.items()] # Can't you just cast I to a list?
        )), axis=1))

    def _how_is_index_adjacent(self, I: tuple, J: tuple) -> tuple[int, bool]:
        index_inequality = [1 if i != j else 0 for i, j in zip(I, J)]
        num_unequal = sum(index_inequality)
        
        if num_unequal == 1:
            index = index_inequality.index(1)
            return (index, I[index] > J[index])

        else:
            return (-1, False)

    def get_subbed(self, subs: dict[str, usernum]) -> "TabularEquation":
        """get_subbed returns a copy of self with substitutions applied

        Args:
            subs (dict[str, usernum]): subsitution dictionary

        Returns:
            TabularEquation: copy of self with substitutions applied
        """

        subbed_df = self.df

        for col, val in zip(subs.keys(), subs.values()):
            is_proper_column = col in subbed_df.columns
            is_index_column = col in subbed_df.index.names

            if not is_proper_column and not is_index_column:
                raise ValueError(f"Given column {col} is not a proper column or an index column in TablularEquation\n{self.__str__()}")
            
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
                raise ValueError("Given subs not possibly satisfied by TabularEquation")

        return TabularEquation(subbed_df, preformatted=True, residual_type=self.residual_type)

    def _get_interpolated_row(self, adj_index: int, A: tuple, B: tuple, col: str, val: usernum, is_proper_column: bool) -> Series:
        if is_proper_column:
            val_A, val_B = cast(float, self.at[A, col]), cast(float, self.at[B, col])
        else:
            val_A, val_B = cast(float, A[adj_index]), cast(float, B[adj_index])

        row_A, row_B = cast(Series, self.loc[A]), cast(Series, self.loc[B])
        x = (val - val_A) / (val_B - val_A)
        
        N = A[:adj_index] + (A[adj_index] * (1 - x) + B[adj_index] * (x),) + A[adj_index + 1:]
        row_N: Series = row_A * (1 - x) + row_B * (x)
        row_N.name = N

        return row_N

    def get_lambda_residual(self, subs: dict[str, usernum] = {}) -> tuple[Callable[[ndarray], ndarray], list[str]]:
        """get_lambda_residual returns a residual function and list of function arguments after apply a substitution

        Args:
            subs (dict[str, usernum], optional): substitution dictionary. Defaults to {}.

        Returns:
            tuple[Callable[[ndarray], ndarray], list[str]]: residual function and its arguments
        """

        subbed_table = self.get_subbed(subs)

        def lambda_residual(vals: ndarray) -> ndarray:
            N = tuple(vals[:subbed_table.index.nlevels])
            col_vals_N = vals[subbed_table.index.nlevels:]

            if N in subbed_table.index:
                if subbed_table.residual_type == "proper_column_differential":
                    return concatenate([[0] * subbed_table.index.nlevels, col_vals_N - subbed_table.loc[N].values]) * subbed_table.residual_scaling
                elif subbed_table.residual_type == "all_column_differential":
                    return concatenate([[0] * subbed_table.index.nlevels, col_vals_N - subbed_table.loc[N].values]) * subbed_table.residual_scaling
            
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
                    return concatenate([[0] * subbed_table.index.nlevels, col_vals_N - interpolated_col_vals[0]]) * subbed_table.residual_scaling
                elif subbed_table.residual_type == "all_column_differential":
                    return concatenate([[0] * subbed_table.index.nlevels, col_vals_N - interpolated_col_vals[0]]) * subbed_table.residual_scaling

            index_seperation = [subbed_table._get_index_seperation(N, I) for I in subbed_table.index]
            i_nearest_index = argmin(index_seperation)
            nearest_index: tuple = subbed_table.index.values[i_nearest_index]
            nearest_index_seperation: npfloat = index_seperation[i_nearest_index]
            nearest_index_residual = lambda_residual(nparray(list(nearest_index) + col_vals_N.tolist()))
            unscaled_residual = nearest_index_residual

            if subbed_table.residual_type == "proper_column_differential":
                pass
            elif subbed_table.residual_type == "all_column_differential":
                unscaled_residual[0:subbed_table.index.nlevels] = (nparray(N) - nparray(nearest_index)) * subbed_table.residual_scaling[0:subbed_table.index.nlevels]

            unscaled_residual_sign_vector = where(unscaled_residual < 0, -1.0, 1.0)
            return unscaled_residual + unscaled_residual_sign_vector * nearest_index_seperation

        return lambda_residual, [str(name) for name in subbed_table.index.names] + subbed_table.columns.to_list()

    def _get_index_seperation(self, N: tuple, I: tuple) -> npfloat:
        return npfloat(norm((nparray(N) - nparray(I)) * self.residual_scaling[0:self.index.nlevels]))

    def _get_interpolated_col_vals(self, adj_index: int, A: tuple, N: tuple, B: tuple) -> ndarray:
        adj_index_val_A, adj_index_val_N, adj_index_val_B = A[adj_index], N[adj_index], B[adj_index]
        x: float = (adj_index_val_N - adj_index_val_A) / (adj_index_val_B - adj_index_val_A)
        col_vals_A, col_vals_B = self.loc[A].values, self.loc[B].values

        return col_vals_A * (1 - x) + col_vals_B * (x)
    
    def solve(self, subs: dict[str, usernum] = {}, guess_dict: dict[str, usernum] = {}, verbosity: int = 1, maxfev: int = 0, xtol: usernum = 1.49012e-8, epsfcn: usernum | None = None) -> dict[str, npfloat]:
        """solve returns one solution after applying a substitution, utilizing a guess, and passing optional arguments to scipy's fsolve

        Args:
            subs (dict[str, usernum], optional): substitution dictionary. Defaults to {}.
            guess_dict (dict[str, usernum], optional): guess dictionary, the values of which are passed to fsolve. Defaults to {}.
            verbosity (int, optional): amount of information regarding the solution printed to console according to `{0: 'None', 1: 'Result and message', 2: 'All'}`. Defaults to 1.
            maxfev (int, optional): The maximum number of calls to the function. If zero, then `100*(N+1)` is the maximum where `N` is the number of elements in `guess`. Defaults to 0.
            xtol (usernum, optional): The calculation will terminate if the relative error between two consecutive iterates is at most xtol. Defaults to 1.49012e-8.
            epsfcn (usernum | None, optional): A suitable step length for the forward-difference approximation of the Jacobian (for `fprime=None`). If epsfcn is less than the machine precision, it is assumed that the relative errors in the functions are of the order of the machine precision. Defaults to None.

        Raises:
            ValueError: Insufficient substitutions to possibly solve
            Exception: fsolve exception

        Returns:
            dict[str, npfloat]: solution dictionary
        """
        
        lambda_residual, residual_variables = self.get_lambda_residual(subs)

        exception_output = f"\
                TabularEquation:\n    {self.__str__().replace('\n', '\n    ')}\n\
                subs:\n    {subs.__str__().replace('\n', '\n    ')}\n\
                residual_variables:\n    {residual_variables.__str__().replace('\n', '\n    ')}"
        
        if len(residual_variables) > self.size:
            raise ValueError(f"Insufficient subs to solve TabularEquation\n{exception_output}")
        
        try:
            guess_vect = nparray([guess_dict[variable] if variable in guess_dict.keys() else 1 for variable in residual_variables])
            solution, infodict, ier, msg = fsolve(lambda_residual, guess_vect, full_output=True, maxfev=maxfev, xtol=xtol, epsfcn=epsfcn)

            if verbosity > 0:
                print("fsolve results:")
                print("    " + dict(zip(residual_variables, nparray(solution, dtype=npfloat))).__str__().replace("\n", "    \n"))
                print("    " + msg)

            if verbosity > 1:
                print(infodict)
                print(ier)
        
        except Exception as e:
            raise Exception(f"Unable to solve TabularEquation\n{exception_output}\n{e}")

        return dict(zip(residual_variables, nparray(solution, dtype=npfloat)))  

    def __getitem__(self, *args, **kwargs):
        return self.df.__getitem__(*args, **kwargs)

    def __str__(self):
        return self.df.__str__()


class System(Solvable):
    """System enables the composition of Equation objects into a system solution of that system
    """

    def __init__(self, eqn_list: list[Equation], column_map: dict[Basic, str] = {}) -> None:
        self.eqn_list = eqn_list
        self.column_map = column_map
        self.symbol_map = dict(zip(column_map.values(), column_map.keys()))
        self.size = sum([eqn.size for eqn in self.eqn_list])

        if any([isinstance(eqn, TabularEquation) for eqn in self.eqn_list]) and column_map == {}:
            raise ValueError(f"TabularEquation included in system but no column_map supplied\neqn_list\n{self.eqn_list}\ncolumn_map\n{self.column_map}")
        
    def get_subbed(self, subs: dict[Basic, usernum] = {}) -> "System":
        """get_subbed returns a copy of self with substitutions applied

        Args:
            subs (dict[Basic, usernum]): subsitution dictionary

        Returns:
            System: copy of self with substitutions applied
        """

        if self.column_map != {}:
            table_subs = {self.column_map[key]: value for key, value in subs.items() if key in self.column_map.keys()}
            subbed_eqn_list = [eqn.get_subbed(table_subs) if isinstance(eqn, TabularEquation) else eqn.get_subbed(subs) for eqn in self.eqn_list]

        else:
            subbed_eqn_list = [eqn.get_subbed(subs) for eqn in self.eqn_list]

        return System(subbed_eqn_list, self.column_map)

    def get_lambda_residual(self, subs: dict[Basic, usernum] = {}) -> tuple[Callable[[ndarray], ndarray], list[Basic], int]:
        """get_lambda_residual returns a residual function and list of function arguments after apply a substitution

        Args:
            subs (dict[Basic, usernum], optional): substitution dictionary. Defaults to {}.

        Returns:
            tuple[Callable[[ndarray], ndarray], list[Basic]]: residual function and its arguments
        """

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
            shape(func(nparray([1] * len(farg_indices_list))))[0] if get_type_hints(func).get('return') == ndarray\
            else 1 for func, farg_indices_list in zip(func_list, farg_indices_list)
        ])

        def concatenate_wrapper(func_list: list[Callable[[ndarray], usernum | ndarray]]) -> Callable[[ndarray], ndarray]:
            return lambda args: concatenate([
                func(args[farg_indices_list]) if get_type_hints(func).get('return') == ndarray\
                else nparray([func(args[farg_indices_list])]).ravel()\
                for func, farg_indices_list in zip(func_list, farg_indices_list)
            ])

        return concatenate_wrapper(func_list), arg_list, return_len

    def solve(self, subs: dict[Basic, usernum] = {}, guess_dict: dict[Basic, usernum] = {}, verbosity: int = 1, maxfev: int = 0, xtol: usernum = 1.49012e-8, epsfcn: usernum | None = None)-> dict[Basic, npfloat]:
        """solve returns one solution after applying a substitution, utilizing a guess, and passing optional arguments to scipy's fsolve

        Args:
            subs (dict[Basic, usernum]): subtitution dictionary
            guess_dict (dict[Basic, usernum], optional): guess dictionary, the values of which are passed to fsolve. Defaults to {}.
            verbosity (int, optional): amount of information regarding the solution printed to console according to `{0: 'None', 1: 'Result and message', 2: 'All'}`. Defaults to 1.
            maxfev (int, optional): The maximum number of calls to the function. If zero, then `100*(N+1)` is the maximum where `N` is the number of elements in `guess`. Defaults to 0.
            xtol (usernum, optional): The calculation will terminate if the relative error between two consecutive iterates is at most xtol. Defaults to 1.49012e-8.
            epsfcn (usernum | None, optional): A suitable step length for the forward-difference approximation of the Jacobian (for `fprime=None`). If epsfcn is less than the machine precision, it is assumed that the relative errors in the functions are of the order of the machine precision. Defaults to None.

        Raises:
            ValueError: Insufficient substitutions to possibly solve
            Exception: fsolve exception

        Returns:
            dict[Basic, npfloat]: solution dictionary
        """
        
        lambda_residual, residual_variables, return_len = self.get_lambda_residual(subs)
    
        exception_output = f"\
                System:\n    {self.__str__().replace('\n', '\n    ')}\n\
                subs:\n    {subs.__str__().replace('\n', '\n    ')}\n\
                residual_variables:\n    {residual_variables.__str__().replace('\n', '\n    ')}"

        if len(residual_variables) > self.size:
            raise ValueError(f"Insufficient subs to solve System\n{exception_output}")
        
        guess_list = [guess_dict[variable] if variable in guess_dict.keys() else 1 for variable in residual_variables]
        guess_vect = nparray(guess_list + [1] * (return_len - len(guess_list)))

        try:
            solution, infodict, ier, msg = fsolve(lambda_residual, guess_vect, full_output=True, maxfev=maxfev, xtol=xtol, epsfcn=epsfcn)

            if verbosity > 0:
                print("fsolve results:")
                print("    " + dict(zip(residual_variables, nparray(solution, dtype=npfloat))).__str__().replace("\n", "    \n"))
                print("    " + msg)

            if verbosity > 1:
                print(infodict)
                print(ier)
        
        except Exception as e:
            raise Exception(f"Unable to solve System\n{exception_output}\n{e}")

        return dict(zip(residual_variables, nparray(solution, dtype=npfloat)))        

    def __str__(self) -> str:
        return "| " + "\n| ".join([eqn.__str__().replace('\n', '\n| ') for eqn in self.eqn_list])

