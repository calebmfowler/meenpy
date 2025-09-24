import unittest
from meenpy.numerics import ScalarEquation as seqn, MatrixEquation as meqn, System as sys, TabularEquation as teqn
from meenpy.numerics.utils import *

residual_type = "all_column_differential"
water = teqn(read_csv("test/water.csv"), ["Temperature", "Quality"], residual_type=residual_type)
func, fargs = water.get_lambda_residual()
state_space_ranges = {
    "Temperature": (0, 200, 21),
    "Quality": (-0.5, 1.5, 21)
}
property_ranges = [np.linspace(*state_space_ranges[arg]) for arg in fargs if arg in state_space_ranges.keys()]
state_space = np.meshgrid(*property_ranges)

residual_magnitudes = np.vectorize(lambda *state: np.linalg.norm(func(np.array([*state] + [0.5, 0.751]))))(*state_space).reshape(state_space[0].shape)

def plot_residuals(state_space, residuals, fargs, outpath=None):
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(*state_space, residuals, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax, label="Residual magnitude")
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath)

if __name__ == "__main__":
    plot_residuals(state_space, residual_magnitudes, fargs, outpath=f"test/residuals_table_{residual_type}.png")

