import numpy as np
from scipy.optimize import fsolve

def coex_chemical_potential(
    epsilon: float, df: float, dmu: float, chem_rec_baserate: float, DRIVEN: bool = False,
    scheme: int = 1,
    verbose: bool = False,
) -> float:
    if DRIVEN:

        if verbose:
            print("🏎️ Using numerical solution for driven case.")

        if scheme == 1:
            k = chem_rec_baserate

        if scheme == 2:
            # Positive Drive

            choices = [1.0, np.exp(-df-dmu+2*epsilon)]
            k = np.min(choices) * chem_rec_baserate

            if verbose:
                print("➡️ Using Positive Drive Scheme k ~ min(1,exp(-Δf - Δμ + 2*ε)) \n 🏳️ k =", k)

        if scheme == 3:
            # Negative Drive

            choices = [1.0, np.exp(-df-dmu-2*epsilon)]
            k = np.min(choices) * chem_rec_baserate

            if verbose:
                print("➡️ Using Negative Drive Scheme k ~ min(1,exp(-Δf - Δμ - 2*ε)) \n 🏳️ k =", k)
        initial_guess = 2*epsilon
        j_solution = fsolve(coex_sflex, initial_guess, args=(k, df, epsilon, dmu))
        if verbose:
            print("🏁 Found solution for μ:", j_solution[0])
        return j_solution[0]
    else:
        epsilon = abs(epsilon)
        # This is the exact analytical equilibrium solution
        return -np.log(np.exp(2 * epsilon) - np.exp(df))


def sflex(j, k, f, epsilon, dmu):

    e = np.exp(j) + k * np.exp(j) + k * np.exp(f + j)
    d = (
        np.exp(2 * epsilon)
        + k * np.exp(2 * epsilon)
        + np.exp(2 * epsilon + f + j)
        + k * np.exp(2 * epsilon + f + dmu)
        + k * np.exp(2 * epsilon + f + dmu + j)
        + k * np.exp(2 * epsilon + 2 * f + dmu + j)
    )
    return e / d


def coex_sflex(j, k, f, epsilon, dmu):
    return sflex(j, k, f, epsilon, dmu) - 1