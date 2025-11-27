import pandas as pd
import numpy as np
import pickle

from vpop_calibration import *
from vpop_calibration.test import *

# Create a dummy training data frame
patients = ["patient-01", "patient-02"]
nb_patients = len(patients)
obsIds = ["obs-01", "obs-02"]
protocol_arms = ["arm-A", "arm-B"]
time_steps = np.arange(0, 10.0, 1.0)
patient_descriptors = ["k1", "k2", "k3"]
gp_params = [*patient_descriptors, "time"]
training_df = pd.DataFrame({"id": patients})
for descriptor in patient_descriptors:
    training_df[descriptor] = np_rng.normal(0, 1, nb_patients)
training_df = training_df.merge(
    pd.DataFrame({"protocol_arm": protocol_arms}), how="cross"
)
training_df = training_df.merge(pd.DataFrame({"time": time_steps}), how="cross")
training_df = training_df.merge(pd.DataFrame({"output_name": obsIds}), how="cross")
training_df["value"] = np_rng.normal(0, 1, training_df.shape[0])
training_df_bootstrapped = training_df.sample(frac=0.5, random_state=np_rng)

implemented_kernels = ["RBF", "SMK", "Matern"]
implemented_var_strat = ["IMV", "LMCV"]
implemented_mll = ["ELBO", "PLL"]
deep_or_not = [True, False]

model_file = "vpop_calibration/test/gp_model_for_tests.pkl"


def gp_init_flavor(var_strat, kernel, deep_kernel, mll):
    gp = GP(
        training_df,
        gp_params,
        var_strat=var_strat,
        mll=mll,
        kernel=kernel,
        deep_kernel=deep_kernel,
        nb_latents=2,
        nb_features=5,
        num_mixtures=3,
        nb_training_iter=2,
    )
    gp.train()
    with open(model_file, "wb") as file:
        pickle.dump(gp, file)


def test_all_gp_flavors():
    for deep_kernel in deep_or_not:
        for kernel in implemented_kernels:
            for var_strat in implemented_var_strat:
                for mll in implemented_mll:
                    gp_init_flavor(var_strat, kernel, deep_kernel, mll)


def test_batching_1():
    gp = GP(training_df, gp_params, nb_training_iter=2)
    gp.train(mini_batching=True, mini_batch_size=8)


def test_batching_2():
    gp = GP(training_df, gp_params, nb_training_iter=2)
    gp.train(mini_batching=True, mini_batch_size=None)


def test_eval_with_valid():
    gp = GP(training_df, gp_params, nb_training_iter=2)
    gp.eval_perf()


def test_eval_no_valid():
    gp = GP(training_df, gp_params, nb_training_iter=2, training_proportion=1)
    gp.eval_perf()


def test_gp_incomplete_data():
    gp = GP(training_df_bootstrapped, gp_params, nb_training_iter=2)
    gp.train()
    gp.train(mini_batching=True, mini_batch_size=8)
    gp.eval_perf()
