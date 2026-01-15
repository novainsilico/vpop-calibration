import pickle
import pytest
import os
import pandas as pd
import numpy as np
import itertools

from vpop_calibration import *


@pytest.fixture(scope="session")
def training_data(np_rng):
    # Use this fixture for testing GP training
    patients = {"id": ["p1", "p2"], "k1": [1.0, 2.0]}
    protocol_arms = ["arm-A", "arm-B"]
    outputs = ["s1", "s2"]
    time_steps = np.arange(0, 3.0, 1.0)
    df = pd.DataFrame.from_dict(patients)
    df = df.merge(pd.DataFrame(protocol_arms, columns=["protocol_arm"]), how="cross")
    df = df.merge(pd.DataFrame(outputs, columns=["output_name"]), how="cross")
    df = df.merge(pd.DataFrame(time_steps, columns=["time"]), how="cross")
    df["value"] = np_rng.normal()
    params = ["k1", "time"]
    return df, params


@pytest.fixture(scope="session")
def training_data_bootstrapped(training_data):
    # Same as training data, but incomplete data set
    df, params = training_data
    return (df.sample(frac=0.5), params)


def test_gp_init(training_data, subtests):
    df, params = training_data
    kernel_test = ["RBF", "SMK", "Matern"]
    var_strat_test = ["IMV", "LMCV"]
    mll_test = ["ELBO", "PLL"]
    deep_kernel_test = [True, False]
    training_proportion_test = [1.0, 0.5]
    test_params = itertools.product(
        kernel_test,
        var_strat_test,
        mll_test,
        deep_kernel_test,
        training_proportion_test,
    )
    for i, params_subtest in enumerate(test_params):
        (
            kernel,
            var_strat,
            mll,
            deep_kernel,
            training_proportion,
        ) = params_subtest
        with subtests.test(i=i, msg=f"Testing GP init with {params_subtest}"):
            gp = GP(
                df,
                params,
                var_strat=var_strat,
                mll=mll,
                kernel=kernel,
                deep_kernel=deep_kernel,
                training_proportion=training_proportion,
            )


@pytest.mark.parametrize("mini_batching", [True, False])
@pytest.mark.parametrize("mini_batch_size", [None, 8])
def test_gp_training(training_data, mini_batching, mini_batch_size):
    df, params = training_data
    gp = GP(
        df,
        params,
        nb_training_iter=2,
    )
    gp.train(mini_batching, mini_batch_size)
    gp.eval_perf()


def test_gp_plots(training_data):
    df, params = training_data
    gp = GP(df, params)
    gp.plot_all_solutions("training")
    gp.plot_all_solutions("validation")
    gp.plot_individual_solution(0)
    gp.plot_obs_vs_predicted("training")
    gp.plot_obs_vs_predicted("validation")


def test_gp_incomplete_data(training_data_bootstrapped):
    df, params = training_data_bootstrapped
    gp = GP(df, params, nb_training_iter=2)
    gp.train()
    gp.eval_perf()


@pytest.fixture
def pickle_file():
    model_file = "vpop_calibration/test/gp_model_for_tests.pkl"

    with open(model_file, "wb") as file:
        yield file
    os.remove(model_file)


def test_gp_pickle(training_data, pickle_file):
    df, params = training_data
    gp = GP(df, params)
    pickle.dump(gp, pickle_file)
