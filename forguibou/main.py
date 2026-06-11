from vpop_calibration import *

import pandas as pd
import numpy as np


# Create a placeholder haskell callback
def haskell_callback(
    patients: pd.DataFrame, time_points: list[float], outputs: list[str]
) -> np.ndarray:
    nb_patients = patients.shape[0]
    nb_timesteps = len(time_points)
    nb_outputs = len(outputs)
    dummy_output = np.ones((nb_patients, nb_timesteps, nb_outputs))
    return dummy_output


# Define an obesrvations dataset
nb_patients = 5
patients = [f"p{i}" for i in range(nb_patients)]

timesteps = list(range(5))
df = (
    pd.DataFrame({"id": patients, "protocol_arm": "identity"})
    .merge(pd.DataFrame({"time": timesteps}), how="cross")
    .merge(pd.DataFrame({"output_name": ["out_1", "out_2"]}), how="cross")
)
df["value"] = 0.0

# Define the model priors
nlme_priors = {
    "pdu": {
        "foo": {
            "prior": 10.0,
            "prior_omega": 0.1,
        },
        "bar": {
            "prior": 10.0,
            "prior_omega": 0.1,
        },
        "baz": {
            "prior": 10.0,
            "prior_omega": 0.1,
        },
    },
    "error_model": {
        "out_1": {"error_type": "additive", "sigma": 0.1},
        "out_2": {"error_type": "proportional", "sigma": 0.5},
    },
}

# Create the main config object
config = Config(saem=SaemConfigDict(nb_phase1_iterations=10, nb_phase2_iterations=10))


# Instantiate the binding class
simwork_model = SimworkModelBinding(
    haskell_callback=haskell_callback,
    inputs=["foo", "bar", "baz"],
    outputs=["out_1", "out_2"],
)
# Create the structural model
struct_model = StructuralSimwork(simwork_model)

# Create the main interface
nlme_model = NlmeModel(
    df=df, prior_params=nlme_priors, structural_model=struct_model, config=config
)
# Run the optimizer
nlme_model.optimizer.run()
