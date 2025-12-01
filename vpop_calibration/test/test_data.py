import pandas as pd

from vpop_calibration.model.data import TrainingDataSet
from vpop_calibration.test import *

training_df = pd.DataFrame(
    {
        "id": ["1", "1", "2", "2"],
        "protocol_arm": ["arm-A"] * 4,
        "output_name": ["s1", "s2", "s1", "s2"],
        "k1": [1.0, 1.0, 2.0, 2.0],
        "value": [0.0, 1.0, 2.0, 3.0],
    }
)


def test_loading():
    TrainingDataSet(training_df, ["k1"], 1.0)
    TrainingDataSet(training_df, ["k1"], 1.0, data_already_normalized=True)
    TrainingDataSet(training_df, ["k1"], 1.0, log_inputs=["k1"], log_outputs=["s1"])
    TrainingDataSet(training_df.drop(columns={"protocol_arm"}), ["k1"], 0.5)
