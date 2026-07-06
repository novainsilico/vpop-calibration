from vpop_calibration.interface import NlmeModel

import pandas as pd


def run_saem(nlme_model: NlmeModel) -> pd.DataFrame:
    nlme_model.optimizer.run()

    return nlme_model.optimizer.history
