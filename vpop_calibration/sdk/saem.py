from vpop_calibration.interface import NlmeModel

import pandas as pd


def run_saem(nlme_model: NlmeModel) -> pd.DataFrame:
    nlme_model.optimizer.run()

    # Do we need a more structured output?
    return nlme_model.optimizer.history
