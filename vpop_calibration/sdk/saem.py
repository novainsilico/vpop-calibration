from vpop_calibration.sdk.model import NlmeInterface

import pandas as pd


def run_saem(nlme_model: NlmeInterface) -> pd.DataFrame:
    nlme_model.optimizer.run()

    # Do we need a more structured output?
    return nlme_model.optimizer.history
