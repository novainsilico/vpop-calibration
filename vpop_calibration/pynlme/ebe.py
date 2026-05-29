import torch
from scipy.optimize import minimize
from tqdm import tqdm
import numpy as np

from vpop_calibration.config import smoke_test, device
from vpop_calibration.pynlme.model import NlmeModel


def compute_ebe(
    nlme_model: NlmeModel,
    max_iter: int = 50,
) -> torch.Tensor:
    """
    Returns: ebe_estimates: dim(nb_patients, nb_PDU)
    """

    init_etas = nlme_model.eta_samples_chains

    if smoke_test:
        max_iter = 2

    # Taking conditional distribution samples means as a starting point for optimization
    init_samples = init_etas.mean(dim=0)

    ebe_etas = torch.zeros((nlme_model.nb_patients, nlme_model.nb_pdu))
    print("Computing EBEs for each patient:")
    for i, p in tqdm(enumerate(nlme_model.patients)):
        log_posterior_function = nlme_model.single_patient_likelihood_factory(id=p)

        def objective_function(eta_array: np.ndarray) -> float:
            eta_tensor = (
                torch.from_numpy(eta_array).float().view(1, 1, nlme_model.nb_pdu)
            )
            with torch.no_grad():
                log_post = log_posterior_function(eta_tensor).log_posterior
            assert log_post.shape == (1, 1)
            return -log_post.float().item()

        x0 = init_samples[i].numpy()

        res = minimize(
            objective_function,
            x0,
            method="L-BFGS-B",
            options={"maxiter": max_iter},
        )
        ebe_etas[i] = torch.from_numpy(res.x).to(device)

    ebe_etas = ebe_etas.expand(1, -1, -1)
    ebe_gaussian = nlme_model.convert_etas_to_gaussian_all_patients(ebe_etas)
    ebe_physical = nlme_model.convert_gaussian_to_physical(
        ebe_gaussian, nlme_model.log_mi
    )
    return ebe_physical
