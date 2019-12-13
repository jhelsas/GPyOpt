# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from ..core.bo_biobj import BO_biobj

class ModularBayesianMultiobjectiveOptimization(BO_biobj):

    """
    Modular Bayesian optimization. This class wraps the optimization loop around the different handlers.

    :param model: GPyOpt model class.
    :param space: GPyOpt space class.
    :param objective: GPyOpt objective class.
    :param acquisition: GPyOpt acquisition class.
    :param evaluator: GPyOpt evaluator class.
    :param X_init: 2d numpy array containing the initial inputs (one per row) of the model.
    :param Y_init: 2d numpy array containing the initial outputs (one per row) of the model.
    :param cost: GPyOpt cost class (default, none).
    :param normalize_Y: whether to normalize the outputs before performing any optimization (default, True).
    :param model_update_interval: interval of collected observations after which the model is updated (default, 1).
    :param de_duplication: instantiated de_duplication GPyOpt class.
    """

    def __init__(self, model, space, objective, acquisition, evaluator, X_init, 
                 Y_init=None, C_init = None, model_c = [], cost = None, 
                 normalize_Y = True, model_update_interval = 1, de_duplication=False):

        self.initial_iter = True
        self.modular_optimization = True

        # --- Create optimization space
        super(ModularBayesianMultiobjectiveOptimization ,self).__init__(model                  = model,
                                                                        model_c                = model_c,
                                                                        space                  = space,
                                                                        objective              = objective,
                                                                        acquisition            = acquisition,
                                                                        evaluator              = evaluator,
                                                                        X_init                 = X_init,
                                                                        Y_init                 = Y_init,
                                                                        C_init                 = C_init,
                                                                        cost                   = cost,
                                                                        normalize_Y            = normalize_Y,
                                                                        model_update_interval  = model_update_interval,
                                                                        de_duplication         = de_duplication)
