self.optimizer = PerturbedGradientDescent(
    self.model.parameters(),
    lr=params['learning_rate'],
    mu=params['mu']
) 