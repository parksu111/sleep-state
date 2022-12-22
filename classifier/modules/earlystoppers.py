import numpy as np
import logging

class EarlyStopper():

    def __init__(self, patience: int, logger: logging.RootLogger=None) -> None:
        self.patience = patience
        self.logger = logger

        self.patience_counter = 0
        self.stop = False
        self.best_loss = np.inf

        msg = f'Initiated early stopper, best score: {self.best_loss}, patience: {self.patience}'
        self.logger.info(msg) if self.logger else None

    def check_early_stopping(self, loss: float) -> None:
        if loss > self.best_loss:
            self.patience_counter += 1
        else:
            self.patience_counter = 0
            self.best_loss = loss
        if self.logger is not None:
            msg = f"Early stopper, counter {self.patience_counter}/{self.patience}, best:{self.best_loss} -> now: {loss}"
            self.logger.info(msg)



