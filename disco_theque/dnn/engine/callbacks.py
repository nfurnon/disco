import numpy as np


class SaveAndStop:
    """
    Returns boolean to save model if it is the current best one.
    Also keep track of how many times we have not improved. If it exceeds the patience, the function `early_stop` returns
    a bool to say that learning can be stopped.
    """
    def __init__(self, patience=np.inf, mode='min', delta=0):
        self.waited = 0     # Number of epoches without improvement
        self.patience = patience
        self.mode = mode
        self.delta = delta
        if mode == 'min':
            self.current_value = np.inf
        elif mode == 'max':
            self.current_value = -np.inf
        else:
            raise ValueError('`mode` can be only "min" or "max"')

    def save_model_query(self, value):
        """
        Returns a bool to say whether we want to save the model or not.
        Args:
            value (float): value of the (validation) loss
        Returns:
            bool. `True` if `value` is a better (in terms of training) value than the previous ones.
        """
        if self.mode == 'min':
            if value < self.current_value - self.delta:
                self.current_value = value
                out = True
                self.waited = 0
            else:
                out = False
                self.waited += 1
        else:
            if value > self.current_value + self.delta:
                self.current_value = value
                out = True
                self.waited = 0
            else:
                out = False
                self.waited += 1

        return out

    def early_stop_query(self):
        """Returns a boolean value which is `True` if the validation loss has not improved (in terms of training)
        for `self.patience` epochs""".
        if self.waited > self.patience:
            return True
        else:
            return False

