import numpy as np


class EarlyStopping(object):

    # Credit to https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
    def __init__(
        self, name, mode="min", min_delta=0.001, patience=10, percentage=False
    ):
        """Stops the model training from overfitting by checking metric improvement

        Sets up a "is_better" lamda function for comparing the new metric with prior metric. If the new
        metric is better (mode determines if it should be lower: 'min' or higher: 'max), by the min_delta.

        Can be used multiple times to simultaneously track metric and another metric if desired.

        Args:
            name (str): useful for logging if there are multiple instances of EarlyStopping
            mode (str, optional): minimizing or maximizing the variable being tracked. Defaults to "min".
            min_delta (float, optional): minimum delta of improvement required. Defaults to 0.001.
            patience (int, optional): number of times to allow no improvement before stopping (returning false). Defaults to 10.
            percentage (bool, optional): improvement measured as a percentage or an abasolute value. Defaults to False (absolute value).
        """
        self.name = name
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.percentage = percentage

        self.best_so_far = None
        self.num_epochs_w_no_improvement = 0
        self.is_better = None
        self._init_is_better()

    def _init_is_better(self):
        """ Set up the is_better lambda function for checking if the metric has improved

        Raises:
            ValueError: metric must be either increasing 'max' or decreasing 'min'
        """
        if self.mode not in {"min", "max"}:
            raise ValueError(f"self.mode {self.mode} is unknown! mode must be either increasing 'max' or decreasing 'min'")

        if self.patience == 0:
            # if patience is 0 we can directly compare the new and old values
            self.is_better = lambda a, b: True
            self.step = lambda a: False
        else:
            if not self.percentage:
                if self.mode == "min":
                    self.is_better = lambda a, best: a < best - self.min_delta
                if self.mode == "max":
                    self.is_better = lambda a, best: a > best + self.min_delta
            else:
                if self.mode == "min":
                    self.is_better = lambda a, best: a < best - (best * self.min_delta / 100)
                if self.mode == "max":
                    self.is_better = lambda a, best: a > best + (best * self.min_delta / 100)

    def step(self, metric, verbose=True):
        """Compare metric against last time step to determine if training should stop.



        Args:
            metric (float): Metric can be a loss value or a model performance metric (e.g. accuracy)
            verbose (bool): Print a status explain why stop or continue is recommended

        Returns:
            [bool]: Indication of whether or not to stop now (True-> stop; False-> continue)
        """
        if self.best_so_far is None:
            if verbose:
                print("Initializing best metric score so far.")
            self.best_so_far = metric
            return False

        if np.isnan(metric):
            if verbose:
                print("NAN score, stopping model training")
            return True

        if self.is_better(metric, self.best_so_far):
            if verbose:
                print("Metric improved, updating best so far to:", metric)
            self.num_epochs_w_no_improvement = 0
            self.best_so_far = metric

        else:
            if verbose:
                print("No improvement, thinning patience")
            self.num_epochs_w_no_improvement += 1

        print(f"Num epochs without improvement: {self.num_epochs_w_no_improvement}, patience is {self.patience}")
        if self.num_epochs_w_no_improvement >= self.patience:
            if verbose:
                print(f"No improvement, {self.name} lost patience. Recommending stopping.")
            return True

        return False