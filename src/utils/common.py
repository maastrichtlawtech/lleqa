from time import perf_counter


class catchtime:
    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.readout = f'Time: {self.time:.3f} seconds'


def log_step(funct):
    """ Decorator to log the time taken by a function to execute.
    """
    import timeit, datetime
    from functools import wraps
    @wraps(funct)
    def wrapper(*args, **kwargs):
        tic = timeit.default_timer()
        result = funct(*args, **kwargs)
        time_taken = datetime.timedelta(seconds=timeit.default_timer() - tic)
        print(f"Just ran '{funct.__name__}' function. Took: {time_taken}")
        return result
    return wrapper


def read_json_file(path_or_url: str):
    """ Read a JSON file from a local path or URL.
    """
    import re
    import json
    import urllib.request
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    if bool(url_pattern.match(path_or_url)):
        with urllib.request.urlopen(path_or_url) as f:
            return json.load(f)
    with open(path_or_url, 'r') as f:
        return json.load(f)


def set_seed(seed: int):
    """ Ensure that all operations are deterministic on CPU and GPU (if used) for reproducibility.
    """
    import torch
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def count_trainable_parameters(model, verbose=True):
    """ Count the number of trainable parameters in a model.
    """
    all_params = 0
    trainable_params = 0
    for _, p in model.named_parameters():
        all_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    if verbose:
        print(f"Trainable params: {round(trainable_params/1e6, 1)}M || All params: {round(all_params/1e6, 1)}M || Trainable ratio: {round(100 * trainable_params / all_params, 2)}%")
    return trainable_params
