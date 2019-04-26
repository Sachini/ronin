import json

import numpy as np


class MSEAverageMeter:
    def __init__(self, ndim, retain_axis, n_values=3):
        """
        Calculate average without overflows
        :param ndim: Number of dimensions
        :param retain_axis: Dimension to get average along
        :param n_values: Number of values along retain_axis
        """
        self.count = 0
        self.average = np.zeros(n_values, dtype=np.float64)
        self.retain_axis = retain_axis
        self.targets = []
        self.predictions = []
        self.axis = tuple(np.setdiff1d(np.arange(0, ndim), retain_axis))

    def add(self, pred, targ):
        self.targets.append(targ)
        self.predictions.append(pred)
        val = np.average((targ - pred) ** 2, axis=self.axis)
        c = np.prod([targ.shape[i] for i in self.axis])
        ct = c + self.count
        self.average = self.average * (self.count / ct) + val * (c / ct)
        self.count = ct

    def get_channel_avg(self):
        return self.average

    def get_total_avg(self):
        return np.average(self.average)

    def get_elements(self, axis):
        return np.concatenate(self.predictions, axis=axis), np.concatenate(self.targets, axis=axis)


def load_config(default_config, args, unknown_args):
    """
    Combine the arguments passed by user with configuration file given by user [and/or] default configuration. Convert extra named arguments to correct format.
    :param default_config: path to file
    :param args: known arguments passed by user
    :param unknown_args: unknown arguments passed by user
    :return: known_arguments, unknown_arguments
    """
    kwargs = {}

    def convert_value(y):
        try:
            return int(y)
        except:
            pass
        try:
            return float(y)
        except:
            pass
        if y == 'True' or y == 'False':
            return y == 'True'
        else:
            return y

    def convert_arrry(x):
        if not x:
            return True
        elif len(x) == 1:
            return x[0]
        return x

    i = 0
    while i < len(unknown_args):
        if unknown_args[i].startswith('--'):
            token = unknown_args[i].lstrip('-')
            options = []
            i += 1
            while i < len(unknown_args) and not unknown_args[i].startswith('--'):
                options.append(convert_value(unknown_args[i]))
                i += 1
            kwargs[token] = convert_arrry(options)

    if 'config' in kwargs:
        args.config = kwargs['config']
        del kwargs['config']
    with open(args.config, 'r') as f:
        config = json.load(f)

    values = vars(args)

    def add_missing_config(dictionary, remove=False):
        for key in values:
            if values[key] in [None, False] and key in dictionary:
                values[key] = dictionary[key]
                if remove:
                    del dictionary[key]

    add_missing_config(kwargs, True)        # specified args listed as unknowns
    add_missing_config(config)              # configuration from file for unspecified variables
    if args.config != default_config:       # default config
        with open(default_config, 'r') as f:
            default_configs = json.load(f)
        add_missing_config(default_configs)

    try:
        if args.channels is not None and type(args.channels) is str:
            args.channels = [int(i) for i in args.channels.split(',')]
    except:
        pass

    if 'kwargs' in config:
        kwargs = {**config['kwargs'], **kwargs}

    return args, kwargs
