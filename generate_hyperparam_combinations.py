import itertools
import yaml


if __name__ == "__main__":
    # file to which hyperparameter sets will be written to
    OUT_FNAME = "hyperparameters.yml"

    # hyperparameters to tune
    hyperparams = {'max_leaf': [5000, 10000],
                    'loss': ['Log'],
                    'l2': [1e-10, 1e-8],
                    'learning_rate': [5e-3, 5e-2],
                    'verbose': [True]}

    # generate a list of dicts with specific values for each parameter
    keys, values = zip(*hyperparams.items())
    hyperparam_allsets = [dict(hyperparam_set=dict(zip(keys, v))) for v in itertools.product(*values)]
    print("Total number of hyperparameter sets: "+str(len(hyperparam_allsets)))

    # write to yaml file, each "entry" is a particular realization of hyperparameters
    with open(OUT_FNAME, 'w') as outfile:
        yaml.dump(hyperparam_allsets, outfile)
    print("Hyperparameter sets saved to: " + OUT_FNAME)
