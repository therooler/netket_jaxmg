import pickle
from typing import Tuple, Union

import jax
import netket as nk

import jax.numpy as jnp
from flax import serialization
import numpy as np
from jax.experimental.multihost_utils import process_allgather


class Logger:
    def __init__(self, path: str, fields: Union[Tuple[Tuple[str, str], ...], dict[str, str]], log_name: str = 'log',
                 save_every: int = 50,
                 save_params: bool = True,
                 save_samples: bool = True, rank=0):
        """Custom Logger for netket to handle checkpointing"""
        self._path = path
        self._log_name = log_name
        self._save_every = save_every
        self._save_params = save_params
        self._save_samples = save_samples
        self._rank = rank
        self._done = False

        if isinstance(fields, dict):
            self._fields = []
            for f in fields.keys():
                self._fields.append((f, fields[f]))
            self._fields = tuple(self._fields)

        else:
            self._fields = fields
        self.initialize()

    @property
    def path(self):
        return self._path

    @property
    def done(self):
        return self._done

    @path.setter
    def path(self, val):
        self._path = val

    @property
    def data(self):
        return self._data

    @property
    def step(self):
        return self._step

    def __call__(self, step: int, logdata, state: nk.vqs.VariationalState):
        """Call the logger and store the data in a sensible way"""
        self._data['iters']['values'].append(self._step)
        # assert that the field is in the logger
        for f in self._fields:
            if f[0] in logdata.keys():
                field = f[0]
                attribute = f[1]
                # Make empty list on first call
                if field not in self._data.keys():
                    self._data[field] = {}
                if attribute not in self._data[field].keys():
                    self._data[field][attribute] = []
                if isinstance(logdata[field], nk.stats.Stats):
                    self._data[field][attribute].append(np.array(getattr(logdata[field], attribute)).squeeze())
                else:  # assume it's a scalar
                    self._data[field][attribute].append(np.array(logdata[field]).squeeze())

        if (self._step + 1) % self._save_every == 0:
            self.flush(state)
        self._step += 1

    def flush(self, state: nk.vqs.VariationalState = None, done=False, var_name=""):
        """Save the model"""
        if self.rank == 0:
            with open(self._path + self._log_name + f"{var_name}.log", "wb") as file:
                self._data['done'] = done
                pickle.dump(self._data, file)
            if self._save_params and state is not None:
                self.save_parameters(state, var_name)
            # if self._save_samples and state is not None:
            #     self.save_samples(state, var_name)

    def save_parameters(self, state: nk.vqs.VariationalState, var_name=""):
        binary_data = serialization.to_bytes(state.variables)
        with open(self._path + self._log_name + f"_params{var_name}.mpack", "wb") as outfile:
            outfile.write(binary_data)

    def save_samples(self, state: nk.vqs.VariationalState, var_name=""):
        jnp.save(self._path + self._log_name + f"_samples{var_name}.npy",
                 np.array(process_allgather(state.samples)))

    def restore(self, state: nk.vqs.VariationalState = None, var_name=""):
        """Restore the state from the logger"""
        try:
            with open(self._path + self._log_name + f"{var_name}.log", "rb") as file:
                self._data = pickle.load(file)
                for f in self._fields:
                    if f[0] not in self._data.keys():
                        print(fr"No data found for `{f[0]}` when restoring logger.")
                        if f[0] not in self._data.keys():
                            self._data[f[0]] = {}
                        self._data[f[0]][f[1]] = []
        except FileNotFoundError:
            return False
        except EOFError:
            print("Corrupt log")
            return False
        except pickle.UnpicklingError:
            print("Corrupt log")
            return False

        self._step = self._data['iters']['values'][-1] + 1
        self._done = self._data['done']
        if self._save_params and state is not None:
            return self.restore_state(state, var_name)
        else:
            return True

    def __getitem__(self, item: str):
        return self._data[item]

    def restore_state(self, state: nk.vqs.VariationalState, var_name):
        try:
            with open(self._path + self._log_name + f"_params{var_name}.mpack", "rb") as infile:
                binary_data = infile.read()
                state.variables = serialization.from_bytes(state.variables, binary_data)
                state.variables = jax.tree.map(lambda x: jnp.array(x), state.variables)
            return True
        except FileNotFoundError:
            return False
        except ValueError:
            print("Corrupt weights, restarting...")
            return False

    def restore_samples(self, state: nk.vqs.VariationalState, var_name=""):
        try:
            samples = jnp.load(self._path + self._log_name + f"_samples{var_name}.npy")
            target_samples = state.sample()
            try:
                samples = jax.device_put(samples, target_samples.sharding)
            except ValueError:
                raise ValueError("Unable to shard restored samples")
            if samples.shape!=target_samples.shape:
                raise ValueError(f"Shape mismatch between restored samples and sampler state: {samples.shape} != {target_samples.shape}")
            target_shape = state.sampler_state.σ.shape
            state.sampler_state = state.sampler_state.replace(σ=samples.reshape(target_shape))
            return True
        except FileNotFoundError:
            return False
        except ValueError:
            print("Corrupt samples, restarting...")
            return False

    @property
    def rank(self):
        return self._rank

    def initialize(self):
        self._step = 0
        self._data = {}
        self._data['iters'] = {'values': []}
        allowed_attributes = ['Mean', 'Variance', 'values', 'R_hat', 'Generator', 'tau_corr', ]

        for f in self._fields:
            assert isinstance(f[0], str) and isinstance(f[1], str), 'Fields must be a tuple of type `(str, str)`'
            assert f[1] in allowed_attributes, \
                f'Attribute {f[1]} is not in list of allowed attributes {allowed_attributes}'
            if f[0] not in self._data.keys():
                self._data[f[0]] = {}
            self._data[f[0]][f[1]] = []
        self._data['done'] = False

    def merge(self, other):
        assert self._fields == other._fields
        self._data['iters']['values'].extend(
            [i + len(self._data['iters']['values']) for i in other._data['iters']['values']])
        for f in self._fields:
            self._data[f[0]][f[1]].extend(other[f[0]][f[1]])
