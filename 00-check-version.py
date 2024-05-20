# %%
import polars as pl
import numpy as np
import torch
import tensorflow as tf
import torch.version
import subprocess
import transformers

print(f"======================================================================")
subprocess.run(["nvcc", "--version"])
print(f"======================================================================")
print(f"Tersorflow : {tf.version.VERSION}")
print(f"Tersorflow[CUDA] : {'OK' if tf.test.is_built_with_cuda() else 'disabled'}")

print(f"======================================================================")
print(f"Pytorch : {torch.version.__version__}")
print(
    f"Pytorch[CUDA] : {f'OK({torch.version.cuda})' if torch.cuda.is_available() else 'disabled'}"
)

# %%

num_rows = 5000
rng = np.random.default_rng(seed=7)

buildings_data = {
    "sqft": rng.exponential(scale=1000, size=num_rows),
    "year": rng.integers(low=1995, high=2000, size=num_rows),
    "building_type": rng.choice(["A", "B", "C"], size=num_rows),
}
buildings = pl.DataFrame(buildings_data)

buildings.plot.bar(x="year", y="sqft", width=900, title="Buildings by year")
