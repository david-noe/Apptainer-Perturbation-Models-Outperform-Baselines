# Actual working install

```
conda env create -f env.yaml
conda activate cellflow

# Install additional dependencies
pip install flax orbax dask diffrax pyarrow 
pip install ott-jax==0.5
pip install adjusttext coverage session-info

git clone https://github.com/theislab/CellFlow.git
cd CellFlow
git checkout 0f55ba20c49bcee60da979bd5a9fb4ab16420b1e
## APPLY PATCH TO pyproject.toml ##
## SEE diff.txt FOR PATCH ##
pip install --no-deps . # Uses modified pyproject without pins in main deps

pip install requests
pip install -U "jax[cuda12]"

# Test
python -c "import rapids_singlecell as rsc"
python -c "import torch"
python -c "import cellflow"
```