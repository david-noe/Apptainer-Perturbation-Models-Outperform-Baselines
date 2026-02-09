FROM condaforge/mambaforge:latest

# Install system dependencies
RUN apt-get update && apt-get install -y wget git build-essential && rm -rf /var/lib/apt/lists/*

# Copy conda environment spec and create environment
COPY docker/cellflow/env.yaml /tmp/env.yaml
RUN mamba env create -f /tmp/env.yaml && mamba clean -afy

# Install additional pip dependencies into the cellflow conda env
RUN mamba run -n cellflow pip install --no-cache-dir \
    flax orbax dask diffrax pyarrow \
    ott-jax==0.5 \
    adjusttext coverage session-info \
    requests

# Clone CellFlow, apply patch to remove version pins, install
RUN git clone https://github.com/theislab/CellFlow.git /tmp/CellFlow \
    && cd /tmp/CellFlow \
    && git checkout 0f55ba20c49bcee60da979bd5a9fb4ab16420b1e
RUN cd /tmp/CellFlow \
    && sed -i 's/"scikit-learn==1.5.1"/"scikit-learn"/' pyproject.toml \
    && sed -i 's/"scipy<1.16"/"scipy"/' pyproject.toml \
    && mamba run -n cellflow pip install --no-deps .

# Install JAX with CUDA support
RUN mamba run -n cellflow pip install -U "jax[cuda12]"

# Install CellSimBench
COPY pyproject.toml /tmp/pyproject.toml
COPY cellsimbench /tmp/cellsimbench
RUN mamba run -n cellflow pip install -e /tmp/

# Create app and mount directories
WORKDIR /app
RUN mkdir -p /app/data /app/models /data /model_output /output

# Copy CellFlow wrapper files
COPY docker/cellflow/run_model.py /app/run_model.py
COPY docker/cellflow/cellflow_wrapper.py /app/cellflow_wrapper.py
COPY docker/cellflow/utils.py /app/utils.py

# Make run_model.py executable
RUN chmod +x /app/run_model.py

# Set up PYTHONBREAKPOINT for debugging
ENV PYTHONBREAKPOINT="ipdb.set_trace"
RUN echo "export PYTHONBREAKPOINT='ipdb.set_trace'" >> ~/.bashrc

ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", "cellflow", "python", "/app/run_model.py"]
