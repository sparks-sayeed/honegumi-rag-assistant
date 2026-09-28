# Dockerfile
# * builder steps:
#  - create environment and build package
#  - save environment for runner
# * runner steps:
#  - recreate the same environment and install built package
#  - optionally execute provided console_script in ENTRYPOINT
#
# Alternatively, remove builder steps, take `environment.docker.yml` from your repo
# and `pip install honegumi_rag_assistant` using an artifact store. Faster and more robust.

# FROM condaforge/mambaforge AS builder
# WORKDIR /root

# COPY . /root
# RUN rm -rf dist build

# RUN mamba env create -f environment.yml
# # RUN pip install -e . # not needed since it's in environment.yml
# SHELL ["mamba", "run", "-n", "honegumi_rag_assistant", "/bin/bash", "-c"]

# # Build package
# RUN tox -e build
# RUN mamba env export -n honegumi_rag_assistant -f environment.docker.yml

FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-large-en-v1.5')"

COPY data/processed/ax_docs_vectorstore /app/data/processed/ax_docs_vectorstore

COPY setup.py setup.cfg pyproject.toml ./
COPY src/ ./src/
RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install --no-cache-dir --no-deps -e .

# Bind to every interface: inside a container 127.0.0.1 is the container's own
# loopback, so -p would publish a port that refuses every connection.
ENV AX_DOCS_VECTORSTORE_PATH=/app/data/processed/ax_docs_vectorstore \
    EMBEDDING_LOCAL_FILES_ONLY=true \
    MCP_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
ENTRYPOINT ["honegumi-rag-mcp"]


# FROM  condaforge/mambaforge AS runner

# # Create default pip.conf.
# # Replace value of `index-url` with your artifact store if needed.
# RUN echo "[global]\n" \
#          "timeout = 60\n" \
#          "index-url = https://pypi.org/simple/\n" > /etc/pip.conf

# # Create and become user with lower privileges than root
# RUN useradd -u 1001 -m app && usermod -aG app app
# USER app
# WORKDIR /home/app

# # WORKAROUND: Assure ownership for older versions of docker
# RUN mkdir /home/app/.cache
# RUN mkdir /home/app/.conda

# # Copy caches, environment.docker.yml and built package
# COPY --from=builder --chown=app:app /root/.cache/pip /home/app/.cache/pip
# COPY --from=builder --chown=app:app /opt/conda/pkgs /home/app/.conda/pkgs
# COPY --from=builder --chown=app:app /root/environment.docker.yml environment.yml
# COPY --from=builder --chown=app:app /root/dist/*.whl .

# RUN mamba env create -f environment.yml
# RUN mamba clean --packages --yes
# RUN rm -rf /home/app/.cache/pip

# # Make RUN commands use the conda environment
# SHELL ["conda", "run", "-n", "honegumi_rag_assistant", "/bin/bash", "-c"]

# RUN pip install ./*.whl

# # Expose a port for a web-service for instance.
# # EXPOSE 80/tcp

# # Code to run when container is started. Replace `YOUR_CONSOLE_SCRIPT` with the
# # value of `console_scripts` in section `[options.entry_points]` of `setup.cfg`.
# # ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "honegumi_rag_assistant", "YOUR_CONSOLE_SCRIPT"]
