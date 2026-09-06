# fr3_trajopt — Drake nightly pinned to the same build rby1-constrained-planning uses
# (see rby1-constrained-planning/pyproject.toml / CLAUDE.md: drake==0.0.20260806, Python 3.12).
# Bump this pin deliberately, not lazily — nightly wheels are retained ~56 days.
#
# models/fr3_marker/meshes/ is a symlink out to ../../cricket/resources/fr3_marker/meshes
# (to avoid duplicating mesh data on disk), which doesn't survive being copied out of this
# directory. Build with the sibling cricket/ directory attached as a second, named build
# context so its real mesh files can replace the symlink (requires BuildKit/buildx, the
# default backend for `docker build` on Docker 23+):
#
#   docker build -f fr3_trajopt/Dockerfile -t fr3-trajopt:dev \
#     --build-context cricket=./cricket fr3_trajopt
#
# (paths above assume you run it from the PVAMP repo root; adjust if not.)
#
# fr3_trajopt is not an installed package -- common.py/scene.py are plain modules, and
# every project path they use (FR3_MARKER_URDF, MAZE_CUBOIDS_JSON) is derived from
# __file__ at import time, not the CWD or any install-time path recorded by pip. That
# means this image's files can be bind-mounted over at runtime (dev iteration) from
# wherever the mount lands, as long as the directory layout underneath stays the same
# (common.py, scene.py, models/, resources/, notebooks/ all siblings) -- no WORKDIR
# matching or reinstall required.
FROM python:3.12-slim-bookworm

# Runtime libs pydrake's import machinery reports as required (verified against this exact
# wheel on Debian bookworm; see drake.mit.edu/installation.html for the Ubuntu equivalents).
RUN apt-get update && apt-get install -y --no-install-recommends \
      libx11-6 \
      libsm6 \
      libglib2.0-0 \
      git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/fr3_trajopt

COPY requirements.txt ./
RUN pip install --no-cache-dir --extra-index-url https://drake-packages.csail.mit.edu/whl/nightly \
      -r requirements.txt

COPY common.py scene.py ./
COPY models/ ./models/
COPY resources/ ./resources/
COPY notebooks/ ./notebooks/

EXPOSE 7000

CMD ["python", "notebooks/fr3_model_test.py"]
