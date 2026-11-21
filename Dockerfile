# Blast Radius selector.
# The base image is the one the commits were labeled in, so re-running sqlglot's
# tests inside it behaves the way the labels were produced.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# The packages sqlglot's tests were labeled with (PyPI).
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir --root-user-action=ignore -r /app/requirements.lock

# sqlglot at the pinned commit, straight from its own repository.
COPY tools/fetch_sqlglot.py /app/tools/fetch_sqlglot.py
RUN python /app/tools/fetch_sqlglot.py eb11aec4be8cb2122afe713a4816662c3f0fb2fa /app/src

COPY selector /app/selector
COPY maps /app/maps
COPY tools /app/tools

# Bytecode for everything the run imports (the run-time disk is read-only), and the
# fingerprint of what importing the pinned sqlglot computes (see selector/state.py).
RUN python -m compileall -q /app/src/sqlglot /app/src/tests /app/selector \
    && python /app/tools/build_state.py /app/src /app/maps/state.json.z \
    && chmod -R a+rX /app

# The run has 2 cores and a cap on processes/threads: numeric libraries must not start
# one thread per core of the host.
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    SKIP_INTEGRATION=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1
WORKDIR /app
CMD ["python", "-m", "selector.main"]
