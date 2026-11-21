FROM alpine/git:v2.52.0@sha256:4a0e72d49596a1f5d3701aeedafdadc5c0da4062be4657c7bdc4017387f591cc AS checkout
WORKDIR /sqlglot
RUN git init -q && \
    git fetch --depth 1 https://github.com/tobymao/sqlglot.git eb11aec4be8cb2122afe713a4816662c3f0fb2fa && \
    git checkout -q --detach FETCH_HEAD

# Use the scorer's Python and locked dependencies across both coverage build and runtime.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS runtime-deps
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock

FROM runtime-deps AS coverage-env
WORKDIR /sqlglot
COPY --from=checkout /sqlglot /sqlglot
COPY build_coverage.py /app/build_coverage.py

FROM coverage-env AS coverage-builder
RUN SKIP_INTEGRATION=0 PYTHON_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/sqlglot \
    python /app/build_coverage.py /app/coverage.pkl.zlib

FROM scratch AS coverage-export
COPY --from=coverage-builder /app/coverage.pkl.zlib /coverage.pkl.zlib

FROM runtime-deps
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/sqlglot \
    OPENBLAS_NUM_THREADS=1 \
    SKIP_INTEGRATION=0
COPY --from=checkout /sqlglot/sqlglot /sqlglot/sqlglot
COPY --from=checkout /sqlglot/tests /sqlglot/tests
COPY --from=checkout /sqlglot/sqlglotc /sqlglot/sqlglotc
COPY selector.py patches.py schedule.py equiv.py runner.py coverage.pkl.zlib extra_index.pkl.zlib ./
RUN python -m compileall -q /app /sqlglot/sqlglot /sqlglot/tests && \
    python /app/runner.py && \
    chmod -R a+rX /app /sqlglot
CMD ["python", "/app/selector.py"]
