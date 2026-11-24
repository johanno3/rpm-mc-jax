FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml LICENSE ./
COPY src ./src
COPY tests ./tests
COPY validation ./validation
RUN pip install --no-cache-dir -e ".[jax,test]"

# Default: run the test suite (JAX implementation against the NumPy reference).
CMD ["python", "-m", "pytest", "-q"]
