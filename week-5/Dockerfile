FROM agrigorev/zoomcamp-model:2025

# Install uv
RUN pip install uv

# Copy our project's dependency files
COPY pyproject.toml uv.lock /code/

# Set the working directory
WORKDIR /code

# Use uv to install all our dependencies (fastapi, uvicorn, etc.)
RUN uv pip install --system -r pyproject.toml

COPY main.py /code/

EXPOSE 8000

# The command to run when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]