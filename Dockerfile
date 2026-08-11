# 1. Start from a lightweight official Python 3.11 image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements file first (Optimization step)
COPY requirements.txt .

# 4. Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn pydantic

# 5. Copy the rest of the application code into the container
COPY src/ ./src/

# 6. Expose the port the app runs on
EXPOSE 8000

# 7. Define the command to run the API server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]