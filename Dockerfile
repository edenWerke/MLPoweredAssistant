# Use the official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (wordnet and omw-1.4)
RUN python -m nltk.downloader wordnet omw-1.4

# Copy the rest of your code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Start FastAPI with Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]