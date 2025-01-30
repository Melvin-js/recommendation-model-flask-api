# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 80 for the Flask app
EXPOSE 80

# Run the Flask app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:80", "api.index:app"]