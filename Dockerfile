# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (Heroku dynamically assigns it)
EXPOSE 5000

# Command to start the application
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT:-5000}", "app:app"]
