# Use the official Python image from the Docker Hub
FROM python:3.11.4

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the working directory
COPY requirements.txt ./

# Install Python dependencies specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the entire content of the current directory (including the code and .env file) into the container's working directory
COPY . ./

# Expose port 5000 (change if your app uses a different port)
EXPOSE 5000

# Specify the command to run the application
CMD ["python", "api.py"]
