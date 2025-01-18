# Use a base image that includes Java and Python
FROM openjdk:11-jre-slim

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Install required Python packages
RUN pip3 install kafka-python pandas requests torch tensorflow keras pyspark flask scikit-fuzzy scipy

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Verify that files are copied correctly
RUN ls -l /app

# List installed packages to verify installation
RUN pip3 list

# Default command
CMD ["python3", "flask_server.py"]