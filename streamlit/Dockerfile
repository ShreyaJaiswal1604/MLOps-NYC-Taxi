# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory to /app
WORKDIR /app

# Copy the contents of the streamlit folder into the container at /app
COPY main.py requirements.txt /app/

# Install Streamlit and any other dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "/app/main.py"]