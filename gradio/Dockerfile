FROM registry.access.redhat.com/ubi9/python-311

# Set working directory
WORKDIR /app

# Use non-root user
USER 1001

# Copy application code and models
COPY --chown=1001:0 . .

# Install dependencies and fix permissions
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -f requirements.txt

# Expose Gradio port
EXPOSE 7860

# Run app
CMD ["python", "app.py"]
