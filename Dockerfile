FROM python:3.11-slim

WORKDIR /app

# Install nmap for the attacker scripts
RUN apt-get update && apt-get install -y nmap iputils-ping && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Step 1: Install PyTorch CPU from PyTorch's own wheel server FIRST
# This must be separate because +cpu builds don't exist on PyPI
RUN pip install --no-cache-dir torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install everything else from PyPI (torch is already satisfied, pip skips it)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Default command keeps the container alive
CMD ["tail", "-f", "/dev/null"]