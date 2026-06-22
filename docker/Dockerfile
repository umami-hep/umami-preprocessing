FROM python:3.14-trixie

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git nano vim \
    && rm -rf /var/lib/apt/lists/*

# copy and install package together with the dev dependency group (PEP 735)
COPY . .
RUN python -m pip install --no-cache-dir -e . \
    && python -m pip install --no-cache-dir --group dev
