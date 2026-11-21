# Mech Royale bot. Build: docker build -t mech-bot .
# Match sandbox: no network, read-only root (writable /tmp 64 MB), 1 CPU, 512 MB, uid 65534, map at /arena.
FROM python:3.13.1-slim
WORKDIR /bot
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Build-time precompute of static map grids (map never changes). Output: /bot/data/map.npz
RUN python3 tools/precompute_map.py --arena /bot/arena --out /bot/data/map.npz
ENTRYPOINT ["python3", "-u", "/bot/main.py"]
