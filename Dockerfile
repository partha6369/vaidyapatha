# Smaller base, Debian trixie (same family as HF runners)
FROM python:3.10-slim

# ---- Environment (speed & reliability) ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/home/user/nltk_data

# ---- System dependencies ----
# Note: libgl1 replaces removed libgl1-mesa-glx on Debian trixie
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 cmake rsync \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install --system

# ---- App directory & user ----
WORKDIR /home/user/app
RUN useradd -m -u 1000 user || true && chown -R 1000:1000 /home/user
USER 1000:1000

# ---- Dependencies (cache-friendly) ----
# Copy only requirements first to leverage Docker layer caching
COPY --chown=1000:1000 requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip wheel \
 && pip install --no-cache-dir -r requirements.txt

# Optional: pre-download NLTK data to avoid runtime writes
# RUN python -m nltk.downloader -d "$NLTK_DATA" punkt punkt_tab || \
#     python -m nltk.downloader -d "$NLTK_DATA" punkt

# ---- App code ----
COPY --chown=1000:1000 . .

# ---- Run ----
# Your app should bind to 0.0.0.0 and use PORT env var inside app.py
CMD ["python", "VaidyaPatha.py"]