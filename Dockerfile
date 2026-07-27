# ============================================================================
#  Bolgrot — image de déploiement (Hugging Face Spaces, SDK Docker)
#  Multi-stage : (1) build du front Vue, (2) runtime Python qui sert front + API.
# ============================================================================

# ---------- Stage 1 : build du front Vue -> /web/dist -----------------------
FROM node:20-slim AS front
WORKDIR /web
# On copie d'abord les manifests seuls : tant qu'ils ne changent pas, la couche
# `npm ci` reste en cache même si le code du front change (build plus rapide).
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build            # -> /web/dist (bundle statique)

# ---------- Stage 2 : runtime Python ----------------------------------------
FROM python:3.12-slim

# HF Spaces (SDK Docker) exécute le conteneur en utilisateur non-root UID 1000.
# On crée cet utilisateur et on travaille dans son home (writable : cache torch,
# etc.), sinon des écritures échoueraient en prod.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    SDL_VIDEODRIVER=dummy \
    AZ_POOL_WORKERS=2

WORKDIR /home/user/app

# torch EN PREMIER et isolé (le reste vient de PyPI). L'install dépend de l'archi :
#  - x86_64  : `pip install torch` tirerait la build CUDA (~2-3 Go) -> on force
#              l'index CPU (torch==2.12.1, aligné uv.lock) -> image ~700 Mo.
#  - aarch64 : torch de PyPI est DÉJÀ CPU-only (pas de CUDA sur ARM) et l'index
#              +cpu n'a pas forcément le wheel ARM -> on prend PyPI, non épinglé
#              (le modèle az.pt est un simple state_dict, compatible entre versions).
# Cible retenue = VM Oracle Ampere (ARM) -> branche aarch64. Le Dockerfile marche
# aussi en x86_64 (build local, autre hébergeur).
RUN ARCH="$(uname -m)" && \
    if [ "$ARCH" = "x86_64" ]; then \
        pip install --no-cache-dir --user \
            --index-url https://download.pytorch.org/whl/cpu torch==2.12.1 ; \
    else \
        pip install --no-cache-dir --user torch ; \
    fi

# Reste des deps runtime (numpy/fastapi/uvicorn/pygame) depuis PyPI.
COPY --chown=user requirements-deploy.txt ./
RUN pip install --no-cache-dir --user -r requirements-deploy.txt

# Vérif fail-fast : si un import casse (ex. lib système manquante pour pygame),
# le BUILD échoue avec un message clair plutôt que le conteneur en prod.
RUN python -c "import torch, numpy, fastapi, uvicorn, pygame; print('imports OK', torch.__version__)"

# Code applicatif + modèle AZ (src/ai/az.pt est versionné) + front buildé.
COPY --chown=user src/ ./src/
COPY --chown=user server/ ./server/
COPY --chown=user --from=front /web/dist ./web/dist

# Port : honore $PORT si l'hébergeur l'injecte (Google Cloud Run l'impose), sinon
# 7860 par défaut. `exec` -> uvicorn devient PID 1 et reçoit bien SIGTERM (arrêt
# propre / scale-to-zero). Compatible Cloud Run ($PORT), Oracle VM, HF (7860)...
ENV PORT=7860
EXPOSE 7860
CMD exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT}
