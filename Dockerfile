# Analysis UI — one image with the web app, the Python engine and the R engine.
#
#   docker build -t analysis-ui .
#   docker run --rm -p 8765:8765 analysis-ui      ->  http://localhost:8765

FROM node:22-slim AS web
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
# vite writes the build to ../backend/analysis_ui/static
RUN mkdir -p /src/backend/analysis_ui && npm run build

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip \
      r-base-core r-cran-ggplot2 r-cran-jsonlite r-cran-svglite r-cran-emmeans r-cran-lme4 \
      r-cran-lmertest r-cran-car r-cran-survival r-cran-multcomp r-cran-afex r-cran-ggpubr \
      r-cran-ggbeeswarm r-cran-ggsignif r-cran-proc r-cran-rstatix r-cran-patchwork r-cran-ggrepel \
      r-cran-viridis r-cran-scales r-cran-minpack.lm r-cran-nlme r-cran-survminer r-cran-ggridges \
      r-cran-effectsize r-cran-broom r-cran-pheatmap r-cran-ggforce r-cran-cowplot r-cran-rcolorbrewer \
      r-cran-haven r-cran-readxl r-cran-dplyr r-cran-tidyr r-cran-mass r-cran-nortest r-cran-ggthemes \
      r-cran-ggsci r-cran-pwr r-cran-psych r-cran-systemfonts r-cran-ragg r-cran-gridextra r-cran-coin \
      r-cran-epitools \
      fonts-dejavu fonts-liberation fonts-open-sans fonts-roboto fonts-lato \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/ /app/backend/
COPY --from=web /src/backend/analysis_ui/static /app/backend/analysis_ui/static
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -e /app/backend
ENV PATH=/opt/venv/bin:$PATH
EXPOSE 8765
CMD ["analysis-ui", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]
