# Reproduzierbare Laufzeit: offizielles Freqtrade-Image (Version fest gepinnt) + kitb-Paket.
FROM freqtradeorg/freqtrade:2026.9

USER root
COPY pyproject.toml constraints.txt /opt/kitb/
COPY src /opt/kitb/src
# --no-deps: alle Laufzeitabhängigkeiten kommen aus dem Freqtrade-Image. Nur pytest zusätzlich.
RUN pip install --no-cache-dir --no-deps /opt/kitb \
 && pip install --no-cache-dir "pytest==9.1.1"
USER ftuser

ENV KITB_HOME=/kitb
WORKDIR /kitb
ENTRYPOINT ["kitb"]
CMD ["check-safety"]
