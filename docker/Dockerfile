# Reproduzierbare Laufzeit: offizielles Freqtrade-FreqAI-Image (Version fest gepinnt, enthält scikit-learn, LightGBM, XGBoost) + kitb-Paket.
FROM freqtradeorg/freqtrade:2026.9_freqai

USER root
COPY pyproject.toml constraints.txt /opt/kitb/
COPY src /opt/kitb/src
# --no-deps: Laufzeitabhängigkeiten kommen aus dem Freqtrade-Image. Zusätzlich nur pytest und das Dashboard.
RUN pip install --no-cache-dir --no-deps /opt/kitb \
 && pip install --no-cache-dir -c /opt/kitb/constraints.txt "pytest==9.1.1" "streamlit==1.66.1"
USER ftuser

ENV KITB_HOME=/kitb
WORKDIR /kitb
ENTRYPOINT ["kitb"]
CMD ["check-safety"]
