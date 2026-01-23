FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
# copy model (from repo root or output/) - when the workflow builds, make sure to copy the correct path
# In our workflow we will ensure model.pkl is at repo root before docker build (downloaded artifact at ./output/model.pkl)
COPY output/model.pkl ./model.pkl

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
