#!/bin/sh
apt-get update && apt-get install -y libglib2.0-0
gunicorn --bind 0.0.0.0:8000 app:app  # Adjust if your app uses another entry point
