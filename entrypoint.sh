#!/bin/bash
# entrypoint.sh

set -e

echo "Starting Django application..."

# Wait for database to be ready
echo "Checking database connection..."
sleep 2

# Run migrations
echo "Running database migrations..."
python manage.py migrate --noinput

# Load data if needed (SIMPLIFIED VERSION)
echo "Loading initial data..."
python -c "
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recommendation_frontend.settings')
django.setup()

try:
    from recommendations.load_data import load_data_to_db
    load_data_to_db()
    print('✅ Data loaded successfully')
except Exception as e:
    print(f'⚠️ Data loading failed: {e}')
"

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput --clear || echo "No static files to collect"

# Start server
echo "🚀 Starting Django server..."
exec "$@"