# Deploying Stock Spike Replicator

This guide provides instructions on how to deploy the Stock Spike Replicator application to a production environment. It covers server requirements, environment variable configuration, database setup, and steps for deploying both the backend and frontend.

## Table of Contents

1. [Server Requirements](#server-requirements)
2. [Environment Variable Configuration](#environment-variable-configuration)
3. [Database Setup](#database-setup)
4. [Backend Deployment](#backend-deployment)
5. [Frontend Deployment](#frontend-deployment)
6. [Nginx Configuration](#nginx-configuration)
7. [SSL Certificate Setup](#ssl-certificate-setup)
8. [Monitoring and Logging](#monitoring-and-logging)

## Server Requirements

- Ubuntu 20.04 LTS or later
- Python 3.7+
- Node.js 14+
- PostgreSQL 12+
- Nginx
- Certbot (for SSL)

## Environment Variable Configuration

1. Create a `.env` file in the root directory of the backend:

```
PROJECT_NAME=StockSpikeReplicator
PROJECT_VERSION=1.0.0
ALLOWED_ORIGINS=https://yourdomain.com
DATABASE_URL=postgresql://user:password@localhost/dbname
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
SPREADSHEET_ID=your_google_sheets_id
MAIN_SHEET=Sheet1
SERVICE_ACCOUNT_FILE=/path/to/service_account.json
TRANSACTION_COST=0.001
SLIPPAGE=0.001
BACKTEST_START_DATE=2020-01-01
BACKTEST_END_DATE=2023-12-31
INITIAL_CAPITAL=10000
```

Replace the placeholders with your actual values.

2. For the frontend, create a `.env` file in the root of the frontend directory:

```
REACT_APP_API_URL=https://api.yourdomain.com
```

## Database Setup

1. Install PostgreSQL:
   ```
   sudo apt update
   sudo apt install postgresql postgresql-contrib
   ```

2. Create a new database and user:
   ```
   sudo -u postgres psql
   CREATE DATABASE stockspikereplicator;
   CREATE USER ssruser WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE stockspikereplicator TO ssruser;
   \q
   ```

3. Update the `DATABASE_URL` in your `.env` file with the new database information.

## Backend Deployment

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/stock-spike-replicator.git
   cd stock-spike-replicator
   ```

2. Set up a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run database migrations:
   ```
   alembic upgrade head
   ```

5. Install Gunicorn:
   ```
   pip install gunicorn
   ```

6. Create a systemd service file for the backend:
   ```
   sudo nano /etc/systemd/system/stockspikereplicator.service
   ```

   Add the following content:

   ```
   [Unit]
   Description=Stock Spike Replicator Backend
   After=network.target

   [Service]
   User=your_username
   Group=your_group
   WorkingDirectory=/path/to/stock-spike-replicator
   Environment="PATH=/path/to/stock-spike-replicator/venv/bin"
   ExecStart=/path/to/stock-spike-replicator/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.main:app

   [Install]
   WantedBy=multi-user.target
   ```

7. Start and enable the service:
   ```
   sudo systemctl start stockspikereplicator
   sudo systemctl enable stockspikereplicator
   ```

## Frontend Deployment

1. Navigate to the frontend directory:
   ```
   cd stock-spike-replicator-frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Build the production version:
   ```
   npm run build
   ```

4. Install a static file server:
   ```
   npm install -g serve
   ```

5. Create a systemd service file for the frontend:
   ```
   sudo nano /etc/systemd/system/stockspikereplicator-frontend.service
   ```

   Add the following content:

   ```
   [Unit]
   Description=Stock Spike Replicator Frontend
   After=network.target

   [Service]
   User=your_username
   Group=your_group
   WorkingDirectory=/path/to/stock-spike-replicator-frontend/build
   ExecStart=/usr/bin/serve -s . -l 3000

   [Install]
   WantedBy=multi-user.target
   ```

6. Start and enable the service:
   ```
   sudo systemctl start stockspikereplicator-frontend
   sudo systemctl enable stockspikereplicator-frontend
   ```

## Nginx Configuration

1. Install Nginx:
   ```
   sudo apt install nginx
   ```

2. Create a new Nginx configuration file:
   ```
   sudo nano /etc/nginx/sites-available/stockspikereplicator
   ```

   Add the following content:

   ```
   server {
       listen 80;
       server_name yourdomain.com www.yourdomain.com;

       location / {
           proxy_pass http://localhost:3000;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection 'upgrade';
           proxy_set_header Host $host;
           proxy_cache_bypass $http_upgrade;
       }

       location /api {
           proxy_pass http://localhost:8000;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection 'upgrade';
           proxy_set_header Host $host;
           proxy_cache_bypass $http_upgrade;
       }
   }
   ```

3. Enable the site:
   ```
   sudo ln -s /etc/nginx/sites-available/stockspikereplicator /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

## SSL Certificate Setup

1. Install Certbot:
   ```
   sudo apt install certbot python3-certbot-nginx
   ```

2. Obtain and install SSL certificate:
   ```
   sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
   ```

3. Follow the prompts to configure HTTPS.

## Monitoring and Logging

1. Set up logging for the backend:
   - Add logging configuration to your FastAPI app
   - Configure log rotation

2. Set up monitoring:
   - Install and configure Prometheus for metrics collection
   - Set up Grafana for visualization

3. Regular maintenance:
   - Set up automated backups for your database
   - Implement a strategy for log analysis and alerting

Remember to regularly update your application, dependencies, and server software to ensure security and performance.