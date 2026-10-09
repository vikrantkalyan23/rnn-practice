# Run locally with Docker

1. Install and start Docker Desktop (macOS/Windows), or Docker Engine with the Compose plugin (Linux).
2. Open a terminal in this project directory. Keep these trained artifacts in `models/`: `next_word_model.keras`, `tokenizer.json`, `config.json`, and `training_text.txt`. They are included in the backend image; training is not needed to start the app.
3. Build and start both containers:

   ```bash
   docker compose up -d --build
   docker compose ps
   ```

4. Open `http://localhost:3000`. Check the backend through the frontend proxy:

   ```bash
   curl http://localhost:3000/api/health
   curl http://localhost:3000/api/predict \
     -H 'Content-Type: application/json' \
     -d '{"text":"machine learning","num_words":5,"top_k":5,"temperature":1.0,"sample":false}'
   ```

5. View logs, rebuild after changes, or stop the app:

   ```bash
   docker compose logs --tail=100 -f
   docker compose up -d --build
   docker compose down
   ```

The backend runs on the private Compose network at `backend:8000`. Next.js forwards `/api/*` to it. Only the frontend port is published. Use `APP_PORT=8080 docker compose up -d` to use a different host port. The frontend API URL is set during its image build. Backend dependencies use `uv.lock`; frontend dependencies use `yarn.lock`. Allow time and disk space for TensorFlow installation during the first build.

# Hostinger: deployment steps only

1. Use a **Hostinger VPS**. On a new VPS, select the **Ubuntu 24.04 with Docker** template in hPanel. Hostinger's [Docker template instructions](https://www.hostinger.com/support/8306612-how-to-use-the-docker-vps-template-at-hostinger/) describe the preinstalled Docker setup. Do not reinstall the OS on an existing server with data; install Docker on its existing OS instead.
2. Connect from your computer, replacing `YOUR_VPS_IP` with the VPS address:

   ```bash
   ssh root@YOUR_VPS_IP
   docker --version
   docker compose version
   mkdir -p /opt/next-word-prediction
   exit
   ```

3. From this project's directory on your computer, upload the app and Docker files:

   ```bash
   tar -czf /tmp/next-word-prediction.tar.gz \
     --exclude=node_modules --exclude=.next --exclude='.env*' \
     Dockerfile compose.yaml .dockerignore pyproject.toml uv.lock app models frontend
   scp /tmp/next-word-prediction.tar.gz root@YOUR_VPS_IP:/opt/next-word-prediction/
   ```

4. Connect to the VPS and unpack the project:

   ```bash
   ssh root@YOUR_VPS_IP
   cd /opt/next-word-prediction
   tar -xzf next-word-prediction.tar.gz
   ```

5. Set the public HTTP port and start the app. Port 80 must be available:

   ```bash
   printf 'APP_PORT=80\n' > .env
   docker compose up -d --build
   docker compose ps
   ```

6. In the VPS firewall in hPanel, allow inbound TCP port **80**. Keep SSH port **22** allowed. If a server firewall is enabled, also allow port 80 there.
7. Verify the deployment:

   ```bash
   curl http://127.0.0.1/api/health
   docker compose logs --tail=100
   ```

8. Open `http://YOUR_VPS_IP` in your browser.
9. Optional: point your domain's DNS A record to the VPS IP and open `http://your-domain.example`. For HTTPS, put a TLS reverse proxy in front of the frontend, change `.env` to `APP_PORT=127.0.0.1:3000`, rebuild the Compose deployment, proxy the domain to `127.0.0.1:3000`, obtain a certificate, and allow inbound port **443**.
10. After uploading updated source or model files, redeploy:

    ```bash
    cd /opt/next-word-prediction
    docker compose up -d --build
    docker compose ps
    ```

11. To stop the app:

    ```bash
    cd /opt/next-word-prediction
    docker compose down
    ```
