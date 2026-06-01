# AWS EC2 deployment

Production target: **single EC2 instance** running the full stack via Docker Compose (postgres, api, web, scheduler).

## Option A: Terraform

```bash
cd deploy/aws/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit: admin_cidr, ssh_public_key, repo_url, postgres_password

terraform init
terraform plan
terraform apply
```

Outputs: `public_ip`, `dashboard_url`, `ssh_command`.

After apply (5–10 min for cloud-init):

1. SSH to the instance.
2. Upload predict CSV:

```bash
scp technical_analysis_trainer/data/predict/btcusdt_1h_predict.csv \
  ubuntu@PUBLIC_IP:/opt/tan-model/technical_analysis_trainer/data/predict/
```

3. Check stack: `curl http://PUBLIC_IP/health`
4. Configure GitHub Actions secrets (see [github-secrets.md](github-secrets.md)) for CI deploys.

## Option B: Manual bootstrap

On a fresh Ubuntu 22.04 instance:

```bash
export REPO_URL=https://github.com/YOUR_USER/tan-model.git
sudo bash deploy/aws/scripts/bootstrap-ec2.sh
```

Then edit `/opt/tan-model/deploy/.env` and start compose as in the root README.

## GitHub Actions deploy

Pushes to `main` build images and SSH to EC2 to `pull` + `up --no-build`. See `.github/workflows/deploy-ec2.yml`.

## Security notes

- Restrict SSH `admin_cidr` to your IP in Terraform.
- Change default Postgres password; do not commit `deploy/.env`.
- Add HTTPS later (Caddy, nginx + Certbot, or ALB) — initial setup is HTTP on port 80.
