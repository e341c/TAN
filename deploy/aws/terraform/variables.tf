variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Resource name prefix"
  type        = string
  default     = "tan-model"
}

variable "instance_type" {
  description = "EC2 instance type (onnx inference needs at least t3.small)"
  type        = string
  default     = "t3.small"
}

variable "admin_cidr" {
  description = "CIDR allowed for SSH (your IP), e.g. 203.0.113.10/32"
  type        = string
}

variable "ssh_public_key" {
  description = "OpenSSH public key for EC2 access"
  type        = string
}

variable "repo_url" {
  description = "Git clone URL for tan-model (HTTPS or git@)"
  type        = string
  default     = "https://github.com/YOUR_USER/tan-model.git"
}

variable "repo_branch" {
  description = "Branch to deploy"
  type        = string
  default     = "main"
}

variable "app_dir" {
  description = "Directory on EC2 for the application"
  type        = string
  default     = "/opt/tan-model"
}

variable "postgres_password" {
  description = "Postgres password written to deploy/.env on the instance"
  type        = string
  sensitive   = true
}
