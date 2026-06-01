output "public_ip" {
  description = "Elastic IP of the EC2 instance"
  value       = aws_eip.app.public_ip
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.app.id
}

output "ssh_command" {
  description = "SSH to the instance (replace key path)"
  value       = "ssh -i ~/.ssh/your-key.pem ubuntu@${aws_eip.app.public_ip}"
}

output "dashboard_url" {
  description = "Prediction dashboard URL"
  value       = "http://${aws_eip.app.public_ip}/"
}
