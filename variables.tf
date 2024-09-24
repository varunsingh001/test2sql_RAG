# Variables definition file (variables.tf)
variable "location" {
  description = "Azure location for resources"
  default     = "eastus"
}

variable "resource_group_name" {
  description = "Name of the resource group"
  default     = "MyOpenAI001"
}

variable "openai_account_name" {
  description = "Name for the Azure OpenAI account"
  default     = "my-openai-account"
}

variable "vnet_name" {
  description = "Name of the Virtual Network"
  default     = "vnet-openai"
}

variable "subnet_name" {
  description = "Name of the Subnet"
  default     = "snet-private-endpoint"
}

variable "private_endpoint_name" {
  description = "Name of the Private Endpoint"
  default     = "pe-openai"
}

variable "key_vault_name" {
  description = "The name of the Key Vault."
  default     = "key-vault-5662e2d7"
}

variable "app_object_id" {
  description = "The object ID of the application or service principal accessing the Key Vault."
}