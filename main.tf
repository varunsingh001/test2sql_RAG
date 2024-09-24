terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

# Configure the Microsoft Azure Provider
provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "this" {
  name     = "MyOpenAI001"
  location = "eastus2"
}

resource "azurerm_cognitive_account" "this" {
  name                = var.openai_account_name
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  kind                = "OpenAI"
  sku_name            = "S0"
  public_network_access_enabled = true
  custom_subdomain_name = "my-openai-service-1"
}

resource "azurerm_cognitive_deployment" "model" {
  name                 = "test2sql"
  cognitive_account_id = azurerm_cognitive_account.this.id
  model {
    format  = "OpenAI"
    name    = "gpt-35-turbo-16k"
  }
  sku {
    name = "Standard"
  }
}

resource "azurerm_cognitive_deployment" "embeddings" {
  name                 = "text-embedding-ada-002"
  cognitive_account_id = azurerm_cognitive_account.this.id
  model {
    format  = "OpenAI"
    name    = "text-embedding-ada-002"
    version = "2"
  }
  sku {
    name = "Standard"
  }
}

resource "azurerm_virtual_network" "this" {
  name                = "vnet01"
  address_space       = ["10.0.0.0/24"]
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
}

resource "azurerm_subnet" "this" {
  name                 = "snet01"
  resource_group_name  = azurerm_resource_group.this.name
  virtual_network_name = azurerm_virtual_network.this.name
  address_prefixes     = ["10.0.0.0/25"]
}

resource "azurerm_private_endpoint" "this" {
  name                = "pe-openai-we"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  subnet_id           = azurerm_subnet.this.id


  private_service_connection {
    name                           = "pe-openai-we"
    private_connection_resource_id = azurerm_cognitive_account.this.id
    subresource_names              = ["account"]
    is_manual_connection           = false
  }

    private_dns_zone_group {
      name                 = "default"
      private_dns_zone_ids = [azurerm_private_dns_zone.this.id]
  }
}

resource "azurerm_private_dns_zone" "this" {
  name                = "privatelink.openai.azure.com"
  resource_group_name = azurerm_resource_group.this.name
}

resource "azurerm_private_dns_zone_virtual_network_link" "this" {
  name                  = "this-vnet-link"
  resource_group_name   = azurerm_resource_group.this.name
  private_dns_zone_name = azurerm_private_dns_zone.this.name
  virtual_network_id    = azurerm_virtual_network.this.id
}

# Create a Key Vault to store the Cognitive Services account key

data "azurerm_client_config" "current" {}

resource "azurerm_key_vault" "kv" {
  name                = "${var.key_vault_name}"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"
}

# Add an access policy for the app (if needed)
resource "azurerm_key_vault_access_policy" "kv_policy" {
  key_vault_id = azurerm_key_vault.kv.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = var.app_object_id # The object ID of your app or service principal

  secret_permissions = ["Get"]
}

# Store the OpenAI key in Key Vault
resource "azurerm_key_vault_secret" "openai_secret" {
  name         = "OpenAI-Api-Key"
  value        = azurerm_cognitive_account.this.primary_access_key
  key_vault_id = azurerm_key_vault.kv.id
}


output "openai_endpoint" {
  value = azurerm_cognitive_account.this.endpoint
  description = "The endpoint of the OpenAI service."
}

output "key_vault_uri" {
  value = azurerm_key_vault.kv.vault_uri
  description = "The URI of the Key Vault."
}