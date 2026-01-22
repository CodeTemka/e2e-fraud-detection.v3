"""Utitlity to prepare Azure resources for the fraud detection system"""
"""!!! Run this script on PowerShell, Command Prompt, or Bash. Don't run it on Git Bash"""

import json
import shutil
import subprocess

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import ClientSecretCredential

from fraud_detection.config import get_settings
from fraud_detection.utils.logging import get_logger


def main():
    settings = get_settings()

    az_path = shutil.which("az")
    if az_path is None:
        raise RuntimeError("Azure CLI (az) not found on this system.")


    # Create a resource group
    def create_resource_group():
        resource_group_name = settings.resource_group
        location = settings.location

        subprocess.run([az_path, "login"], check=True)

        #Check if the resource group exists
        show_cmd = [az_path, "group", "show", "--name", resource_group_name]

        result = subprocess.run(show_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            create_cmd = [
                az_path, "group", "create", "--name", resource_group_name, "--location", location, "--output", "json",
            ]
            result = subprocess.run(create_cmd, capture_output=True, text=True, check=True)
        else:
            print(f"Resource group {resource_group_name} already exists. Skipping creation.")
        return True
    
    
    # Create a service principal
    def service_principal_create():
        if not create_resource_group():
            raise RuntimeError("Failed to create resource group.")
        
        subscription_id = settings.subscription_id
        resource_group = settings.resource_group
        sp_name = "fraud-detection-demo-sp"
        role = "Contributor"
        output_file = "sp_credentials.json"

        def sp_exists(sp_name):
            cmd = [az_path, "ad", "sp", "list", "--display-name", sp_name, "--query", "[].appId", "-o", "tsv",]
            result = subprocess.run(cmd, capture_output=True, text=True)
            app_id = result.stdout.strip()
            return bool(app_id)

        if sp_exists(sp_name):
            print(f"Service principal {sp_name} already exists. Skipping creation.")
            return True
        
        cmd = [
            az_path, "ad", "sp", "create-for-rbac", "--name", sp_name, "--role", role, "--scopes", f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Save SP credentials into JSON
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result.stdout)
        
        return True
    
    # Create workspace using service principal
    def create_workspace():
        if not service_principal_create():
            raise RuntimeError("Failed to create service principal.")
        
        sp_credentials_path = "sp_credentials.json"
        with open(sp_credentials_path, encoding="utf-8") as f:
            sp_data = json.load(f)
        
        # Support both new and legacy az service principal output keys.
        tenant_id = sp_data.get("tenant") or sp_data.get("tenantId")
        client_id = sp_data.get("appId") or sp_data.get("clientId")
        client_secret = sp_data.get("password") or sp_data.get("clientSecret")
        if not all((tenant_id, client_id, client_secret)):
            raise KeyError("Missing required service principal fields in sp_credentials.json")

        credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )
        subscription_id = settings.subscription_id
        resource_group = settings.resource_group
        workspace_name = settings.workspace_name
        location = settings.location

        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
        )

        ws = Workspace(
            name=workspace_name,
            location=location,
            display_name=workspace_name,
            description="workspace for e2e fraud detection ML system",
            )
        
        try:
            ml_client.workspaces.get(name=workspace_name)
        except Exception:
            ml_client.workspaces.begin_create(ws).result()
        
    
    create_workspace()



if __name__ == "__main__":
    main()
