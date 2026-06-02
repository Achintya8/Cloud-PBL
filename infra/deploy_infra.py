"""
infra/deploy_infra.py
=====================
Flocci-style Boto3 infrastructure provisioning script for the
Areca Nut Price Prediction System.

Provisions (idempotently):
  1. VPC + Subnets + Internet Gateway + Route Tables
  2. Security Groups (EC2, RDS, Lambda)
  3. RDS PostgreSQL instance
  4. EC2 instance (frontend + Grafana host)
  5. IAM Role + Policy for Lambda execution
  6. Lambda function (FastAPI via Mangum)
  7. API Gateway HTTP API → Lambda integration
  8. EventBridge rules for ETL + ML scheduled jobs
  9. CloudWatch Log Groups

Design principles:
  - Every resource check is idempotent (describe → create if not exists)
  - Tagged consistently for cost tracking
  - All secrets read from environment variables (never hardcoded)
  - Flocci pattern: declarative resource specs + boto3 execution
"""

import base64
import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import botocore.exceptions

from config.logging_config import get_logger
from config.settings import aws as aws_cfg, db as db_cfg, app as app_cfg

logger = get_logger(__name__, log_file="/var/log/areca/deploy_infra.log")


# ---------------------------------------------------------------------------
# Flocci: Declarative Resource Specification
# ---------------------------------------------------------------------------

RESOURCE_TAGS = [
    {"Key": "Project",     "Value": "areca-price-system"},
    {"Key": "ManagedBy",   "Value": "flocci-boto3"},
    {"Key": "Environment", "Value": app_cfg.environment},
    {"Key": "Owner",       "Value": "MLOps-Team"},
]

VPC_CIDR          = "10.42.0.0/16"
PUBLIC_SUBNET_CIDR  = "10.42.1.0/24"
PRIVATE_SUBNET_CIDR = "10.42.2.0/24"
PRIVATE_SUBNET2_CIDR = "10.42.3.0/24"  # Second AZ for RDS Multi-AZ


# ---------------------------------------------------------------------------
# Flocci Infrastructure Client
# ---------------------------------------------------------------------------

class FlocciProvisioner:
    """
    Flocci-style declarative AWS provisioner using boto3.

    Each provision_* method is idempotent: it describes the resource first
    and only creates it if absent, then returns the resource ID.
    """

    def __init__(self, region: str = aws_cfg.region):
        self.region = region
        self.ec2     = boto3.client("ec2",      region_name=region)
        self.rds     = boto3.client("rds",      region_name=region)
        self.lmb     = boto3.client("lambda",   region_name=region)
        self.iam     = boto3.client("iam",       region_name=region)
        self.agw     = boto3.client("apigatewayv2", region_name=region)
        self.events  = boto3.client("events",   region_name=region)
        self.logs    = boto3.client("logs",     region_name=region)
        self.ssm     = boto3.client("ssm",      region_name=region)

        # State cache (populated as resources are created)
        self.state: Dict[str, str] = {}

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _tag_spec(self, resource_type: str, name: str) -> List[Dict]:
        return [{
            "ResourceType": resource_type,
            "Tags": RESOURCE_TAGS + [{"Key": "Name", "Value": name}],
        }]

    def _tag_resource(self, resource_id: str, name: str) -> None:
        self.ec2.create_tags(
            Resources=[resource_id],
            Tags=RESOURCE_TAGS + [{"Key": "Name", "Value": name}],
        )

    # -----------------------------------------------------------------------
    # VPC
    # -----------------------------------------------------------------------

    def provision_vpc(self) -> str:
        """Create or retrieve VPC."""
        # Check existing
        resp = self.ec2.describe_vpcs(Filters=[
            {"Name": "tag:Project", "Values": ["areca-price-system"]},
            {"Name": "tag:Name",    "Values": ["areca-vpc"]},
        ])
        if resp["Vpcs"]:
            vpc_id = resp["Vpcs"][0]["VpcId"]
            logger.info("VPC already exists", vpc_id=vpc_id)
            return vpc_id

        resp = self.ec2.create_vpc(
            CidrBlock=VPC_CIDR,
            TagSpecifications=self._tag_spec("vpc", "areca-vpc"),
        )
        vpc_id = resp["Vpc"]["VpcId"]

        # Enable DNS hostnames (required for RDS)
        self.ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsHostnames={"Value": True})
        self.ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsSupport={"Value": True})

        logger.info("VPC created", vpc_id=vpc_id)
        return vpc_id

    def provision_internet_gateway(self, vpc_id: str) -> str:
        """Create or retrieve Internet Gateway and attach to VPC."""
        resp = self.ec2.describe_internet_gateways(Filters=[
            {"Name": "tag:Project", "Values": ["areca-price-system"]},
        ])
        if resp["InternetGateways"]:
            igw_id = resp["InternetGateways"][0]["InternetGatewayId"]
            logger.info("IGW already exists", igw_id=igw_id)
            return igw_id

        resp = self.ec2.create_internet_gateway(
            TagSpecifications=self._tag_spec("internet-gateway", "areca-igw")
        )
        igw_id = resp["InternetGateway"]["InternetGatewayId"]
        self.ec2.attach_internet_gateway(InternetGatewayId=igw_id, VpcId=vpc_id)
        logger.info("IGW created and attached", igw_id=igw_id)
        return igw_id

    def provision_subnet(
        self,
        vpc_id: str,
        cidr: str,
        az: str,
        name: str,
        public: bool = True,
    ) -> str:
        """Create or retrieve a subnet."""
        resp = self.ec2.describe_subnets(Filters=[
            {"Name": "vpc-id",          "Values": [vpc_id]},
            {"Name": "cidr-block",      "Values": [cidr]},
        ])
        if resp["Subnets"]:
            subnet_id = resp["Subnets"][0]["SubnetId"]
            logger.info("Subnet already exists", subnet_id=subnet_id, cidr=cidr)
            return subnet_id

        resp = self.ec2.create_subnet(
            VpcId=vpc_id,
            CidrBlock=cidr,
            AvailabilityZone=az,
            TagSpecifications=self._tag_spec("subnet", name),
        )
        subnet_id = resp["Subnet"]["SubnetId"]

        if public:
            self.ec2.modify_subnet_attribute(
                SubnetId=subnet_id,
                MapPublicIpOnLaunch={"Value": True},
            )

        logger.info("Subnet created", subnet_id=subnet_id, cidr=cidr, public=public)
        return subnet_id

    def provision_route_table(
        self, vpc_id: str, subnet_id: str, igw_id: str
    ) -> str:
        """Create public route table with internet route."""
        resp = self.ec2.describe_route_tables(Filters=[
            {"Name": "vpc-id",       "Values": [vpc_id]},
            {"Name": "tag:Name",     "Values": ["areca-public-rt"]},
        ])
        if resp["RouteTables"]:
            rt_id = resp["RouteTables"][0]["RouteTableId"]
            logger.info("Route table already exists", rt_id=rt_id)
            return rt_id

        resp = self.ec2.create_route_table(
            VpcId=vpc_id,
            TagSpecifications=self._tag_spec("route-table", "areca-public-rt"),
        )
        rt_id = resp["RouteTable"]["RouteTableId"]

        self.ec2.create_route(
            RouteTableId=rt_id,
            DestinationCidrBlock="0.0.0.0/0",
            GatewayId=igw_id,
        )
        self.ec2.associate_route_table(RouteTableId=rt_id, SubnetId=subnet_id)
        logger.info("Route table created", rt_id=rt_id)
        return rt_id

    # -----------------------------------------------------------------------
    # Security Groups
    # -----------------------------------------------------------------------

    def provision_security_group(
        self,
        vpc_id: str,
        name: str,
        description: str,
        ingress_rules: List[Dict],
    ) -> str:
        """Create or retrieve a security group."""
        resp = self.ec2.describe_security_groups(Filters=[
            {"Name": "vpc-id",     "Values": [vpc_id]},
            {"Name": "group-name", "Values": [name]},
        ])
        if resp["SecurityGroups"]:
            sg_id = resp["SecurityGroups"][0]["GroupId"]
            logger.info("Security group exists", name=name, sg_id=sg_id)
            return sg_id

        resp = self.ec2.create_security_group(
            GroupName=name,
            Description=description,
            VpcId=vpc_id,
            TagSpecifications=self._tag_spec("security-group", name),
        )
        sg_id = resp["GroupId"]

        if ingress_rules:
            self.ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=ingress_rules,
            )

        # Allow all outbound (default)
        logger.info("Security group created", name=name, sg_id=sg_id)
        return sg_id

    # -----------------------------------------------------------------------
    # RDS PostgreSQL
    # -----------------------------------------------------------------------

    def provision_rds_subnet_group(
        self, private_subnet_ids: List[str]
    ) -> str:
        """Create RDS DB subnet group."""
        group_name = "areca-rds-subnet-group"
        try:
            self.rds.describe_db_subnet_groups(DBSubnetGroupName=group_name)
            logger.info("RDS subnet group exists")
            return group_name
        except botocore.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] != "DBSubnetGroupNotFoundFault":
                raise

        self.rds.create_db_subnet_group(
            DBSubnetGroupName=group_name,
            DBSubnetGroupDescription="Areca Price System RDS Subnet Group",
            SubnetIds=private_subnet_ids,
            Tags=[{"Key": k, "Value": v} for t in RESOURCE_TAGS for k, v in [t.values()]],
        )
        logger.info("RDS subnet group created")
        return group_name

    def provision_rds_instance(self, sg_id: str, subnet_group: str) -> str:
        """Create or retrieve the RDS PostgreSQL instance."""
        identifier = aws_cfg.rds_identifier

        try:
            resp = self.rds.describe_db_instances(DBInstanceIdentifier=identifier)
            endpoint = resp["DBInstances"][0]["Endpoint"]["Address"]
            status   = resp["DBInstances"][0]["DBInstanceStatus"]
            logger.info("RDS instance exists", identifier=identifier, status=status)
            return endpoint
        except botocore.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] != "DBInstanceNotFound":
                raise

        logger.info("Creating RDS PostgreSQL instance", identifier=identifier)
        self.rds.create_db_instance(
            DBInstanceIdentifier=identifier,
            DBInstanceClass=aws_cfg.rds_instance_class,
            Engine="postgres",
            EngineVersion=aws_cfg.rds_engine_version,
            MasterUsername=db_cfg.user,
            MasterUserPassword=db_cfg.password,
            DBName=db_cfg.name,
            AllocatedStorage=aws_cfg.rds_allocated_storage,
            StorageType="gp3",
            StorageEncrypted=True,
            MultiAZ=aws_cfg.rds_multi_az,
            PubliclyAccessible=False,
            VpcSecurityGroupIds=[sg_id],
            DBSubnetGroupName=subnet_group,
            BackupRetentionPeriod=7,
            DeletionProtection=True,
            EnablePerformanceInsights=True,
            Tags=RESOURCE_TAGS,
        )

        # Wait for instance to be available
        logger.info("Waiting for RDS instance to become available (this may take ~10 min)")
        waiter = self.rds.get_waiter("db_instance_available")
        waiter.wait(
            DBInstanceIdentifier=identifier,
            WaiterConfig={"Delay": 30, "MaxAttempts": 40},
        )

        resp = self.rds.describe_db_instances(DBInstanceIdentifier=identifier)
        endpoint = resp["DBInstances"][0]["Endpoint"]["Address"]
        logger.info("RDS instance ready", endpoint=endpoint)
        return endpoint

    # -----------------------------------------------------------------------
    # EC2 (Frontend / Grafana host)
    # -----------------------------------------------------------------------

    def provision_ec2_instance(
        self, subnet_id: str, sg_id: str, rds_endpoint: str
    ) -> Tuple[str, str]:
        """
        Launch or retrieve the EC2 instance running:
          - Nginx (reverse proxy)
          - Python/Streamlit frontend
          - Grafana (port 3000)
        """
        if aws_cfg.ec2_instance_id:
            # Check if the specified instance is still running
            try:
                resp = self.ec2.describe_instances(InstanceIds=[aws_cfg.ec2_instance_id])
                state = resp["Reservations"][0]["Instances"][0]["State"]["Name"]
                public_ip = resp["Reservations"][0]["Instances"][0].get("PublicIpAddress", "")
                if state in ("running", "stopped"):
                    logger.info("Using existing EC2 instance", id=aws_cfg.ec2_instance_id, state=state)
                    return aws_cfg.ec2_instance_id, public_ip
            except Exception:
                pass

        # Check for existing tagged instance
        resp = self.ec2.describe_instances(Filters=[
            {"Name": "tag:Project", "Values": ["areca-price-system"]},
            {"Name": "tag:Name",    "Values": ["areca-frontend"]},
            {"Name": "instance-state-name", "Values": ["running", "stopped", "pending"]},
        ])
        if resp["Reservations"]:
            inst = resp["Reservations"][0]["Instances"][0]
            inst_id   = inst["InstanceId"]
            public_ip = inst.get("PublicIpAddress", "")
            logger.info("EC2 instance already exists", id=inst_id)
            return inst_id, public_ip

        user_data = self._build_user_data_script(rds_endpoint)
        user_data_b64 = base64.b64encode(user_data.encode()).decode()

        resp = self.ec2.run_instances(
            ImageId=aws_cfg.ec2_ami_id,
            InstanceType=aws_cfg.ec2_instance_type,
            KeyName=aws_cfg.ec2_key_pair,
            MinCount=1,
            MaxCount=1,
            SubnetId=subnet_id,
            SecurityGroupIds=[sg_id],
            UserData=user_data_b64,
            BlockDeviceMappings=[{
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize": 30,
                    "VolumeType": "gp3",
                    "Encrypted": True,
                    "DeleteOnTermination": True,
                },
            }],
            IamInstanceProfile={"Name": "areca-ec2-grafana-role"},
            TagSpecifications=self._tag_spec("instance", "areca-frontend"),
            MetadataOptions={
                "HttpTokens": "required",  # IMDSv2 enforcement
                "HttpEndpoint": "enabled",
            },
        )

        instance_id = resp["Instances"][0]["InstanceId"]
        logger.info("EC2 instance launched", id=instance_id)

        # Wait for running state
        waiter = self.ec2.get_waiter("instance_running")
        waiter.wait(InstanceIds=[instance_id])

        resp = self.ec2.describe_instances(InstanceIds=[instance_id])
        public_ip = resp["Reservations"][0]["Instances"][0].get("PublicIpAddress", "")
        logger.info("EC2 instance running", id=instance_id, public_ip=public_ip)
        return instance_id, public_ip

    @staticmethod
    def _build_user_data_script(rds_endpoint: str) -> str:
        """Build cloud-init user data script for EC2 bootstrap."""
        return f"""#!/bin/bash
set -euxo pipefail

# --- System Update ---
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get upgrade -y

# --- Install dependencies ---
apt-get install -y \
    python3.11 python3.11-venv \
    nginx postgresql-client \
    wget curl git unzip \
    apt-transport-https software-properties-common

# --- Install Grafana ---
wget -q -O /usr/share/keyrings/grafana.key https://apt.grafana.com/gpg.key
echo "deb [signed-by=/usr/share/keyrings/grafana.key] https://apt.grafana.com stable main" \
    > /etc/apt/sources.list.d/grafana.list
apt-get update -y && apt-get install -y grafana

# --- Configure Grafana ---
systemctl daemon-reload
systemctl enable grafana-server
systemctl start grafana-server

# --- Setup application directories ---
mkdir -p /opt/areca/{{models,logs}}
mkdir -p /var/log/areca

# --- Clone/Deploy application code ---
# In production, replace with your git repo or S3 artifact download
# git clone https://your-repo.git /opt/areca/app
# For now, create a placeholder structure
mkdir -p /opt/areca/app

# --- Install uv (fast Python package manager) ---
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# --- Python virtual environment ---
uv venv /opt/areca/venv --python python3.11
source /opt/areca/venv/bin/activate
uv pip install \
    fastapi uvicorn gunicorn mangum \
    psycopg2-binary sqlalchemy \
    pandas numpy lightgbm scikit-learn \
    requests beautifulsoup4 \
    streamlit plotly

# --- Set environment variables ---
cat >> /etc/environment << 'EOF'
RDS_HOST={rds_endpoint}
RDS_DB_NAME={db_cfg.name}
RDS_USER={db_cfg.user}
APP_ENV=production
EOF

# --- Configure Nginx ---
cat > /etc/nginx/sites-available/areca << 'EOF'
server {{
    listen 80;
    server_name _;

    location / {{
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }}

    location /grafana/ {{
        proxy_pass http://localhost:3000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }}

    location /api/ {{
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }}
}}
EOF

ln -sf /etc/nginx/sites-available/areca /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx
systemctl enable nginx

# --- Setup cron for ETL & ML ---
(crontab -l 2>/dev/null; echo "0 6 * * * /opt/areca/venv/bin/python /opt/areca/app/etl/etl_pipeline.py >> /var/log/areca/etl.log 2>&1") | crontab -
(crontab -l 2>/dev/null; echo "0 8 * * * /opt/areca/venv/bin/python /opt/areca/app/ml_engine/train_predict.py >> /var/log/areca/ml.log 2>&1") | crontab -

echo "Bootstrap complete at $(date)" >> /var/log/areca/bootstrap.log
"""

    # -----------------------------------------------------------------------
    # IAM Role for Lambda
    # -----------------------------------------------------------------------

    def provision_lambda_iam_role(self) -> str:
        """Create IAM execution role for Lambda with RDS access."""
        role_name = "areca-lambda-execution-role"

        try:
            resp = self.iam.get_role(RoleName=role_name)
            role_arn = resp["Role"]["Arn"]
            logger.info("IAM role exists", role_name=role_name)
            return role_arn
        except botocore.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] != "NoSuchEntity":
                raise

        trust_policy = json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }]
        })

        resp = self.iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=trust_policy,
            Description="Execution role for Areca Price System Lambda functions",
            Tags=RESOURCE_TAGS,
        )
        role_arn = resp["Role"]["Arn"]

        # Attach managed policies
        for policy_arn in [
            "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole",
            "arn:aws:iam::aws:policy/AWSLambdaBasicExecutionRole",
        ]:
            self.iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)

        # Allow SSM parameter reads (for DB credentials)
        inline_policy = json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["ssm:GetParameter", "ssm:GetParameters"],
                    "Resource": f"arn:aws:ssm:{self.region}:{aws_cfg.account_id}:parameter/areca/*",
                },
                {
                    "Effect": "Allow",
                    "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                    "Resource": "*",
                },
            ]
        })
        self.iam.put_role_policy(
            RoleName=role_name,
            PolicyName="areca-lambda-inline",
            PolicyDocument=inline_policy,
        )

        logger.info("IAM role created", role_arn=role_arn)
        # IAM propagation delay
        time.sleep(10)
        return role_arn

    # -----------------------------------------------------------------------
    # Lambda Function
    # -----------------------------------------------------------------------

    def build_lambda_package(self, source_dir: str) -> bytes:
        """
        Create a Lambda deployment ZIP package from the source directory.
        Includes all .py files from backend/, config/, database/.
        """
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            source_path = Path(source_dir)
            for folder in ["backend", "config", "database"]:
                folder_path = source_path / folder
                if not folder_path.exists():
                    continue
                for py_file in folder_path.rglob("*.py"):
                    arcname = py_file.relative_to(source_path)
                    zf.write(py_file, arcname)

            # Create __init__.py for packages if missing
            for pkg in ["backend", "config", "database"]:
                init = source_path / pkg / "__init__.py"
                if not (source_path / pkg).exists():
                    continue
                if not init.exists():
                    zf.writestr(f"{pkg}/__init__.py", "")

        buffer.seek(0)
        package_bytes = buffer.read()
        size_kb = len(package_bytes) / 1024
        logger.info("Lambda package built", size_kb=round(size_kb, 1))
        return package_bytes

    def provision_lambda_function(
        self,
        role_arn: str,
        sg_ids: List[str],
        subnet_ids: List[str],
        source_dir: str,
        rds_host: str,
    ) -> str:
        """Create or update the FastAPI Lambda function."""
        func_name = aws_cfg.lambda_function_name

        zip_bytes = self.build_lambda_package(source_dir)
        zip_hash = hashlib.sha256(zip_bytes).hexdigest()[:8]

        env_vars = {
            "RDS_HOST":    rds_host,
            "RDS_DB_NAME": db_cfg.name,
            "RDS_USER":    db_cfg.user,
            "RDS_PASSWORD": db_cfg.password,
            "APP_ENV":     app_cfg.environment,
            "LOG_LEVEL":   app_cfg.log_level,
        }

        try:
            resp = self.lmb.get_function(FunctionName=func_name)
            logger.info("Lambda function exists — updating code", func=func_name)
            self.lmb.update_function_code(
                FunctionName=func_name,
                ZipFile=zip_bytes,
            )
            self.lmb.update_function_configuration(
                FunctionName=func_name,
                Environment={"Variables": env_vars},
                VpcConfig={"SubnetIds": subnet_ids, "SecurityGroupIds": sg_ids},
                Timeout=aws_cfg.lambda_timeout,
                MemorySize=aws_cfg.lambda_memory,
            )
            return resp["Configuration"]["FunctionArn"]

        except botocore.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] != "ResourceNotFoundException":
                raise

        logger.info("Creating Lambda function", func=func_name)
        resp = self.lmb.create_function(
            FunctionName=func_name,
            Runtime=aws_cfg.lambda_runtime,
            Role=role_arn,
            Handler="backend.lambda_function.lambda_handler",
            Code={"ZipFile": zip_bytes},
            Description="Areca Nut Price Prediction API (FastAPI + Mangum)",
            Timeout=aws_cfg.lambda_timeout,
            MemorySize=aws_cfg.lambda_memory,
            Environment={"Variables": env_vars},
            VpcConfig={"SubnetIds": subnet_ids, "SecurityGroupIds": sg_ids},
            Tags={t["Key"]: t["Value"] for t in RESOURCE_TAGS},
        )

        func_arn = resp["FunctionArn"]
        logger.info("Lambda function created", arn=func_arn)
        return func_arn

    # -----------------------------------------------------------------------
    # API Gateway (HTTP API v2)
    # -----------------------------------------------------------------------

    def provision_api_gateway(self, lambda_arn: str) -> str:
        """Create or retrieve API Gateway HTTP API with Lambda integration."""
        api_name = aws_cfg.api_gateway_name

        resp = self.agw.get_apis()
        for api in resp.get("Items", []):
            if api["Name"] == api_name:
                api_id  = api["ApiId"]
                api_url = api["ApiEndpoint"]
                logger.info("API Gateway exists", api_id=api_id, url=api_url)
                return api_url

        # Create API
        resp = self.agw.create_api(
            Name=api_name,
            ProtocolType="HTTP",
            CorsConfiguration={
                "AllowOrigins": app_cfg.cors_origins,
                "AllowMethods": ["GET", "POST", "OPTIONS"],
                "AllowHeaders": ["content-type", "authorization"],
                "MaxAge": 3600,
            },
            Tags={t["Key"]: t["Value"] for t in RESOURCE_TAGS},
        )
        api_id = resp["ApiId"]

        # Lambda integration
        integration_resp = self.agw.create_integration(
            ApiId=api_id,
            IntegrationType="AWS_PROXY",
            IntegrationSubtype="EventBridge-PutEvents",
            IntegrationUri=lambda_arn,
            PayloadFormatVersion="2.0",
            TimeoutInMillis=29000,
        )
        integration_id = integration_resp["IntegrationId"]

        # Catch-all route
        self.agw.create_route(
            ApiId=api_id,
            RouteKey="$default",
            Target=f"integrations/{integration_id}",
        )

        # Auto-deploy stage
        self.agw.create_stage(
            ApiId=api_id,
            StageName=aws_cfg.api_stage,
            AutoDeploy=True,
            DefaultRouteSettings={
                "DetailedMetricsEnabled": True,
                "ThrottlingBurstLimit": 100,
                "ThrottlingRateLimit": 50,
            },
        )

        # Grant API Gateway permission to invoke Lambda
        self.lmb.add_permission(
            FunctionName=aws_cfg.lambda_function_name,
            StatementId="api-gateway-invoke",
            Action="lambda:InvokeFunction",
            Principal="apigateway.amazonaws.com",
            SourceArn=f"arn:aws:execute-api:{self.region}:{aws_cfg.account_id}:{api_id}/*",
        )

        api_url = f"https://{api_id}.execute-api.{self.region}.amazonaws.com/{aws_cfg.api_stage}"
        logger.info("API Gateway created", api_id=api_id, url=api_url)
        return api_url

    # -----------------------------------------------------------------------
    # EventBridge Scheduled Rules (ETL + ML)
    # -----------------------------------------------------------------------

    def provision_eventbridge_rules(self, etl_lambda_arn: str, ml_lambda_arn: str) -> None:
        """
        Create EventBridge scheduled rules for automated ETL and ML jobs.

        - ETL:  Daily at 06:00 IST (00:30 UTC) — scrapes previous day's prices
        - ML:   Daily at 08:00 IST (02:30 UTC) — retrains and generates forecasts
        """
        rules = [
            {
                "name":       "areca-etl-daily",
                "schedule":   "cron(30 0 * * ? *)",
                "description":"Daily ETL — scrape areca nut prices from Agmarknet",
                "target_arn": etl_lambda_arn,
                "input":      json.dumps({"backfill_days": 0}),
            },
            {
                "name":       "areca-ml-daily",
                "schedule":   "cron(30 2 * * ? *)",
                "description":"Daily ML retraining and 7/30-day forecast generation",
                "target_arn": ml_lambda_arn,
                "input":      json.dumps({}),
            },
        ]

        for rule in rules:
            try:
                self.events.put_rule(
                    Name=rule["name"],
                    ScheduleExpression=rule["schedule"],
                    Description=rule["description"],
                    State="ENABLED",
                    Tags=RESOURCE_TAGS,
                )
                self.events.put_targets(
                    Rule=rule["name"],
                    Targets=[{
                        "Id":    rule["name"] + "-target",
                        "Arn":   rule["target_arn"],
                        "Input": rule["input"],
                    }],
                )
                # Grant EventBridge permission to invoke Lambda
                try:
                    self.lmb.add_permission(
                        FunctionName=rule["target_arn"].split(":")[-1],
                        StatementId=f"eventbridge-{rule['name']}",
                        Action="lambda:InvokeFunction",
                        Principal="events.amazonaws.com",
                        SourceArn=f"arn:aws:events:{self.region}:{aws_cfg.account_id}:rule/{rule['name']}",
                    )
                except botocore.exceptions.ClientError as exc:
                    if exc.response["Error"]["Code"] != "ResourceConflictException":
                        raise

                logger.info("EventBridge rule created/updated", name=rule["name"])
            except Exception as exc:
                logger.error("Failed to create EventBridge rule", name=rule["name"], error=str(exc))

    # -----------------------------------------------------------------------
    # CloudWatch Log Groups
    # -----------------------------------------------------------------------

    def provision_log_groups(self) -> None:
        """Create CloudWatch Log Groups for all Lambda functions."""
        log_groups = [
            f"/aws/lambda/{aws_cfg.lambda_function_name}",
            "/aws/lambda/areca-etl",
            "/aws/lambda/areca-ml",
            "/areca/frontend",
            "/areca/grafana",
        ]
        for group in log_groups:
            try:
                self.logs.create_log_group(
                    logGroupName=group,
                    tags={t["Key"]: t["Value"] for t in RESOURCE_TAGS},
                )
                self.logs.put_retention_policy(
                    logGroupName=group,
                    retentionInDays=30,
                )
                logger.info("Log group created", group=group)
            except botocore.exceptions.ClientError as exc:
                if exc.response["Error"]["Code"] != "ResourceAlreadyExistsException":
                    raise

    # -----------------------------------------------------------------------
    # SSM Parameter Store (DB credentials)
    # -----------------------------------------------------------------------

    def provision_ssm_parameters(self) -> None:
        """Store sensitive configuration in SSM Parameter Store."""
        params = [
            ("/areca/db/host",     db_cfg.host,     "String"),
            ("/areca/db/name",     db_cfg.name,     "String"),
            ("/areca/db/user",     db_cfg.user,     "String"),
            ("/areca/db/password", db_cfg.password, "SecureString"),
        ]
        for path, value, param_type in params:
            self.ssm.put_parameter(
                Name=path,
                Value=value,
                Type=param_type,
                Overwrite=True,
                Tags=RESOURCE_TAGS if param_type == "String" else [],
            )
        logger.info("SSM parameters stored")


# ---------------------------------------------------------------------------
# Main Provisioning Orchestrator
# ---------------------------------------------------------------------------

class InfrastructureOrchestrator:
    """
    Top-level orchestrator that provisions all system components in order.
    Each step is idempotent and logs its outcome.
    """

    def __init__(self, source_dir: str = "/opt/areca/app"):
        self.flocci = FlocciProvisioner()
        self.source_dir = source_dir

    def provision_all(self) -> Dict[str, Any]:
        """Run the full infrastructure provisioning workflow."""
        logger.info("Starting full infrastructure provisioning")
        output: Dict[str, Any] = {}

        # Step 1: VPC + Networking
        logger.info("Step 1/9: Provisioning VPC and networking")
        vpc_id = self.flocci.provision_vpc()
        igw_id = self.flocci.provision_internet_gateway(vpc_id)

        azs = boto3.client("ec2", region_name=aws_cfg.region).describe_availability_zones()
        az_names = [az["ZoneName"] for az in azs["AvailabilityZones"][:2]]

        public_subnet_id = self.flocci.provision_subnet(
            vpc_id, PUBLIC_SUBNET_CIDR, az_names[0], "areca-public-subnet", public=True
        )
        private_subnet_id = self.flocci.provision_subnet(
            vpc_id, PRIVATE_SUBNET_CIDR, az_names[0], "areca-private-subnet-1", public=False
        )
        private_subnet2_id = self.flocci.provision_subnet(
            vpc_id, PRIVATE_SUBNET2_CIDR, az_names[1], "areca-private-subnet-2", public=False
        )
        self.flocci.provision_route_table(vpc_id, public_subnet_id, igw_id)
        output.update({
            "vpc_id": vpc_id, "public_subnet_id": public_subnet_id,
            "private_subnet_id": private_subnet_id,
        })

        # Step 2: Security Groups
        logger.info("Step 2/9: Provisioning security groups")
        ec2_sg_id = self.flocci.provision_security_group(
            vpc_id,
            "areca-ec2-sg",
            "Security group for Areca frontend EC2 instance",
            ingress_rules=[
                {
                    "IpProtocol": "tcp", "FromPort": 80, "ToPort": 80,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "HTTP"}],
                },
                {
                    "IpProtocol": "tcp", "FromPort": 443, "ToPort": 443,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "HTTPS"}],
                },
                {
                    "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH (restrict in prod)"}],
                },
                {
                    "IpProtocol": "tcp", "FromPort": 3000, "ToPort": 3000,
                    "IpRanges": [{"CidrIp": "10.42.0.0/16", "Description": "Grafana internal"}],
                },
            ],
        )

        rds_sg_id = self.flocci.provision_security_group(
            vpc_id,
            "areca-rds-sg",
            "Security group for Areca RDS PostgreSQL",
            ingress_rules=[
                {
                    "IpProtocol": "tcp", "FromPort": 5432, "ToPort": 5432,
                    "UserIdGroupPairs": [
                        {"GroupId": ec2_sg_id, "Description": "EC2 frontend"},
                    ],
                },
            ],
        )
        output.update({"ec2_sg_id": ec2_sg_id, "rds_sg_id": rds_sg_id})

        # Step 3: RDS
        logger.info("Step 3/9: Provisioning RDS PostgreSQL")
        subnet_group = self.flocci.provision_rds_subnet_group(
            [private_subnet_id, private_subnet2_id]
        )
        rds_endpoint = self.flocci.provision_rds_instance(rds_sg_id, subnet_group)
        output["rds_endpoint"] = rds_endpoint

        # Step 4: EC2
        logger.info("Step 4/9: Provisioning EC2 instance")
        ec2_id, ec2_ip = self.flocci.provision_ec2_instance(
            public_subnet_id, ec2_sg_id, rds_endpoint
        )
        output.update({"ec2_instance_id": ec2_id, "ec2_public_ip": ec2_ip})

        # Step 5: IAM Role for Lambda
        logger.info("Step 5/9: Provisioning Lambda IAM role")
        role_arn = self.flocci.provision_lambda_iam_role()
        output["lambda_role_arn"] = role_arn

        # Step 6: Lambda (API)
        logger.info("Step 6/9: Deploying Lambda API function")
        lambda_sg_id = rds_sg_id  # Lambda needs DB access
        lambda_arn = self.flocci.provision_lambda_function(
            role_arn=role_arn,
            sg_ids=[lambda_sg_id],
            subnet_ids=[private_subnet_id],
            source_dir=self.source_dir,
            rds_host=rds_endpoint,
        )
        output["lambda_arn"] = lambda_arn

        # Step 7: API Gateway
        logger.info("Step 7/9: Provisioning API Gateway")
        api_url = self.flocci.provision_api_gateway(lambda_arn)
        output["api_gateway_url"] = api_url

        # Step 8: EventBridge Rules
        logger.info("Step 8/9: Provisioning EventBridge scheduled rules")
        self.flocci.provision_eventbridge_rules(lambda_arn, lambda_arn)

        # Step 9: CloudWatch + SSM
        logger.info("Step 9/9: Setting up CloudWatch logs and SSM parameters")
        self.flocci.provision_log_groups()
        self.flocci.provision_ssm_parameters()

        logger.info("Infrastructure provisioning complete", output=output)
        return output


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Areca Price System — Flocci Infrastructure Deploy")
    parser.add_argument(
        "--source-dir",
        default=str(Path(__file__).parent.parent),
        help="Root of the application source code",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned resources without creating them",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — Resources that would be provisioned:")
        print(json.dumps({
            "vpc": {"cidr": VPC_CIDR},
            "rds": {
                "identifier": aws_cfg.rds_identifier,
                "instance_class": aws_cfg.rds_instance_class,
                "engine": "postgres",
                "version": aws_cfg.rds_engine_version,
            },
            "ec2": {
                "instance_type": aws_cfg.ec2_instance_type,
                "ami": aws_cfg.ec2_ami_id,
            },
            "lambda": {
                "function_name": aws_cfg.lambda_function_name,
                "runtime": aws_cfg.lambda_runtime,
            },
            "api_gateway": aws_cfg.api_gateway_name,
        }, indent=2))
    else:
        orchestrator = InfrastructureOrchestrator(source_dir=args.source_dir)
        result = orchestrator.provision_all()
        print(json.dumps(result, indent=2, default=str))
