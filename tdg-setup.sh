#!/bin/bash

# Technical Drawing Generator Setup Script
# This script sets up the complete technical drawing generator system

set -e  # Exit on any error

# Color codes for prettier output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print section header
print_header() {
    echo ""
    echo -e "${BLUE}==============================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}==============================================${NC}"
    echo ""
}

# Print success message
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
    echo ""
}

# Print warning message
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    echo ""
}

# Print error message
print_error() {
    echo -e "${RED}❌ $1${NC}"
    echo ""
}

# Print info message
print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
    echo ""
}

# Check if command should use sudo
use_sudo=false
check_sudo_needed() {
    if ! docker info > /dev/null 2>&1; then
        if sudo -n true 2>/dev/null; then
            use_sudo=true
            print_warning "Using sudo for Docker commands (current user lacks Docker permissions)"
        else
            print_error "ERROR: Current user cannot run Docker commands."
            echo "Please either:"
            echo "  1. Add your user to the docker group: 'sudo usermod -aG docker $USER' and log out/in"
            echo "  2. Run this script with sudo"
            exit 1
        fi
    fi
}

# Run docker command with sudo if needed
docker_cmd() {
    if $use_sudo; then
        sudo docker "$@"
    else
        docker "$@"
    fi
}

# Run docker-compose command with sudo if needed
compose_cmd() {
    if $use_sudo; then
        sudo docker compose "$@"
    else
        docker compose "$@"
    fi
}

# Check if Docker is installed
check_docker() {
    print_header "Checking Docker Installation"
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker before running this script."
        echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
        exit 1
    fi
    
    # Check Docker Compose v2 (part of Docker CLI)
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose V2 is not available. Please update Docker to a newer version."
        echo "Visit https://docs.docker.com/compose/install/ for installation instructions."
        exit 1
    fi
    
    # Check for Docker permissions
    check_sudo_needed
    
    print_success "Docker and Docker Compose are installed"
}

# Clean up existing containers and volumes if needed
cleanup_existing() {
    print_header "Cleaning up any existing resources"
    
    # Check if directory exists
    if [ -d "technical-drawing-generator" ]; then
        print_warning "Found existing technical-drawing-generator directory"
        
        # Check if docker-compose file exists
        if [ -f "technical-drawing-generator/docker-compose.yml" ]; then
            print_info "Stopping and removing existing containers..."
            cd technical-drawing-generator
            
            # Try to bring down any running containers
            compose_cmd down --volumes --remove-orphans || true
            
            # Force remove any stuck containers
            containers=$(docker_cmd ps -a --filter "name=technical-drawing-generator" -q)
            if [ ! -z "$containers" ]; then
                print_info "Forcibly removing stuck containers..."
                docker_cmd rm -f $containers || true
            fi
            
            # Remove any related volumes
            volumes=$(docker_cmd volume ls --filter "name=technical-drawing-generator" -q)
            if [ ! -z "$volumes" ]; then
                print_info "Removing related volumes..."
                docker_cmd volume rm $volumes || true
            fi
            
            cd ..
        fi
        
        # Ask if user wants to delete the existing directory
        read -p "$(echo -e ${YELLOW}Do you want to delete the existing technical-drawing-generator directory? \(y/n\):${NC} )" -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing directory..."
            rm -rf technical-drawing-generator
        else
            print_error "Please move or rename the existing directory before running this script."
            exit 1
        fi
    fi
    
    print_success "Cleanup completed"
}

# Get API key from user
setup_env_file() {
    print_header "Setting up Environment Variables"
    
    # Create .env file directory
    mkdir -p technical-drawing-generator
    cd technical-drawing-generator
    
    # Check if .env file exists
    if [ -f ".env" ]; then
        print_info "Found existing .env file."
        read -p "$(echo -e ${YELLOW}Do you want to replace the existing .env file? \(y/n\):${NC} )" -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Keeping existing .env file."
            cd ..
            return
        fi
    fi
    
    # Generate random API key or ask user to provide one
    read -p "Do you want to use an auto-generated secure API key? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Generate a random 32-character API key
        API_KEY=$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c 32)
        print_info "Generated API key: $API_KEY"
    else
        # Ask user to input their own API key
        read -p "Please enter your API key (leave blank to generate one): " API_KEY
        if [ -z "$API_KEY" ]; then
            API_KEY=$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c 32)
            print_info "Generated API key: $API_KEY"
        fi
    fi
    
    # Create .env file
    cat > .env << EOF
# Environment variables for Technical Drawing Generator
# Generated on $(date)

# API Key for authentication
REACT_APP_API_KEY=$API_KEY

# PostgreSQL settings
POSTGRES_PASSWORD=postgres
POSTGRES_USER=postgres
POSTGRES_DB=drawings

# Service settings
STORAGE_PATH=/app/storage
EOF
    
    print_success "Environment file (.env) created"
    cd ..
}

# Create project directory structure
create_directories() {
    print_header "Creating Directory Structure"
    
    cd technical-drawing-generator
    
    mkdir -p web/src web/public api processor
    mkdir -p data/db data/queue data/uploads data/outputs logs
    
    # Set proper permissions for Docker volume directories
    # This sets group permissions to ensure Docker can access them
    chmod -R 775 data logs
    
    print_success "Directory structure created with proper permissions"
    
    cd ..
}

# Create Docker Compose configuration
create_docker_compose() {
    print_header "Creating Docker Compose Configuration"
    
    cd technical-drawing-generator
    
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  web:
    build:
      context: ./web
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    depends_on:
      api:
        condition: service_healthy
    environment:
      - REACT_APP_API_URL=http://localhost:5000
      - REACT_APP_API_KEY=${REACT_APP_API_KEY}
    networks:
      - app-network
    volumes:
      - ./logs:/var/log/app
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:8080"]
      interval: 10s
      timeout: 5s
      retries: 3

  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    depends_on:
      db:
        condition: service_healthy
      queue:
        condition: service_healthy
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB}
      - QUEUE_HOST=queue
      - STORAGE_PATH=${STORAGE_PATH}
      - API_KEY=${REACT_APP_API_KEY}
    networks:
      - app-network
    volumes:
      - ./data/uploads:/app/storage/uploads
      - ./data/outputs:/app/storage/outputs
      - ./logs:/var/log/app
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 10s
      timeout: 5s
      retries: 3

  processor:
    build:
      context: ./processor
      dockerfile: Dockerfile
    depends_on:
      db:
        condition: service_healthy
      queue:
        condition: service_healthy
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB}
      - QUEUE_HOST=queue
      - STORAGE_PATH=${STORAGE_PATH}
    networks:
      - app-network
    volumes:
      - ./data/uploads:/app/storage/uploads
      - ./data/outputs:/app/storage/outputs
      - ./logs:/var/log/app
    restart: unless-stopped

  queue:
    image: rabbitmq:3-management
    ports:
      - "15672:15672"  # Management interface
    networks:
      - app-network
    volumes:
      - queue-data:/var/lib/rabbitmq
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "check_port_connectivity"]
      interval: 10s
      timeout: 5s
      retries: 3

  db:
    image: postgres:14
    environment:
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_DB=${POSTGRES_DB}
    ports:
      - "5432:5432"
    networks:
      - app-network
    volumes:
      - db-data:/var/lib/postgresql/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

networks:
  app-network:
    driver: bridge

volumes:
  queue-data:
  db-data:
EOF
    
    print_success "Docker Compose configuration created"
    
    cd ..
}

# Create API service files
create_api_service() {
    print_header "Creating API Service Files"
    
    cd technical-drawing-generator
    
    # Create Dockerfile
    cat > api/Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/storage/uploads /app/storage/outputs /var/log/app

# Ensure permissions are set correctly
RUN chmod -R 777 /app/storage /var/log/app

# Start application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
EOF
    
    # Create requirements.txt
    cat > api/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn==0.23.2
pydantic==2.4.2
pika==1.3.2
psycopg2-binary==2.9.9
python-multipart==0.0.6
python-dotenv==1.0.0
prometheus-client==0.17.1
EOF
    
    # Create main.py with added health endpoint
    cat > api/main.py << 'EOF'
import os
import uuid
import json
import logging
import shutil
from typing import List, Optional
from datetime import datetime

import pika
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/var/log/app/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api-service")

# Create FastAPI app
app = FastAPI(
    title="Technical Drawing Generator API",
    description="API for generating technical drawings from 3D model files",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
STORAGE_PATH = os.environ.get("STORAGE_PATH", "/app/storage")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@db:5432/drawings")
QUEUE_HOST = os.environ.get("QUEUE_HOST", "queue")
API_KEY = os.environ.get("API_KEY", "test-api-key")  # Set from environment

# Security
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify the API key."""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return api_key

# Database connection
def get_db_connection():
    """Create a connection to the PostgreSQL database."""
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    return conn

# RabbitMQ connection
def get_rabbitmq_channel():
    """Create a connection to RabbitMQ and return a channel."""
    # Add retry logic for RabbitMQ connection
    max_retries = 30
    retry_count = 0
    retry_delay = 2  # seconds
    last_error = None
    
    while retry_count < max_retries:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(
                host=QUEUE_HOST,
                connection_attempts=3,
                retry_delay=3
            ))
            channel = connection.channel()
            channel.queue_declare(queue='drawing_tasks', durable=True)
            return channel, connection
        except pika.exceptions.AMQPConnectionError as e:
            retry_count += 1
            last_error = e
            if retry_count < max_retries:
                logger.warning(f"Failed to connect to RabbitMQ (attempt {retry_count}/{max_retries}): {str(e)}. Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to RabbitMQ after {max_retries} attempts.")
                raise
    
    # If we get here, all retries failed
    raise last_error

# File validation
def validate_file(file: UploadFile) -> bool:
    """
    Validate uploaded file.
    
    Checks:
    - File extension is supported
    - File size is within limits
    - Basic integrity check
    """
    # Check file extension
    allowed_extensions = ['stl', 'f3d', 'step', 'iges', 'brep']
    file_ext = file.filename.split('.')[-1].lower()
    
    if file_ext not in allowed_extensions:
        logger.warning(f"Unsupported file extension: {file_ext}")
        return False
    
    # Check file size (50MB limit)
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset file pointer
    
    if file_size > 50 * 1024 * 1024:  # 50MB
        logger.warning(f"File too large: {file.filename}, {file_size} bytes")
        return False
    
    return True

# Health check endpoint for Docker healthcheck
@app.get("/health")
async def health_check():
    """Health check endpoint for service monitoring"""
    # Check database connection
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        db_status = "unhealthy"
    
    # Check RabbitMQ connection
    try:
        channel, connection = get_rabbitmq_channel()
        connection.close()
        queue_status = "healthy"
    except Exception as e:
        logger.error(f"RabbitMQ health check failed: {str(e)}")
        queue_status = "unhealthy"
    
    # Overall health status
    if db_status == "healthy" and queue_status == "healthy":
        status = "healthy"
    else:
        status = "unhealthy"
        
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "components": {
            "database": db_status,
            "queue": queue_status
        }
    }

# Routes
@app.post("/upload/", dependencies=[Depends(verify_api_key)])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a 3D model file for processing.
    
    Returns a job ID that can be used to check the status and retrieve the drawings.
    """
    if not validate_file(file):
        raise HTTPException(status_code=400, detail="Invalid file")
    
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    
    # Create directories
    upload_dir = os.path.join(STORAGE_PATH, "uploads", job_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save the uploaded file
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Send message to processing queue
    channel, connection = get_rabbitmq_channel()
    message = {
        "job_id": job_id,
        "file_path": file_path,
        "filename": file.filename,
        "timestamp": datetime.now().isoformat()
    }
    
    channel.basic_publish(
        exchange='',
        routing_key='drawing_tasks',
        body=json.dumps(message),
        properties=pika.BasicProperties(
            delivery_mode=2,  # make message persistent
        )
    )
    
    connection.close()
    
    # Save job to database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """
            INSERT INTO jobs (id, filename, status, created_at)
            VALUES (%s, %s, %s, %s)
            """,
            (job_id, file.filename, "pending", datetime.now())
        )
    finally:
        cursor.close()
        conn.close()
    
    return {"job_id": job_id, "status": "pending"}

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str = Path(...)):
    """
    Get the status of a job.
    
    Returns the job status and, if completed, information about the generated drawings.
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get job info
        cursor.execute(
            """
            SELECT id, filename, status, created_at, completed_at
            FROM jobs
            WHERE id = %s
            """,
            (job_id,)
        )
        
        job = cursor.fetchone()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # If job is completed, get drawing set info
        if job["status"] == "completed":
            cursor.execute(
                """
                SELECT id, original_file, original_format
                FROM drawing_sets
                WHERE job_id = %s
                """,
                (job_id,)
            )
            
            drawing_set = cursor.fetchone()
            
            if drawing_set:
                # Get drawings in the set
                cursor.execute(
                    """
                    SELECT id, path, type, size
                    FROM drawings
                    WHERE set_id = %s
                    """,
                    (drawing_set["id"],)
                )
                
                drawings = cursor.fetchall()
                drawing_set["drawings"] = drawings
                job["drawing_set"] = drawing_set
        
        return job
        
    finally:
        cursor.close()
        conn.close()

@app.get("/drawings/", dependencies=[Depends(verify_api_key)])
async def list_drawings(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, enum=["pending", "processing", "completed", "failed"])
):
    """
    List all drawing sets.
    
    Optional filters:
    - status: Filter by job status
    
    Pagination:
    - limit: Number of results to return (default: 10, max: 100)
    - offset: Offset for pagination (default: 0)
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Build query based on filters
        query = """
            SELECT js.id, js.filename, js.status, js.created_at, js.completed_at,
                ds.id as set_id, ds.original_format
            FROM jobs js
            LEFT JOIN drawing_sets ds ON js.id = ds.job_id
        """
        
        params = []
        
        # Add filters
        if status:
            query += " WHERE js.status = %s"
            params.append(status)
        
        # Add ordering and pagination
        query += " ORDER BY js.created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Get total count for pagination
        count_query = "SELECT COUNT(*) FROM jobs"
        if status:
            count_query += " WHERE status = %s"
            cursor.execute(count_query, [status])
        else:
            cursor.execute(count_query)
            
        total = cursor.fetchone()["count"]
        
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "results": results
        }
        
    finally:
        cursor.close()
        conn.close()

@app.get("/download/{drawing_id}")
async def download_drawing(drawing_id: int = Path(...)):
    """
    Get the download URL for a specific drawing.
    
    This endpoint returns a pre-signed URL or direct path
    that can be used to download the drawing file.
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cursor.execute(
            """
            SELECT d.path, d.type, ds.job_id
            FROM drawings d
            JOIN drawing_sets ds ON d.set_id = ds.id
            WHERE d.id = %s
            """,
            (drawing_id,)
        )
        
        drawing = cursor.fetchone()
        
        if not drawing:
            raise HTTPException(status_code=404, detail="Drawing not found")
        
        # In a real-world scenario, you'd generate a pre-signed URL here
        # For simplicity, we'll just return the path
        download_url = f"/api/files/{drawing['job_id']}/{drawing['path']}"
        
        return {
            "drawing_id": drawing_id,
            "file_type": drawing["type"],
            "download_url": download_url
        }
        
    finally:
        cursor.close()
        conn.close()

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the database and storage directory."""
    # Ensure storage directories exist
    os.makedirs(os.path.join(STORAGE_PATH, "uploads"), exist_ok=True)
    os.makedirs(os.path.join(STORAGE_PATH, "outputs"), exist_ok=True)
    
    # Initialize database tables
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Create jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP
            )
        """)
        
        # Create drawing_sets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drawing_sets (
                id SERIAL PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES jobs(id),
                original_file TEXT NOT NULL,
                original_format TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
        
        # Create drawings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drawings (
                id SERIAL PRIMARY KEY,
                set_id INTEGER NOT NULL REFERENCES drawing_sets(id),
                path TEXT NOT NULL,
                type TEXT NOT NULL,
                size INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
        
        conn.commit()
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.exception(f"Error initializing database: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
EOF
    
    print_success "API service files created"
    
    cd ..
}

# Create Processor service files
create_processor_service() {
    print_header "Creating Processor Service Files"
    
    cd technical-drawing-generator
    
    # Create Dockerfile
    cat > processor/Dockerfile << 'EOF'
FROM ubuntu:22.04

WORKDIR /app

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install FreeCAD and other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    freecad \
    python3-pip \
    python3-pika \
    python3-psycopg2 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/storage/uploads /app/storage/outputs /var/log/app
RUN chmod -R 777 /app/storage /var/log/app

CMD ["python3", "processor.py"]
EOF
    
    # Create requirements.txt
    cat > processor/requirements.txt << 'EOF'
pika==1.3.2
psycopg2-binary==2.9.9
python-dotenv==1.0.0
EOF
    
    # Create processor.py with improved error handling
    cat > processor/processor.py << 'EOF'
import os
import uuid
import json
import logging
import time
from typing import Dict, List, Tuple, Optional
import subprocess
import tempfile
import psycopg2
import pika

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/var/log/app/processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("cad-processor")

# Database connection
def get_db_connection():
    """Create a connection to the PostgreSQL database."""
    max_retries = 30
    retry_count = 0
    retry_delay = 2  # seconds
    
    while retry_count < max_retries:
        try:
            conn = psycopg2.connect(
                os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@db:5432/drawings")
            )
            return conn
        except psycopg2.OperationalError as e:
            retry_count += 1
            if retry_count < max_retries:
                logger.warning(f"Failed to connect to PostgreSQL (attempt {retry_count}/{max_retries}): {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to PostgreSQL after {max_retries} attempts.")
                raise
    
    # Should never reach here, but just in case
    raise Exception("Failed to connect to PostgreSQL database")

def update_job_status(job_id: str, status: str, conn=None):
    """Update the status of a job in the database."""
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True
    
    cursor = conn.cursor()
    
    try:
        if status == "completed":
            cursor.execute(
                """
                UPDATE jobs
                SET status = %s, completed_at = NOW()
                WHERE id = %s
                """,
                (status, job_id)
            )
        else:
            cursor.execute(
                """
                UPDATE jobs
                SET status = %s
                WHERE id = %s
                """,
                (status, job_id)
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.exception(f"Failed to update job status: {str(e)}")
        raise
    finally:
        cursor.close()
        if close_conn:
            conn.close()

class CADProcessor:
    """
    Processes CAD files to generate technical drawings using FreeCAD's Python API
    """
    
    def __init__(self, storage_path: str):
        """Initialize the processor with storage path for files."""
        self.storage_path = storage_path
        self.supported_formats = ['stl', 'f3d', 'step', 'iges', 'brep']
        
    def validate_file(self, file_path: str) -> bool:
        """Validate file format and integrity."""
        file_ext = os.path.splitext(file_path)[1][1:].lower()
        
        if file_ext not in self.supported_formats:
            logger.error(f"Unsupported file format: {file_ext}")
            return False
            
        # Basic file integrity check
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logger.error(f"Empty file: {file_path}")
                return False
                
            # For STL files, do basic validation
            if file_ext == 'stl':
                with open(file_path, 'rb') as f:
                    header = f.read(5)
                    # Check if binary STL (starts with solid but actually binary)
                    if header == b'solid' and file_size > 84:
                        f.seek(80)  # Skip header
                        try:
                            triangle_count = int.from_bytes(f.read(4), byteorder='little')
                            expected_size = 84 + (triangle_count * 50)
                            if abs(file_size - expected_size) > 100:  # Allow small difference
                                logger.warning(f"STL file size mismatch: {file_path}")
                                return False
                        except:
                            logger.error(f"Failed to parse STL triangle count: {file_path}")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"File validation error: {str(e)}")
            return False
    
    def process_file(self, file_path: str, job_id: str) -> Dict:
        """
        Process the CAD file to generate technical drawings
        Returns metadata about the generated drawings
        """
        if not self.validate_file(file_path):
            return {"status": "error", "message": "Invalid file"}
            
        file_ext = os.path.splitext(file_path)[1][1:].lower()
        base_name = os.path.basename(file_path)
        output_dir = os.path.join(self.storage_path, "outputs", job_id)
        os.makedirs(output_dir, exist_ok=True)
        
        drawing_files = []
        
        try:
            # Use FreeCAD to generate technical drawings
            # This uses FreeCAD's Python API via a subprocess to avoid 
            # potential crashes affecting the main application
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
                script_file.write(self._get_freecad_script(file_path, output_dir, file_ext))
                script_path = script_file.name
                
            # Execute FreeCAD with our script
            result = subprocess.run(
                ["freecadcmd", script_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"FreeCAD processing error: {result.stderr}")
                return {"status": "error", "message": "Processing failed", "details": result.stderr}
                
            # Collect generated files
            for root, _, files in os.walk(output_dir):
                for filename in files:
                    if filename.endswith(('.svg', '.pdf', '.dxf')):
                        rel_path = os.path.join(os.path.relpath(root, self.storage_path), filename)
                        drawing_files.append({
                            "path": rel_path,
                            "type": os.path.splitext(filename)[1][1:],
                            "size": os.path.getsize(os.path.join(root, filename))
                        })
            
            # Generate metadata
            metadata = {
                "status": "success",
                "original_file": base_name,
                "original_format": file_ext,
                "drawing_count": len(drawing_files),
                "drawings": drawing_files,
                "job_id": job_id
            }
            
            # Write metadata to a file in the output directory
            with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
                
            return metadata
            
        except Exception as e:
            logger.exception(f"Processing error for {file_path}: {str(e)}")
            return {"status": "error", "message": str(e)}
        finally:
            # Clean up the temporary script file
            if 'script_path' in locals():
                os.unlink(script_path)
    
    def _get_freecad_script(self, file_path: str, output_dir: str, file_ext: str) -> str:
        """Generate a Python script for FreeCAD to execute."""
        # This script will be executed by FreeCAD to generate technical drawings
        script = f"""
import os
import sys
import FreeCAD
import Part
import Import
import TechDraw
import importSVG
import importDXF
import Draft

# Input and output paths
input_file = "{file_path}"
output_dir = "{output_dir}"

# Create a new document
doc = FreeCAD.newDocument("TechnicalDrawings")

# Import the 3D model based on file type
if "{file_ext}" == "stl":
    mesh = Mesh.Mesh(input_file)
    shape = Part.Shape()
    shape.makeShapeFromMesh(mesh.Topology, 0.1)
    obj = doc.addObject("Part::Feature", "Model")
    obj.Shape = shape
elif "{file_ext}" == "f3d":
    # For Fusion 360 files, we'll need a converter
    # This is a simplified example - real implementation would use
    # appropriate converter tools or libraries
    Import.open(input_file)
else:
    Import.open(input_file)

# Get all objects in the document
objects = [obj for obj in doc.Objects if hasattr(obj, "Shape")]

if not objects:
    sys.exit("No valid 3D objects found in the model")

# Create a basic page template
template = TechDraw.makePageTemplate("Template", doc)

# Create projections
# Front view
page1 = TechDraw.makePage(template, doc)
page1.Label = "Front_Top_Right"

# Add front view
frontView = TechDraw.makeViewPart(page1, doc.Objects[0])
frontView.Direction = FreeCAD.Vector(0, -1, 0)  # Front view
frontView.X = 100
frontView.Y = 100
frontView.Scale = 1.0

# Add top view
topView = TechDraw.makeViewPart(page1, doc.Objects[0])
topView.Direction = FreeCAD.Vector(0, 0, 1)  # Top view
topView.X = 100
topView.Y = 200
topView.Scale = 1.0

# Add right view
rightView = TechDraw.makeViewPart(page1, doc.Objects[0])
rightView.Direction = FreeCAD.Vector(1, 0, 0)  # Right view
rightView.X = 200
rightView.Y = 100
rightView.Scale = 1.0

# Add isometric view
page2 = TechDraw.makePage(template, doc)
page2.Label = "Isometric"

isoView = TechDraw.makeViewPart(page2, doc.Objects[0])
isoView.Direction = FreeCAD.Vector(1, 1, 1)  # Isometric
isoView.Scale = 1.0
isoView.X = 150
isoView.Y = 150

# Add dimensions
# Simplified - in a real app, you'd analyze the model to add appropriate dimensions
FreeCAD.ActiveDocument = doc

# Compute all views
doc.recompute()

# Export as SVG and DXF
for page in [page1, page2]:
    svg_path = os.path.join(output_dir, f"{{page.Label}}.svg")
    importSVG.export([page], svg_path)
    
    dxf_path = os.path.join(output_dir, f"{{page.Label}}.dxf")
    importDXF.export([page], dxf_path)

# Export 2D projection as PDF
pdf_path = os.path.join(output_dir, "TechnicalDrawing.pdf")
TechDraw.exportPageAsPdf(page1, pdf_path)

sys.exit(0)  # Success
"""
        return script

def save_to_database(metadata: Dict) -> int:
    """Save the drawing metadata to the database and return the record ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Insert the main record for the drawing set
        cursor.execute("""
            INSERT INTO drawing_sets (job_id, original_file, original_format, status)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (
            metadata["job_id"],
            metadata["original_file"],
            metadata["original_format"],
            metadata["status"]
        ))
        
        set_id = cursor.fetchone()[0]
        
        # Insert each drawing file
        for drawing in metadata["drawings"]:
            cursor.execute("""
                INSERT INTO drawings (set_id, path, type, size)
                VALUES (%s, %s, %s, %s)
            """, (
                set_id,
                drawing["path"],
                drawing["type"],
                drawing["size"]
            ))
        
        conn.commit()
        return set_id
        
    except Exception as e:
        conn.rollback()
        logger.exception(f"Database error: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def process_message(ch, method, properties, body):
    """Process a message from the RabbitMQ queue."""
    try:
        data = json.loads(body)
        job_id = data.get("job_id")
        file_path = data.get("file_path")
        
        logger.info(f"Processing job {job_id} for file {file_path}")
        
        # Update job status to processing
        update_job_status(job_id, "processing")
        
        # Process the file
        processor = CADProcessor(os.environ.get("STORAGE_PATH", "/app/storage"))
        metadata = processor.process_file(file_path, job_id)
        
        # Save to database if processing was successful
        if metadata["status"] == "success":
            set_id = save_to_database(metadata)
            logger.info(f"Saved drawing set {set_id} to database")
            update_job_status(job_id, "completed")
        else:
            logger.error(f"Processing failed for job {job_id}: {metadata.get('message')}")
            update_job_status(job_id, "failed")
            
        # Acknowledge the message
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except Exception as e:
        logger.exception(f"Error processing message: {str(e)}")
        try:
            # Update job status to failed
            update_job_status(job_id, "failed")
        except:
            pass
        # Negative acknowledgment to requeue the message
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)  # Don't requeue failed jobs

def main():
    """Main function to set up the RabbitMQ connection and start consuming messages."""
    queue_host = os.environ.get("QUEUE_HOST", "queue")
    
    # Add retry logic for RabbitMQ connection
    max_retries = 30
    retry_count = 0
    retry_delay = 5  # seconds
    
    while retry_count < max_retries:
        try:
            logger.info(f"Attempting to connect to RabbitMQ at {queue_host} (attempt {retry_count+1}/{max_retries})")
            # Connect to RabbitMQ
            connection = pika.BlockingConnection(pika.ConnectionParameters(
                host=queue_host,
                connection_attempts=3,
                retry_delay=3
            ))
            channel = connection.channel()
            
            # Declare the queue
            channel.queue_declare(queue='drawing_tasks', durable=True)
            
            # Set up the consumer
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue='drawing_tasks', on_message_callback=process_message)
            
            logger.info("CAD Processor started. Waiting for messages...")
            
            # Start consuming messages
            channel.start_consuming()
            break  # If we get here, connection succeeded
            
        except pika.exceptions.AMQPConnectionError as e:
            retry_count += 1
            if retry_count < max_retries:
                logger.warning(f"Failed to connect to RabbitMQ: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to RabbitMQ after {max_retries} attempts. Giving up.")
                raise
        except Exception as e:
            logger.exception(f"Unexpected error: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                logger.warning(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed after {max_retries} attempts. Giving up.")
                raise

if __name__ == "__main__":
    # Wait a bit before starting to ensure other services are ready
    logger.info("Waiting for other services to initialize...")
    time.sleep(10)
    main()
EOF
    
    print_success "Processor service files created"
    
    cd ..
}

# Create Web frontend files with improved UI
create_web_frontend() {
    print_header "Creating Web Frontend Files"
    
    cd technical-drawing-generator
    
    # Create Dockerfile
    cat > web/Dockerfile << 'EOF'
FROM node:16-alpine

WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY . .

# Build the application
RUN npm run build

# Install serve to run the application
RUN npm install -g serve

# Expose port
EXPOSE 8080

# Create directories for logs
RUN mkdir -p /var/log/app
RUN chmod -R 777 /var/log/app

# Start application
CMD ["serve", "-s", "build", "-l", "8080"]
EOF
    
    # Create package.json with updated dependencies
    cat > web/package.json << 'EOF'
{
  "name": "technical-drawing-generator",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "@testing-library/jest-dom": "^5.16.5",
    "@testing-library/react": "^13.4.0",
    "@testing-library/user-event": "^13.5.0",
    "axios": "^1.6.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-icons": "^4.11.0",
    "react-router-dom": "^6.18.0",
    "react-scripts": "5.0.1",
    "react-toastify": "^9.1.3",
    "web-vitals": "^2.1.4"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
EOF
    
    # Create public/index.html
    mkdir -p web/public
    cat > web/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#2563eb" />
    <meta
      name="description"
      content="Technical Drawing Generator - Create technical drawings from 3D models"
    />
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <title>Technical Drawing Generator</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
EOF
    
    # Create src/index.js
    mkdir -p web/src
    cat > web/src/index.js << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
    <ToastContainer position="top-right" autoClose={5000} />
  </React.StrictMode>
);
EOF
    
    # Create src/index.css with modern styling
    cat > web/src/index.css << 'EOF'
:root {
  --primary: #2563eb;
  --primary-hover: #1d4ed8;
  --secondary: #6b7280;
  --success: #10b981;
  --error: #ef4444;
  --warning: #f59e0b;
  --info: #3b82f6;
  --background: #f9fafb;
  --card: #ffffff;
  --border: #e5e7eb;
  --text: #1f2937;
  --text-secondary: #4b5563;
  --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
  --radius: 0.375rem;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  margin: 0;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background-color: var(--background);
  color: var(--text);
  line-height: 1.5;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}

a {
  color: var(--primary);
  text-decoration: none;
  transition: color 0.2s;
}

a:hover {
  color: var(--primary-hover);
  text-decoration: underline;
}

button {
  cursor: pointer;
  font-family: inherit;
}

/* Status Colors */
.status-pending {
  color: var(--warning);
  font-weight: 500;
}

.status-processing {
  color: var(--info);
  font-weight: 500;
}

.status-completed {
  color: var(--success);
  font-weight: 500;
}

.status-failed {
  color: var(--error);
  font-weight: 500;
}

/* Card Styles */
.card {
  background-color: var(--card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  overflow: hidden;
  transition: box-shadow 0.3s ease;
}

.card:hover {
  box-shadow: var(--shadow-md);
}

.card-header {
  padding: 1rem;
  border-bottom: 1px solid var(--border);
}

.card-body {
  padding: 1rem;
}

.card-footer {
  padding: 1rem;
  border-top: 1px solid var(--border);
}

/* Buttons */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.5rem 1rem;
  border-radius: var(--radius);
  font-weight: 500;
  transition: all 0.2s;
  border: none;
  cursor: pointer;
}

.btn-primary {
  background-color: var(--primary);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background-color: var(--primary-hover);
}

.btn-secondary {
  background-color: var(--secondary);
  color: white;
}

.btn-secondary:hover:not(:disabled) {
  opacity: 0.9;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* Form Elements */
.form-group {
  margin-bottom: 1rem;
}

.form-label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 500;
}

.form-control {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  font-family: inherit;
  font-size: 1rem;
  transition: border-color 0.2s;
}

.form-control:focus {
  outline: none;
  border-color: var(--primary);
}

/* Grid Layout */
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1rem;
}

@media (max-width: 768px) {
  .grid {
    grid-template-columns: 1fr;
  }
}

/* Utility Classes */
.text-center {
  text-align: center;
}

.mt-1 { margin-top: 0.25rem; }
.mt-2 { margin-top: 0.5rem; }
.mt-3 { margin-top: 0.75rem; }
.mt-4 { margin-top: 1rem; }
.mt-5 { margin-top: 1.25rem; }

.mb-1 { margin-bottom: 0.25rem; }
.mb-2 { margin-bottom: 0.5rem; }
.mb-3 { margin-bottom: 0.75rem; }
.mb-4 { margin-bottom: 1rem; }
.mb-5 { margin-bottom: 1.25rem; }

.flex {
  display: flex;
}

.flex-wrap {
  flex-wrap: wrap;
}

.justify-between {
  justify-content: space-between;
}

.items-center {
  align-items: center;
}

.gap-2 {
  gap: 0.5rem;
}

.gap-4 {
  gap: 1rem;
}
EOF
    
    # Create src/App.css with modern styling
    cat > web/src/App.css << 'EOF'
.app-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 1rem;
}

header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 0;
  margin-bottom: 2rem;
  border-bottom: 1px solid var(--border);
}

.logo {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.logo svg {
  color: var(--primary);
}

.logo h1 {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--text);
}

nav {
  display: flex;
  gap: 1.5rem;
}

nav a {
  text-decoration: none;
  color: var(--text-secondary);
  font-weight: 500;
  transition: color 0.2s;
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

nav a:hover, nav a.active {
  color: var(--primary);
  text-decoration: none;
}

.upload-container {
  max-width: 600px;
  margin: 0 auto;
  background: var(--card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}

.file-input-container {
  margin: 1.5rem 0;
}

.file-input-container input {
  display: none;
}

.file-input-label {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  border: 2px dashed var(--border);
  border-radius: var(--radius);
  cursor: pointer;
  transition: all 0.2s;
}

.file-input-label:hover {
  border-color: var(--primary);
  background-color: rgba(37, 99, 235, 0.05);
}

.file-input-label svg {
  color: var(--primary);
  margin-bottom: 1rem;
  font-size: 2rem;
}

.file-input-label.has-file {
  border-color: var(--success);
  background-color: rgba(16, 185, 129, 0.05);
}

.file-input-label.has-file svg {
  color: var(--success);
}

.upload-button {
  display: block;
  width: 100%;
  padding: 0.75rem;
  background: var(--primary);
  color: white;
  border: none;
  border-radius: var(--radius);
  font-weight: 500;
  cursor: pointer;
  transition: background 0.2s;
}

.upload-button:hover:not(:disabled) {
  background: var(--primary-hover);
}

.upload-button:disabled {
  background: var(--secondary);
  cursor: not-allowed;
}

.error-message {
  color: var(--error);
  margin-top: 1rem;
  padding: 0.75rem;
  background: rgba(239, 68, 68, 0.1);
  border-radius: var(--radius);
}

.success-message {
  color: var(--success);
  margin-top: 1rem;
  padding: 0.75rem;
  background: rgba(16, 185, 129, 0.1);
  border-radius: var(--radius);
}

.loading {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 2rem;
  color: var(--text-secondary);
}

.loading svg {
  animation: spin 1s linear infinite;
  margin-right: 0.5rem;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.job-card {
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  background: var(--card);
  overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s;
}

.job-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.job-card-header {
  padding: 1rem;
  background: rgba(37, 99, 235, 0.05);
  border-bottom: 1px solid var(--border);
}

.job-card-body {
  padding: 1rem;
}

.job-card-footer {
  padding: 1rem;
  background: var(--background);
  border-top: 1px solid var(--border);
  display: flex;
  justify-content: flex-end;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
}

table th, table td {
  padding: 0.75rem 1rem;
  text-align: left;
  border-bottom: 1px solid var(--border);
}

table th {
  background-color: rgba(37, 99, 235, 0.05);
  font-weight: 600;
}

table tr:hover td {
  background-color: rgba(0, 0, 0, 0.01);
}

.job-list-container {
  background: var(--card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 1rem;
}

.filters {
  display: flex;
  gap: 1rem;
  margin-bottom: 1rem;
  padding: 1rem;
  background: var(--background);
  border-radius: var(--radius);
}

.filters select {
  padding: 0.5rem;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--card);
}

.pagination {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
}

.pagination button {
  padding: 0.5rem 1rem;
  background: var(--primary);
  color: white;
  border: none;
  border-radius: var(--radius);
  font-weight: 500;
  cursor: pointer;
}

.pagination button:disabled {
  background: var(--secondary);
  opacity: 0.5;
  cursor: not-allowed;
}

.download-button {
  padding: 0.5rem 1rem;
  background: var(--primary);
  color: white;
  border: none;
  border-radius: var(--radius);
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.download-button:hover {
  background: var(--primary-hover);
}

.job-status-container {
  background: var(--card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 1.5rem;
}

.job-info {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 1rem;
  margin: 1.5rem 0;
  padding: 1rem;
  background: var(--background);
  border-radius: var(--radius);
}

.job-info p {
  margin: 0;
}

.drawings {
  margin-top: 2rem;
}

.drawings h3 {
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}

.navigation {
  margin-top: 1.5rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
}

.navigation a {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  color: var(--primary);
  font-weight: 500;
}

.no-data {
  text-align: center;
  padding: 2rem;
  color: var(--text-secondary);
  background: var(--background);
  border-radius: var(--radius);
}

footer {
  margin-top: 3rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border);
  color: var(--text-secondary);
  text-align: center;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  header {
    flex-direction: column;
    gap: 1rem;
  }
  
  nav {
    width: 100%;
    justify-content: space-around;
  }
  
  .job-info {
    grid-template-columns: 1fr;
  }
}
EOF
    
    # Create src/App.js with improved UI and features
    cat > web/src/App.js << 'EOF'
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useParams, useLocation } from 'react-router-dom';
import axios from 'axios';
import { toast } from 'react-toastify';
import { FiUpload, FiList, FiHome, FiFileText, FiDownload, FiCheck, FiClock, FiAlertCircle, FiRefreshCw } from 'react-icons/fi';

import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
const API_KEY = process.env.REACT_APP_API_KEY || 'test-api-key';

// Configure axios
axios.defaults.baseURL = API_BASE_URL;
axios.defaults.headers.common['X-API-Key'] = API_KEY;

// Loading Spinner Component
const LoadingSpinner = ({ text = 'Loading...' }) => (
  <div className="loading">
    <FiRefreshCw size={24} />
    <span>{text}</span>
  </div>
);

// Navigation Link Component
const NavLink = ({ to, icon, children }) => {
  const location = useLocation();
  const isActive = location.pathname === to;
  
  return (
    <Link to={to} className={isActive ? 'active' : ''}>
      {icon}
      <span>{children}</span>
    </Link>
  );
};

// File Upload Component
const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError(null);
    setUploadResult(null);
  };

  const validateFile = (file) => {
    const validExtensions = ['stl', 'f3d', 'step', 'iges', 'brep'];
    const fileExtension = file.name.split('.').pop().toLowerCase();
    
    if (!validExtensions.includes(fileExtension)) {
      setError(`File type not supported. Please upload one of: ${validExtensions.join(', ')}`);
      return false;
    }
    
    if (file.size > 50 * 1024 * 1024) { // 50MB
      setError('File too large. Maximum size is 50MB.');
      return false;
    }
    
    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select a file to upload.');
      return;
    }
    
    if (!validateFile(file)) {
      return;
    }
    
    setUploading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await axios.post('/upload/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      setUploadResult(response.data);
      toast.success('File uploaded successfully!');
    } catch (error) {
      console.error('Upload error:', error);
      setError(error.response?.data?.detail || 'An error occurred during upload.');
      toast.error('Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-container card">
      <div className="card-header">
        <h2>Upload 3D Model</h2>
        <p className="text-secondary">Upload STL, F3D, STEP, IGES, or BREP files to generate technical drawings.</p>
      </div>
      
      <div className="card-body">
        <form onSubmit={handleSubmit}>
          <div className="file-input-container">
            <input
              type="file"
              id="file"
              onChange={handleFileChange}
              accept=".stl,.f3d,.step,.iges,.brep"
            />
            <label 
              htmlFor="file" 
              className={`file-input-label ${file ? 'has-file' : ''}`}
            >
              {file ? <FiCheck size={40} /> : <FiUpload size={40} />}
              <span className="mt-2">{file ? file.name : 'Choose a file or drag & drop here'}</span>
              <span className="text-secondary mt-1">
                {file 
                  ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` 
                  : 'STL, F3D, STEP, IGES, BREP (max 50MB)'}
              </span>
            </label>
          </div>
          
          <button 
            type="submit" 
            disabled={!file || uploading}
            className="upload-button"
          >
            {uploading ? (
              <span className="flex items-center justify-center">
                <FiRefreshCw className="mr-2" />
                Uploading...
              </span>
            ) : (
              'Generate Drawings'
            )}
          </button>
        </form>
        
        {error && (
          <div className="error-message mt-4">
            <FiAlertCircle className="mr-2" />
            {error}
          </div>
        )}
        
        {uploadResult && (
          <div className="success-message mt-4">
            <p><FiCheck className="mr-2" /> File uploaded successfully!</p>
            <p className="mt-2"><strong>Job ID:</strong> {uploadResult.job_id}</p>
            <p><strong>Status:</strong> <span className={`status-${uploadResult.status}`}>{uploadResult.status}</span></p>
            <div className="mt-4">
              <Link to={`/jobs/${uploadResult.job_id}`} className="btn btn-primary">
                <FiFileText className="mr-2" /> View Job Status
              </Link>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Job Status Component
const JobStatus = () => {
  const { jobId } = useParams();
  const [jobDetails, setJobDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Fetch job status
  useEffect(() => {
    const fetchJobStatus = async () => {
      try {
        const response = await axios.get(`/jobs/${jobId}`);
        setJobDetails(response.data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching job status:', error);
        setError('Failed to load job status.');
        setLoading(false);
      }
    };
    
    fetchJobStatus();
    
    // Poll for updates if job is still pending or processing
    const intervalId = setInterval(() => {
      if (jobDetails && ['pending', 'processing'].includes(jobDetails.status)) {
        fetchJobStatus();
      }
    }, 5000);
    
    return () => clearInterval(intervalId);
  }, [jobId, jobDetails?.status]);
  
  if (loading) {
    return <LoadingSpinner text="Loading job status..." />;
  }
  
  if (error) {
    return (
      <div className="error-message">
        <FiAlertCircle size={24} />
        <span>{error}</span>
      </div>
    );
  }
  
  const getStatusIcon = (status) => {
    switch(status) {
      case 'pending':
        return <FiClock />;
      case 'processing':
        return <FiRefreshCw />;
      case 'completed':
        return <FiCheck />;
      case 'failed':
        return <FiAlertCircle />;
      default:
        return null;
    }
  };
  
  return (
    <div className="job-status-container">
      <h2 className="flex items-center gap-2">
        Job Status
        {getStatusIcon(jobDetails.status)}
        <span className={`status-${jobDetails.status} ml-2`}>{jobDetails.status}</span>
      </h2>
      
      <div className="job-info mt-4">
        <p><strong>Job ID:</strong> {jobDetails.id}</p>
        <p><strong>Filename:</strong> {jobDetails.filename}</p>
        <p><strong>Created:</strong> {new Date(jobDetails.created_at).toLocaleString()}</p>
        {jobDetails.completed_at && (
          <p><strong>Completed:</strong> {new Date(jobDetails.completed_at).toLocaleString()}</p>
        )}
      </div>
      
      {jobDetails.status === 'processing' && (
        <div className="card mt-4 p-4 bg-blue-50">
          <div className="flex items-center">
            <FiRefreshCw className="animate-spin mr-2 text-blue-500" />
            <p>Your file is being processed. This may take a few minutes depending on the file complexity.</p>
          </div>
        </div>
      )}
      
      {jobDetails.status === 'completed' && jobDetails.drawing_set && (
        <div className="drawings">
          <h3 className="flex items-center gap-2">
            <FiFileText />
            Generated Drawings
          </h3>
          <table className="drawings-table">
            <thead>
              <tr>
                <th>Type</th>
                <th>Size</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {jobDetails.drawing_set.drawings.map((drawing) => (
                <tr key={drawing.id}>
                  <td>{drawing.type.toUpperCase()}</td>
                  <td>{formatFileSize(drawing.size)}</td>
                  <td>
                    <button 
                      onClick={() => downloadDrawing(drawing.id)}
                      className="download-button"
                    >
                      <FiDownload />
                      Download
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      
      {jobDetails.status === 'failed' && (
        <div className="error-message mt-4">
          <p><FiAlertCircle className="mr-2" /> Processing failed. Please try uploading the file again.</p>
        </div>
      )}
      
      <div className="navigation">
        <Link to="/" className="btn btn-secondary">
          <FiUpload className="mr-2" /> Upload Another File
        </Link>
      </div>
    </div>
  );
};

// Job List Component
const JobList = () => {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pagination, setPagination] = useState({
    limit: 10,
    offset: 0,
    total: 0
  });
  const [statusFilter, setStatusFilter] = useState('');
  
  const fetchJobs = async () => {
    try {
      setLoading(true);
      const params = {
        limit: pagination.limit,
        offset: pagination.offset
      };
      
      if (statusFilter) {
        params.status = statusFilter;
      }
      
      const response = await axios.get('/drawings/', { params });
      
      setJobs(response.data.results);
      setPagination({
        ...pagination,
        total: response.data.total
      });
      setLoading(false);
    } catch (error) {
      console.error('Error fetching jobs:', error);
      setError('Failed to load jobs.');
      setLoading(false);
      toast.error('Failed to load jobs.');
    }
  };
  
  useEffect(() => {
    fetchJobs();
  }, [pagination.offset, pagination.limit, statusFilter]);
  
  const handleStatusChange = (e) => {
    setStatusFilter(e.target.value);
    setPagination({
      ...pagination,
      offset: 0 // Reset to first page when changing filter
    });
  };
  
  const handleNextPage = () => {
    if ((pagination.offset + pagination.limit) < pagination.total) {
      setPagination({
        ...pagination,
        offset: pagination.offset + pagination.limit
      });
    }
  };
  
  const handlePrevPage = () => {
    if (pagination.offset > 0) {
      setPagination({
        ...pagination,
        offset: Math.max(0, pagination.offset - pagination.limit)
      });
    }
  };
  
  if (loading) {
    return <LoadingSpinner text="Loading jobs..." />;
  }
  
  if (error) {
    return (
      <div className="error-message">
        <FiAlertCircle size={24} />
        <span>{error}</span>
      </div>
    );
  }

  const getStatusIcon = (status) => {
    switch(status) {
      case 'pending':
        return <FiClock />;
      case 'processing':
        return <FiRefreshCw className="animate-spin" />;
      case 'completed':
        return <FiCheck />;
      case 'failed':
        return <FiAlertCircle />;
      default:
        return null;
    }
  };
  
  return (
    <div className="job-list-container">
      <h2 className="mb-4">Recent Jobs</h2>
      
      <div className="filters">
        <label className="flex items-center gap-2">
          Status:
          <select value={statusFilter} onChange={handleStatusChange} className="ml-2">
            <option value="">All</option>
            <option value="pending">Pending</option>
            <option value="processing">Processing</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
        </label>
        
        <button onClick={fetchJobs} className="btn btn-secondary">
          <FiRefreshCw />
          Refresh
        </button>
      </div>
      
      {jobs.length === 0 ? (
        <div className="no-data">
          <FiFileText size={40} className="mb-2" />
          <p>No jobs found.</p>
        </div>
      ) : (
        <>
          <table className="jobs-table">
            <thead>
              <tr>
                <th>Filename</th>
                <th>Status</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => (
                <tr key={job.id}>
                  <td>{job.filename}</td>
                  <td>
                    <span className={`status-${job.status} flex items-center gap-2`}>
                      {getStatusIcon(job.status)}
                      {job.status}
                    </span>
                  </td>
                  <td>{new Date(job.created_at).toLocaleString()}</td>
                  <td>
                    <Link to={`/jobs/${job.id}`} className="btn btn-primary">
                      View
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          
          <div className="pagination">
            <button 
              onClick={handlePrevPage} 
              disabled={pagination.offset === 0}
              className="btn btn-secondary"
            >
              Previous
            </button>
            <span>
              Showing {pagination.offset + 1} to {Math.min(pagination.offset + pagination.limit, pagination.total)} of {pagination.total}
            </span>
            <button 
              onClick={handleNextPage} 
              disabled={(pagination.offset + pagination.limit) >= pagination.total}
              className="btn btn-secondary"
            >
              Next
            </button>
          </div>
        </>
      )}
    </div>
  );
};

// Helper function to format file size
const formatFileSize = (bytes) => {
  if (bytes < 1024) {
    return bytes + ' B';
  } else if (bytes < 1024 * 1024) {
    return (bytes / 1024).toFixed(1) + ' KB';
  } else {
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }
};

// Helper function to download a drawing
const downloadDrawing = async (drawingId) => {
  try {
    const response = await axios.get(`/download/${drawingId}`);
    window.open(response.data.download_url, '_blank');
  } catch (error) {
    console.error('Download error:', error);
    toast.error('Failed to download drawing. Please try again.');
  }
};

// Main App
const App = () => {
  return (
    <Router>
      <div className="app-container">
        <header>
          <div className="logo">
            <FiFileText size={28} />
            <h1>Technical Drawing Generator</h1>
          </div>
          <nav>
            <NavLink to="/" icon={<FiHome />}>Home</NavLink>
            <NavLink to="/upload" icon={<FiUpload />}>Upload</NavLink>
            <NavLink to="/jobs" icon={<FiList />}>Recent Jobs</NavLink>
          </nav>
        </header>
        
        <main>
          <Routes>
            <Route path="/" element={<FileUpload />} />
            <Route path="/upload" element={<FileUpload />} />
            <Route path="/jobs" element={<JobList />} />
            <Route path="/jobs/:jobId" element={<JobStatus />} />
          </Routes>
        </main>
        
        <footer>
          <p>&copy; {new Date().getFullYear()} Technical Drawing Generator</p>
        </footer>
      </div>
    </Router>
  );
};

export default App;
EOF
    
    print_success "Web frontend files created with modern UI"
    
    cd ..
}

# Management commands function
generate_management_commands() {
    print_header "Generating Management Commands"
    
    cd technical-drawing-generator
    
    # Create management script
    cat > manage.sh << 'EOF'
#!/bin/bash

# Color codes for prettier output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print success message
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
    echo ""
}

# Print warning message
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    echo ""
}

# Print error message
print_error() {
    echo -e "${RED}❌ $1${NC}"
    echo ""
}

# Print info message
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
    echo ""
}

# Check if we need sudo for Docker commands
use_sudo=false
if ! docker info > /dev/null 2>&1; then
    if sudo -n true 2>/dev/null; then
        use_sudo=true
        print_warning "Using sudo for Docker commands (current user lacks Docker permissions)"
    else
        print_error "ERROR: Current user cannot run Docker commands."
        echo "Please either:"
        echo "  1. Add your user to the docker group: 'sudo usermod -aG docker $USER' and log out/in"
        echo "  2. Run this script with sudo"
        exit 1
    fi
fi

# Run docker command with sudo if needed
docker_cmd() {
    if $use_sudo; then
        sudo docker "$@"
    else
        docker "$@"
    fi
}

# Run docker-compose command with sudo if needed
compose_cmd() {
    if $use_sudo; then
        sudo docker compose "$@"
    else
        docker compose "$@"
    fi
}

# Start containers
start() {
    print_info "Starting containers..."
    compose_cmd up -d
    if [ $? -eq 0 ]; then
        print_success "Containers started successfully"
        echo "Web interface: http://localhost:8080"
        echo "API: http://localhost:5000"
        echo "RabbitMQ Management: http://localhost:15672 (guest/guest)"
    else
        print_error "Failed to start containers"
    fi
}

# Stop containers
stop() {
    print_info "Stopping containers..."
    compose_cmd down
    if [ $? -eq 0 ]; then
        print_success "Containers stopped successfully"
    else
        print_error "Failed to stop containers"
    fi
}

# Restart containers
restart() {
    print_info "Restarting containers..."
    compose_cmd restart
    if [ $? -eq 0 ]; then
        print_success "Containers restarted successfully"
    else
        print_error "Failed to restart containers"
    fi
}

# View container logs
logs() {
    if [ -z "$1" ]; then
        print_info "Viewing logs for all containers (press Ctrl+C to exit)..."
        compose_cmd logs -f
    else
        print_info "Viewing logs for $1 service (press Ctrl+C to exit)..."
        compose_cmd logs -f "$1"
    fi
}

# Rebuild containers
rebuild() {
    print_info "Rebuilding containers..."
    compose_cmd build
    if [ $? -eq 0 ]; then
        print_success "Containers rebuilt successfully"
        
        print_info "Restarting containers with new build..."
        compose_cmd up -d
        if [ $? -eq 0 ]; then
            print_success "Containers restarted with new build"
        else
            print_error "Failed to restart containers"
        fi
    else
        print_error "Failed to rebuild containers"
    fi
}

# Get container status
status() {
    print_info "Container status:"
    compose_cmd ps
}

# Display help
show_help() {
    echo "Technical Drawing Generator Management Tool"
    echo ""
    echo "Usage: ./manage.sh [command]"
    echo ""
    echo "Commands:"
    echo "  start       Start all containers"
    echo "  stop        Stop all containers"
    echo "  restart     Restart all containers"
    echo "  status      Show container status"
    echo "  logs [svc]  View logs (optionally for specific service)"
    echo "  rebuild     Rebuild and restart containers"
    echo "  purge       Remove all containers, volumes, and data"
    echo "  help        Show this help message"
    echo ""
    echo "Available services for logs:"
    echo "  web         Web frontend"
    echo "  api         API service"
    echo "  processor   CAD processor service"
    echo "  queue       RabbitMQ queue"
    echo "  db          PostgreSQL database"
    echo ""
}

# Purge all data
purge() {
    read -p "$(echo -e ${RED}WARNING: This will delete all containers, volumes, and data. Are you sure? \(y/n\):${NC} )" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Stopping containers..."
        compose_cmd down --volumes --remove-orphans
        
        print_info "Removing data directories..."
        rm -rf data logs
        
        print_info "Creating empty data directories..."
        mkdir -p data/db data/queue data/uploads data/outputs logs
        chmod -R 775 data logs
        
        print_success "System purged successfully"
    else
        print_info "Purge cancelled"
    fi
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        logs "$2"
        ;;
    rebuild)
        rebuild
        ;;
    purge)
        purge
        ;;
    help)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
EOF

    # Make script executable
    chmod +x manage.sh
    
    print_success "Management script created"
    
    cd ..
}

# Build and start the containers
build_and_start() {
    print_header "Building and Starting Containers"
    
    cd technical-drawing-generator
    
    # Build containers
    compose_cmd build
    
    if [ $? -ne 0 ]; then
        print_error "Failed to build containers. Please check the error messages above."
        cd ..
        return 1
    fi
    
    # Start containers
    compose_cmd up -d
    
    if [ $? -ne 0 ]; then
        print_error "Failed to start containers. Please check the error messages above."
        cd ..
        return 1
    fi
    
    # Check status
    compose_cmd ps
    
    print_success "Containers built and started successfully"
    
    cd ..
    return 0
}

# Check if system is ready
check_and_run() {
    # Check Docker
    check_docker
    
    # Ask for confirmation
    read -p "$(echo -e ${YELLOW}This script will create a new directory 'technical-drawing-generator' and set up all necessary files. Continue? \(y/n\):${NC} )" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Setup cancelled."
        exit 1
    fi
    
    # Clean up any existing resources
    cleanup_existing
    
    # Run setup
    setup_env_file
    create_directories
    create_docker_compose
    create_api_service
    create_processor_service
    create_web_frontend
    generate_management_commands
    
    # Ask if user wants to build and start containers
    read -p "$(echo -e ${YELLOW}Do you want to build and start the containers now? \(y/n\):${NC} )" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        build_and_start
        
        # Print final instructions
        echo ""
        echo -e "${GREEN}✨ Setup Complete! ✨${NC}"
        echo ""
        echo -e "The Technical Drawing Generator is now running. You can access it at:"
        echo -e "- ${CYAN}Web Interface:${NC} http://localhost:8080"
        echo -e "- ${CYAN}API Documentation:${NC} http://localhost:5000/docs"
        echo -e "- ${CYAN}RabbitMQ Management:${NC} http://localhost:15672 (guest/guest)"
        echo ""
        echo "You can upload STL, F3D, STEP, IGES, or BREP files through the web interface."
        echo ""
        echo -e "${YELLOW}MANAGEMENT COMMANDS${NC}"
        echo "Use the provided management script for easy container management:"
        echo -e "  cd technical-drawing-generator && ${CYAN}./manage.sh start${NC}    - Start all containers"
        echo -e "  cd technical-drawing-generator && ${CYAN}./manage.sh stop${NC}     - Stop all containers"
        echo -e "  cd technical-drawing-generator && ${CYAN}./manage.sh restart${NC}  - Restart all containers"
        echo -e "  cd technical-drawing-generator && ${CYAN}./manage.sh logs${NC}     - View logs"
        echo -e "  cd technical-drawing-generator && ${CYAN}./manage.sh status${NC}   - Check container status"
        echo -e "  cd technical-drawing-generator && ${CYAN}./manage.sh help${NC}     - Show all commands"
    else
        echo ""
        echo -e "${GREEN}✨ Setup Complete! ✨${NC}"
        echo ""
        echo "To build and start the containers, use the management script:"
        echo -e "  cd technical-drawing-generator && ${CYAN}./manage.sh start${NC}"
        echo ""
        echo "For a full list of management commands:"
        echo -e "  cd technical-drawing-generator && ${CYAN}./manage.sh help${NC}"
    fi
}

# Run the setup
check_and_run
