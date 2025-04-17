#!/bin/bash
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' 
BLUE='\033[0;34m'

print_message() {
    echo -e "${2}$(date '+%Y-%m-%d %H:%M:%S') - ${1}${NC}"
}

show_spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

progress_bar() {
    local title=$1
    local duration=$2
    local sleep_time=$(echo "scale=2; $duration/20" | bc)
    
    echo -e "${BLUE}$title${NC}"
    echo -ne '['
    for ((i=0; i<20; i++)); do
        sleep $sleep_time
        echo -ne '#'
    done
    echo -e '] 100%'
}

cleanup() {
    print_message "Received termination signal. Shutting down services..." "${YELLOW}"
    print_message "Stopping and removing containers, volumes and orphans..." "${YELLOW}"
    docker-compose down --volumes --remove-orphans
    print_message "Cleanup complete. System has been shut down." "${GREEN}"
    exit 0
}

trap cleanup SIGINT SIGTERM

if ! docker info > /dev/null 2>&1; then
    print_message "Docker is not running. Please start Docker and try again." "${RED}"
    exit 1
fi

print_message "Setting up directory structure..." "${BLUE}"
mkdir -p airflow/dags data models airflow/logs api/app

if [ -f "ollama_model_management.py" ]; then
    print_message "Copying model management DAG to dags directory..." "${BLUE}"
    cp ollama_model_management.py airflow/dags/
    print_message "DAG copied successfully." "${GREEN}"
fi

if [ ! -f "docker-compose.yml" ]; then
    print_message "Error: docker-compose.yml not found!" "${RED}"
    exit 1
fi

print_message "Building required Docker images with Rust support..." "${BLUE}"
docker-compose build --build-arg INSTALL_RUST=true
print_message "Docker images built successfully." "${GREEN}"

print_message "Pulling required Docker images (this may take some time)..." "${BLUE}"
docker-compose pull &
PULL_PID=$!
show_spinner $PULL_PID
print_message "Docker images pulled successfully." "${GREEN}"

print_message "Starting Million Luxury Document Assistant services..." "${BLUE}"
docker-compose up -d

progress_bar "Starting services..." 10

print_message "Checking service health..." "${BLUE}"
sleep 5
EXPECTED_SERVICES=$(grep -E "^\s{2}[a-zA-Z0-9_-]+:" docker-compose.yml | wc -l)
RUNNING_SERVICES=$(docker-compose ps --services --filter "status=running" | wc -l)

if [ "$RUNNING_SERVICES" -lt "$EXPECTED_SERVICES" ]; then
    print_message "Warning: Not all services are running. Check docker-compose logs for errors." "${YELLOW}"
    print_message "Running services: $RUNNING_SERVICES out of $EXPECTED_SERVICES expected" "${YELLOW}"
    echo -e "${YELLOW}Services status:${NC}"
    docker-compose ps
else
    print_message "All services are running successfully!" "${GREEN}"
fi

print_message "Checking Ollama service readiness..." "${BLUE}"
MAX_ATTEMPTS=15
for ((i=1; i<=MAX_ATTEMPTS; i++)); do
    if curl -s "http://localhost:11435/api/tags" > /dev/null 2>&1; then
        print_message "Ollama service is ready!" "${GREEN}"
        break
    fi
    
    if [ $i -eq $MAX_ATTEMPTS ]; then
        print_message "Warning: Ollama service is not responding." "${YELLOW}"
    else
        echo -ne "${YELLOW}Waiting for Ollama service to be ready... ($i/$MAX_ATTEMPTS)${NC}\r"
        sleep 2
    fi
done

echo ""
print_message "Model Management Instructions:" "${BLUE}"
echo -e "  1. ${YELLOW}Access Airflow UI at http://localhost:8080${NC}"
echo -e "  2. ${YELLOW}Activate the 'ollama_model_management' DAG${NC}"
echo -e "  3. ${YELLOW}Trigger the DAG to download initial models${NC}"
echo -e "  4. ${YELLOW}The 'fine_tune_model_task' DAG runs weekly to download models and generate embeddings, making them ready for use in the main pipeline.${NC}"

echo ""
print_message "System is ready! Access the following services:" "${GREEN}"
echo -e "${BLUE}Airflow UI:${NC} http://localhost:8080/home (username: airflow, password: airflow)"
echo -e "${BLUE}FastAPI UI:${NC} http://localhost:8888"
echo -e "${BLUE}API Status:${NC} http://localhost:8888/status"

print_message "Press Ctrl+C to shut down all services" "${YELLOW}"
while true; do sleep 1; done