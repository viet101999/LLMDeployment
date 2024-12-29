# FastAPI LLM RESTful API

## Overview
This project provides a RESTful API for interacting with a quantized Large Language Model (LLM) using FastAPI. It supports text generation and performance metrics visualization.

---

## Features
- **Text Generation:** Generate text from a given prompt using the `/LLM/generate_text` endpoint.
- **Performance Metrics:** Measure system usage (CPU, RAM, VRAM) and inference speed using `/LLM/measure_speed` and `/metrics` endpoints.

---

## Getting Started

### Prerequisites
- Docker
- Python 3.9+ (if not using Docker)

---

### Running the Application with Docker
1. Clone this repository:
   ```bash
   git clone https://github.com/viet101999/LLMDeployment.git
   cd dev

2. Run the following command:
   docker compose -f "docker-compose.yml" up -d --build