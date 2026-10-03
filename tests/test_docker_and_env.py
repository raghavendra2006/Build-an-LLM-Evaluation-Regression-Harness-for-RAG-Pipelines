import os
import yaml
import pytest

def test_docker_compose_setup():
    compose_path = "docker-compose.yml"
    assert os.path.exists(compose_path), "docker-compose.yml does not exist"
    
    with open(compose_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    assert "services" in data, "docker-compose.yml missing 'services' block"
    services = data["services"]
    
    # Must define at least one vector DB service
    vector_db_found = False
    for s_name, s_config in services.items():
        if "qdrant" in s_name.lower() or "chroma" in s_name.lower() or "weaviate" in s_name.lower() or "postgres" in s_name.lower():
            vector_db_found = True
            assert "healthcheck" in s_config, f"Vector DB service '{s_name}' missing healthcheck definition"
            
    assert vector_db_found, "No Vector DB service found in docker-compose.yml"

def test_env_example_setup():
    env_path = ".env.example"
    assert os.path.exists(env_path), ".env.example does not exist"
    
    with open(env_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert "OPENAI_API_KEY=" in content, ".env.example missing OPENAI_API_KEY placeholder"
    assert "VECTOR_DB" in content, ".env.example missing VECTOR_DB config"
    
    # Ensure no real secret keys are present
    assert "sk-proj-" not in content and "sk-admin-" not in content
