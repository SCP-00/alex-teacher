"""Unit tests for alex-teacher config module."""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch


class TestConfig:
    """Tests for configuration loading and validation."""
    
    def test_default_port(self):
        """Test default port configuration."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove PORT from env if exists
            os.environ.pop('PLAN_B_PORT', None)
            
            # Import config (this will use defaults)
            import importlib
            import config
            importlib.reload(config)
            
            # Default port should be 3000
            assert config.PORT == 3000
    
    def test_custom_port_from_env(self):
        """Test port configuration from environment variable."""
        with patch.dict(os.environ, {'PLAN_B_PORT': '4000'}):
            import importlib
            import config
            importlib.reload(config)
            
            assert config.PORT == 4000
    
    def test_custom_port_from_args(self):
        """Test port configuration from command line arguments."""
        test_args = ['server.py', '--port', '5000']
        with patch.object(sys, 'argv', test_args):
            import importlib
            import config
            importlib.reload(config)
            
            assert config.PORT == 5000
    
    def test_ollama_model_config(self):
        """Test Ollama model configuration."""
        with patch.dict(os.environ, {'LLAMA_MODEL': 'custom-model'}):
            import importlib
            import config
            importlib.reload(config)
            
            assert config.LLAMA_MODEL == 'custom-model'
    
    def test_llama_host_config(self):
        """Test Llama host configuration."""
        with patch.dict(os.environ, {'LLAMA_HOST': 'http://custom-host:8080'}):
            import importlib
            import config
            importlib.reload(config)
            
            assert config.LLAMA_HOST == 'http://custom-host:8080'
    
    def test_kokoro_config_structure(self):
        """Test Kokoro configuration structure."""
        import config
        
        # Check that all required languages are present
        required_langs = ['es', 'en', 'ja', 'fr', 'de']
        for lang in required_langs:
            assert lang in config.KOKORO_CONFIG
            assert 'lang' in config.KOKORO_CONFIG[lang]
            assert 'voice' in config.KOKORO_CONFIG[lang]
            assert 'speed' in config.KOKORO_CONFIG[lang]
    
    def test_frontend_dir_exists(self):
        """Test that frontend directory exists."""
        import config
        
        assert config.FRONTEND_DIR.exists()
        assert config.FRONTEND_DIR.is_dir()
    
    def test_project_root_path(self):
        """Test project root path."""
        import config
        
        assert config.PROJECT_ROOT.exists()
        assert config.PROJECT_ROOT.is_dir()
    
    def test_server_mode_config(self):
        """Test server mode configuration."""
        with patch.dict(os.environ, {'CONV_MODE': 'teacher'}):
            import importlib
            import config
            importlib.reload(config)
            
            assert config.SERVER_MODE == 'teacher'
    
    def test_num_ctx_config(self):
        """Test context window configuration."""
        with patch.dict(os.environ, {'NUM_CTX': '16384'}):
            import importlib
            import config
            importlib.reload(config)
            
            assert config.NUM_CTX == 16384
    
    def test_story_dir_config(self):
        """Test story directory configuration."""
        import config
        
        assert hasattr(config, 'STORY_DIR')
        assert hasattr(config, 'STORY_INDEX')
    
    def test_story_cast_config(self):
        """Test story cast configuration."""
        import config
        
        assert hasattr(config, 'STORY_CAST')
        assert hasattr(config, 'STORY_CAST_IDS')
        assert hasattr(config, 'STORY_CAST_HINT')
        
        # Check that cast has required fields
        for character in config.STORY_CAST:
            assert 'id' in character
            assert 'name' in character
            assert 'emoji' in character
            assert 'voices' in character