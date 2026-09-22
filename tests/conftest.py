"""Shared test fixtures for alex-teacher tests."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment."""
    # Add project root to path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Set test environment variables
    import os
    os.environ['ALEX_APP'] = 'alex-teacher-test'
    os.environ['ALEX_LOG_DIR'] = '/tmp/alex-teacher-test-logs'
    
    yield
    
    # Cleanup
    os.environ.pop('ALEX_APP', None)
    os.environ.pop('ALEX_LOG_DIR', None)


@pytest.fixture
def mock_state():
    """Mock state module."""
    with patch('state') as mock:
        mock.cache = Mock()
        mock.cache._maxsize = 100
        mock.cache.get.return_value = None
        mock.stats = Mock()
        mock.stats.get_stats.return_value = {'requests': 0}
        mock.piper = None
        yield mock


@pytest.fixture
def mock_config():
    """Mock config module."""
    with patch('config') as mock:
        mock.PORT = 3000
        mock.LLAMA_HOST = 'http://localhost:8080/v1'
        mock.LLAMA_MODEL = 'test-model'
        mock.NUM_CTX = 32768
        mock.SERVER_MODE = 'teacher'
        mock.FRONTEND_DIR = Path(__file__).parent.parent / 'frontend'
        mock.KOKORO_CONFIG = {
            'es': {'lang': 'es', 'voice': 'em_alex', 'speed': 0.9},
            'en': {'lang': 'en-us', 'voice': 'af_heart', 'speed': 1.0},
            'ja': {'lang': 'ja', 'voice': 'jf_alpha', 'speed': 0.9},
        }
        mock.STORY_DIR = Path('/tmp/test-stories')
        mock.STORY_INDEX = Path('/tmp/test-stories/index.json')
        mock.STORY_CAST = [
            {"id": "narrator", "name": "Narrator", "emoji": "🎙️",
             "voices": {"es": "em_alex", "ja": "jm_kumo", "en": "am_adam"}},
            {"id": "zorro", "name": "Zorro", "emoji": "🦊",
             "voices": {"es": "em_santa", "ja": "jf_tebukuro", "en": "am_michael"}},
        ]
        mock.STORY_CAST_IDS = ["narrator", "zorro"]
        mock.STORY_CAST_HINT = "narrator, zorro"
        yield mock


@pytest.fixture
def sample_english_text():
    """Sample English text for testing."""
    return "Hello, how are you today?"


@pytest.fixture
def sample_spanish_text():
    """Sample Spanish text for testing."""
    return "¡Hola, cómo estás hoy?"


@pytest.fixture
def sample_japanese_text():
    """Sample Japanese text for testing."""
    return "こんにちは、今日はどうですか？"


@pytest.fixture
def sample_audio_data():
    """Sample audio data for testing."""
    import numpy as np
    # Create 1 second of silence at 16kHz
    return np.zeros(16000, dtype=np.float32).tobytes()