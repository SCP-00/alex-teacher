"""Unit tests for alex-teacher stories module."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch


class TestStoryHelpers:
    """Tests for story helper functions."""
    
    def test_safe_story_id_valid(self):
        """Test valid story ID."""
        from stories import _safe_story_id
        
        assert _safe_story_id("test-story_123") is True
        assert _safe_story_id("abc") is True
        assert _safe_story_id("20240101-120000-a1b2c3") is True
    
    def test_safe_story_id_invalid(self):
        """Test invalid story ID."""
        from stories import _safe_story_id
        
        assert _safe_story_id("") is False
        assert _safe_story_id("../etc/passwd") is False
        assert _safe_story_id("test/story") is False
        assert _safe_story_id("test story") is False
    
    def test_extract_json_valid(self):
        """Test JSON extraction from valid JSON string."""
        from stories import _extract_json
        
        text = '{"title": "Test", "lines": []}'
        result = _extract_json(text)
        
        assert isinstance(result, dict)
        assert result['title'] == 'Test'
    
    def test_extract_json_with_prose(self):
        """Test JSON extraction from text with prose."""
        from stories import _extract_json
        
        text = 'Here is the story: {"title": "Test", "lines": []} enjoy!'
        result = _extract_json(text)
        
        assert isinstance(result, dict)
        assert result['title'] == 'Test'
    
    def test_extract_json_invalid(self):
        """Test JSON extraction from invalid text."""
        from stories import _extract_json
        
        text = 'No JSON here'
        
        with pytest.raises(ValueError):
            _extract_json(text)


class TestStoryIndex:
    """Tests for story index operations."""
    
    @pytest.fixture
    def temp_story_dir(self, tmp_path):
        """Create temporary story directory.

        NOTE: stories.py hace `from config import STORY_DIR` (copia en import),
        así que hay que parchear el namespace de `stories`, no el de `config`.
        """
        import stories
        with patch.object(stories, 'STORY_DIR', tmp_path):
            with patch.object(stories, 'STORY_INDEX', tmp_path / 'index.json'):
                yield tmp_path
    
    def test_load_index_empty(self, temp_story_dir):
        """Test loading index when no index exists."""
        from stories import _story_load_index
        
        index = _story_load_index()
        
        assert isinstance(index, list)
        assert len(index) == 0
    
    def test_load_index_with_data(self, temp_story_dir):
        """Test loading index with existing data."""
        from stories import _story_load_index
        
        # Create test index
        test_data = [
            {"id": "test-1", "title": "Test Story 1"},
            {"id": "test-2", "title": "Test Story 2"},
        ]
        (temp_story_dir / 'index.json').write_text(json.dumps(test_data))
        
        index = _story_load_index()
        
        assert len(index) == 2
        assert index[0]['id'] == 'test-1'
    
    def test_story_load_nonexistent(self, temp_story_dir):
        """Test loading nonexistent story."""
        from stories import _story_load
        
        story = _story_load("nonexistent")
        
        assert story is None
    
    def test_story_delete_nonexistent(self, temp_story_dir):
        """Test deleting nonexistent story."""
        from stories import _story_delete
        
        result = _story_delete("nonexistent")
        
        assert result is False


class TestStoryCast:
    """Tests for story cast configuration."""
    
    def test_cast_has_narrator(self):
        """Test that cast includes narrator."""
        from config import STORY_CAST
        
        narrator = next((c for c in STORY_CAST if c['id'] == 'narrator'), None)
        assert narrator is not None
        assert 'voices' in narrator
    
    def test_cast_has_required_characters(self):
        """Test that cast has required characters."""
        from config import STORY_CAST
        
        required_ids = ['narrator', 'zorro', 'panda', 'dragon', 'conejo', 'buho']
        cast_ids = [c['id'] for c in STORY_CAST]
        
        for req_id in required_ids:
            assert req_id in cast_ids
    
    def test_cast_ids_string(self):
        """Test STORY_CAST_IDS string."""
        from config import STORY_CAST_IDS
        
        assert isinstance(STORY_CAST_IDS, list)
        assert len(STORY_CAST_IDS) > 0
    
    def test_cast_hint_string(self):
        """Test STORY_CAST_HINT string."""
        from config import STORY_CAST_HINT
        
        assert isinstance(STORY_CAST_HINT, str)
        assert len(STORY_CAST_HINT) > 0
        # Should contain comma-separated IDs
        assert ',' in STORY_CAST_HINT