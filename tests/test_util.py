"""Unit tests for alex-teacher util module."""

import pytest
from util import _safe_print


class TestSafePrint:
    """Tests for safe print function."""
    
    def test_safe_print_normal(self, capsys):
        """Test safe print with normal text."""
        _safe_print("Hello world")
        
        captured = capsys.readouterr()
        assert "Hello world" in captured.out
    
    def test_safe_print_unicode(self, capsys):
        """Test safe print with unicode characters."""
        _safe_print("¡Hola! 🌍")
        
        captured = capsys.readouterr()
        assert "¡Hola!" in captured.out
    
    def test_safe_print_empty(self, capsys):
        """Test safe print with empty string."""
        _safe_print("")
        
        captured = capsys.readouterr()
        # Should not raise exception
        assert captured.out == "" or "\n" in captured.out
    
    def test_safe_print_special_chars(self, capsys):
        """Test safe print with special characters."""
        _safe_print("Hello @#$%^&*()")
        
        captured = capsys.readouterr()
        assert "Hello" in captured.out
    
    def test_safe_print_long_text(self, capsys):
        """Test safe print with long text."""
        long_text = "x" * 10000
        _safe_print(long_text)
        
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestUtilityFunctions:
    """Tests for other utility functions."""
    
    def test_import_util(self):
        """Test util module import."""
        import util
        
        assert hasattr(util, '_safe_print')
    
    def test_safe_print_is_function(self):
        """Test that safe_print is a callable function."""
        from util import _safe_print
        
        assert callable(_safe_print)