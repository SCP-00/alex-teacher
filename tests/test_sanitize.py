"""Unit tests for alex-teacher sanitize module."""

import pytest
from sanitize import (
    detect_language,
    _sanitize_tts_text,
    _split_sentences,
    _make_silence_pcm,
    _fix_word_spacing,
)


class TestLanguageDetection:
    """Tests for language detection."""
    
    def test_detect_english(self):
        """Test English language detection."""
        text = "Hello, how are you today?"
        lang = detect_language(text)
        assert lang == "en"
    
    def test_detect_spanish(self):
        """Test Spanish language detection."""
        text = "Hola, ¿cómo estás hoy?"
        lang = detect_language(text)
        assert lang == "es"
    
    def test_detect_japanese(self):
        """Test Japanese language detection."""
        text = "こんにちは、今日はどうですか？"
        lang = detect_language(text)
        assert lang == "ja"
    
    def test_detect_empty_text(self):
        """Test detection with empty text."""
        text = ""
        lang = detect_language(text)
        assert lang in ["es", "en"]  # Default language
    
    def test_detect_mixed_text(self):
        """Test detection with mixed language text."""
        text = "Hello 你好"
        lang = detect_language(text)
        # Should detect based on dominant characters
        assert lang in ["en", "ja", "zh"]


class TestTextSanitization:
    """Tests for TTS text sanitization."""
    
    def test_sanitize_english(self):
        """Test English text sanitization."""
        text = "Hello, world!"
        sanitized = _sanitize_tts_text(text, "en")
        assert sanitized == "Hello, world!"
    
    def test_sanitize_spanish(self):
        """Test Spanish text sanitization."""
        text = "¡Hola, mundo!"
        sanitized = _sanitize_tts_text(text, "es")
        assert sanitized == "¡Hola, mundo!"
    
    def test_sanitize_remove_special_chars(self):
        """Test removal of special characters."""
        text = "Hello @#$%^&*() world!"
        sanitized = _sanitize_tts_text(text, "en")
        assert "@" not in sanitized
        assert "#" not in sanitized
    
    def test_sanitize_preserve_punctuation(self):
        """Test preservation of sentence punctuation."""
        text = "Hello! How are you?"
        sanitized = _sanitize_tts_text(text, "en")
        assert "!" in sanitized
        assert "?" in sanitized
    
    def test_sanitize_empty_text(self):
        """Test sanitization of empty text."""
        text = ""
        sanitized = _sanitize_tts_text(text, "en")
        assert sanitized == ""


class TestSentenceSplitting:
    """Tests for sentence splitting."""
    
    def test_split_single_sentence(self):
        """Test splitting single sentence."""
        text = "Hello world"
        sentences = _split_sentences(text, "en")
        assert len(sentences) == 1
        assert sentences[0] == "Hello world"
    
    def test_split_multiple_sentences(self):
        """Test splitting multiple sentences."""
        text = "Hello world. How are you? I'm fine!"
        sentences = _split_sentences(text, "en")
        assert len(sentences) == 3
    
    def test_split_spanish_sentences(self):
        """Test splitting Spanish sentences."""
        text = "¡Hola! ¿Cómo estás? ¡Muy bien!"
        sentences = _split_sentences(text, "es")
        assert len(sentences) == 3
    
    def test_split_empty_text(self):
        """Test splitting empty text."""
        text = ""
        sentences = _split_sentences(text, "en")
        assert len(sentences) == 0


class TestSilenceGeneration:
    """Tests for silence PCM generation."""
    
    def test_generate_silence(self):
        """Test silence generation."""
        sr = 24000
        duration_ms = 100
        silence = _make_silence_pcm(sr, duration_ms)
        
        # Check that silence is bytes
        assert isinstance(silence, bytes)
        
        # Check duration (16-bit mono = 2 bytes per sample)
        expected_samples = int(sr * duration_ms / 1000)
        expected_bytes = expected_samples * 2
        assert len(silence) == expected_bytes
    
    def test_generate_silence_different_durations(self):
        """Test silence generation with different durations."""
        sr = 24000
        
        silence_100 = _make_silence_pcm(sr, 100)
        silence_200 = _make_silence_pcm(sr, 200)
        
        assert len(silence_200) == len(silence_100) * 2


class TestWordSpacing:
    """Tests for word spacing fixes."""
    
    def test_fix_spacing_normal(self):
        """Test normal text spacing."""
        text = "Hello world"
        fixed = _fix_word_spacing(text)
        assert fixed == "Hello world"
    
    def test_fix_spacing_extra_spaces(self):
        """Test extra spaces removal."""
        text = "Hello   world"
        fixed = _fix_word_spacing(text)
        assert "  " not in fixed
    
    def test_fix_spacing_empty(self):
        """Test empty text."""
        text = ""
        fixed = _fix_word_spacing(text)
        assert fixed == ""
    
    def test_fix_spacing_preserve_structure(self):
        """Test preservation of sentence structure."""
        text = "Hello world. How are you?"
        fixed = _fix_word_spacing(text)
        assert "." in fixed
        assert "?" in fixed