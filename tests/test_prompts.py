"""Unit tests for alex-teacher prompts module."""

import pytest
from prompts import parse_multi_output, get_tts_text, get_system_prompt


class TestParseMultiOutput:
    """Tests for multi-output parsing."""
    
    def test_parse_text_output(self):
        """Test parsing TEXT output."""
        text = "【TEXT】Hello, how are you?"
        parsed = parse_multi_output(text)
        
        assert 'text' in parsed
        assert parsed['text'] == "Hello, how are you?"
    
    def test_parse_pronunciation_output(self):
        """Test parsing PRONUNCIATION output."""
        text = "【PRONUNCIATION】Hello: heh-LOH"
        parsed = parse_multi_output(text)
        
        assert 'pronunciation' in parsed
        assert parsed['pronunciation'] == "Hello: heh-LOH"
    
    def test_parse_translation_output(self):
        """Test parsing TRANSLATION output."""
        text = "【TRANSLATION】Hola, ¿cómo estás?"
        parsed = parse_multi_output(text)
        
        assert 'translation' in parsed
        assert parsed['translation'] == "Hola, ¿cómo estás?"
    
    def test_parse_explanation_output(self):
        """Test parsing EXPLANATION output."""
        text = "【EXPLANATION】This is a greeting phrase."
        parsed = parse_multi_output(text)
        
        assert 'explanation' in parsed
        assert parsed['explanation'] == "This is a greeting phrase."
    
    def test_parse_exercise_output(self):
        """Test parsing EXERCISE output."""
        text = "【EXERCISE】Fill in the blank: Hello, ___ are you?"
        parsed = parse_multi_output(text)
        
        assert 'exercise' in parsed
        assert parsed['exercise'] == "Fill in the blank: Hello, ___ are you?"
    
    def test_parse_multiple_outputs(self):
        """Test parsing multiple outputs."""
        text = """【TEXT】Hello!
【PRONUNCIATION】Hello: heh-LOH
【TRANSLATION】¡Hola!
【EXPLANATION】This is a common greeting.
【EXERCISE】Say hello to someone."""
        parsed = parse_multi_output(text)
        
        assert 'text' in parsed
        assert 'pronunciation' in parsed
        assert 'translation' in parsed
        assert 'explanation' in parsed
        assert 'exercise' in parsed
    
    def test_parse_empty_text(self):
        """Test parsing empty text."""
        text = ""
        parsed = parse_multi_output(text)
        
        assert isinstance(parsed, dict)
        assert len(parsed) == 0
    
    def test_parse_no_markers(self):
        """Test parsing text without markers."""
        text = "Hello, how are you?"
        parsed = parse_multi_output(text)
        
        # Should still return a dict
        assert isinstance(parsed, dict)


class TestGetTTSText:
    """Tests for TTS text extraction."""
    
    def test_get_tts_text_teacher_mode(self):
        """Test TTS text extraction in teacher mode."""
        text = """【TEXT】Hello!
【PRONUNCIATION】Hello: heh-LOH
【TRANSLATION】¡Hola!"""
        tts_text = get_tts_text(text, "teacher")
        
        # Should extract pronunciation text
        assert tts_text is not None
        assert len(tts_text) > 0
    
    def test_get_tts_text_conversation_mode(self):
        """Test TTS text extraction in conversation mode."""
        text = "Hello, how are you?"
        tts_text = get_tts_text(text, "conversation")
        
        # Should return the text as-is
        assert tts_text == text
    
    def test_get_tts_text_empty(self):
        """Test TTS text extraction with empty text."""
        text = ""
        tts_text = get_tts_text(text, "teacher")
        
        assert tts_text == ""


class TestGetSystemPrompt:
    """Tests for system prompt generation."""
    
    def test_get_teacher_prompt(self):
        """Test teacher system prompt."""
        prompt = get_system_prompt("teacher")
        
        assert prompt is not None
        assert len(prompt) > 0
        # Should contain multi-output format instructions
        assert "TEXT" in prompt or "【TEXT】" in prompt
    
    def test_get_conversation_prompt(self):
        """Test conversation system prompt."""
        prompt = get_system_prompt("conversation")
        
        assert prompt is not None
        assert len(prompt) > 0
    
    def test_get_unknown_mode_prompt(self):
        """Test system prompt for unknown mode."""
        prompt = get_system_prompt("unknown")
        
        # Should return a default prompt
        assert prompt is not None
        assert len(prompt) > 0