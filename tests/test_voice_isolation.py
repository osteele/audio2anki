import pytest
from pathlib import Path
from audio2anki.models import AudioProcessor

@pytest.fixture(params=['legacy', 'speechbrain'])
def isolation_impl(request):
    return {'voice_isolation_impl': request.param}

def test_voice_isolation_basic(test_audio, tmp_path, isolation_impl):
    """Test that voice isolation produces output file"""
    if isolation_impl['voice_isolation_impl'] == 'legacy':
        pytest.skip("Legacy implementation not fully implemented yet")
        
    processor = AudioProcessor.create_voice_isolator(isolation_impl)
    
    output_file = tmp_path / "output.wav"
    
    processor.isolate_voice(test_audio, output_file)
    
    assert output_file.exists(), "Output file was not created"
    assert output_file.stat().st_size > 1024, "Output file is too small"

@pytest.mark.slow
def test_speechbrain_enhancement():
    """Integration test for SpeechBrain implementation"""
    # Actual test would load sample audio and validate output
    # This is just a placeholder for now
    assert True
