"""Tests for the cache module."""

import os
import tempfile
import time

import pytest

from audio2anki.cache import (
    CacheMetadata,
    cache_delete,
    cache_retrieve,
    cache_size,
    cache_store,
    clear_cache,
    compute_file_hash,
    expire_old_entries,
    get_cache_path,
    init_cache,
)


@pytest.fixture
def temp_input_file():
    """Create a temporary input file for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"test data")
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def setup_cache():
    """Set up and tear down the cache for each test."""
    init_cache()
    yield
    clear_cache()


def test_compute_file_hash(temp_input_file):
    """Test that file hashing is consistent."""
    hash1 = compute_file_hash(temp_input_file)
    hash2 = compute_file_hash(temp_input_file)
    assert hash1 == hash2
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA-256 produces 64 character hex strings


def test_init_cache(setup_cache):
    """Test cache initialization."""
    cache_dir = init_cache()
    assert os.path.exists(cache_dir)
    assert os.path.exists(os.path.join(cache_dir, "metadata.json"))


def test_get_cache_path():
    """Test cache path generation."""
    path = get_cache_path("test_stage", "abc123", ".mp3")
    assert path.endswith("abc123_test_stage.mp3")
    assert "audio2anki" in path


def test_cache_store_and_retrieve(temp_input_file, setup_cache):
    """Test storing and retrieving data from cache."""
    test_data = b"test data"
    stage_name = "test_stage"
    extension = ".txt"

    # Store data
    cache_path = cache_store(stage_name, temp_input_file, test_data, extension)
    assert os.path.exists(cache_path)

    # Retrieve data
    retrieved_data = cache_retrieve(stage_name, temp_input_file, extension, expiry_days=7)
    assert retrieved_data == test_data


def test_cache_expiry(temp_input_file, setup_cache, monkeypatch):
    """Test cache expiry functionality."""
    test_data = b"test data"
    stage_name = "test_stage"
    extension = ".txt"

    # Store data
    cache_path = cache_store(stage_name, temp_input_file, test_data, extension)

    # Mock time to simulate passage of time
    future_time = time.time() + 8 * 24 * 60 * 60  # 8 days in the future
    monkeypatch.setattr(time, "time", lambda: future_time)

    # Data should be expired after 7 days
    retrieved_data = cache_retrieve(stage_name, temp_input_file, extension, expiry_days=7)
    assert retrieved_data is None
    assert not os.path.exists(cache_path)


def test_cache_with_extra_params(temp_input_file, setup_cache):
    """Test cache behavior with extra parameters."""
    test_data = b"test data"
    stage_name = "test_stage"
    extension = ".txt"
    params1 = {"param1": "value1"}
    params2 = {"param1": "value2"}

    # Store with params1
    cache_store(stage_name, temp_input_file, test_data, extension, extra_params=params1)

    # Retrieve with different params should return None
    retrieved_data = cache_retrieve(stage_name, temp_input_file, extension, expiry_days=7, extra_params=params2)
    assert retrieved_data is None

    # Retrieve with same params should return data
    retrieved_data = cache_retrieve(stage_name, temp_input_file, extension, expiry_days=7, extra_params=params1)
    assert retrieved_data == test_data


def test_cache_delete(temp_input_file, setup_cache):
    """Test cache deletion."""
    test_data = b"test data"
    stage_name = "test_stage"
    extension = ".txt"

    # Store data
    cache_path = cache_store(stage_name, temp_input_file, test_data, extension)
    assert os.path.exists(cache_path)

    # Delete cache
    cache_delete(cache_path)
    assert not os.path.exists(cache_path)

    # Retrieve should return None
    retrieved_data = cache_retrieve(stage_name, temp_input_file, extension, expiry_days=7)
    assert retrieved_data is None


def test_expire_old_entries(temp_input_file, setup_cache, monkeypatch):
    """Test expiring old cache entries."""
    test_data = b"test data"
    stage_name = "test_stage"
    extension = ".txt"

    # Store data
    cache_path = cache_store(stage_name, temp_input_file, test_data, extension)

    # Mock time to simulate passage of time
    future_time = time.time() + 8 * 24 * 60 * 60  # 8 days in the future
    monkeypatch.setattr(time, "time", lambda: future_time)

    # Expire old entries
    expire_old_entries(expiry_days=7)
    assert not os.path.exists(cache_path)


def test_cache_size(temp_input_file, setup_cache):
    """Test cache size calculation."""
    test_data = b"test data"
    stage_name = "test_stage"
    extension = ".txt"

    # Initial size should be small (just metadata.json)
    initial_size = cache_size()

    # Store data
    cache_store(stage_name, temp_input_file, test_data, extension)

    # Size should increase
    assert cache_size() > initial_size


def test_clear_cache(temp_input_file, setup_cache):
    """Test clearing the entire cache."""
    test_data = b"test data"
    stage_name = "test_stage"
    extension = ".txt"

    # Store data
    cache_store(stage_name, temp_input_file, test_data, extension)
    assert cache_size() > 0

    # Clear cache
    clear_cache()
    assert cache_size() == 0  # Should only contain empty metadata.json


def test_cache_metadata(monkeypatch):
    """Test CacheMetadata class functionality."""
    current_time = time.time()
    monkeypatch.setattr(time, "time", lambda: current_time)

    metadata = CacheMetadata(
        created_at=current_time,
        stage_name="test_stage",
        input_hash="abc123",
        extra_params={"param1": "value1"},
    )

    # Test serialization
    data_dict = metadata.to_dict()
    assert isinstance(data_dict, dict)
    assert "created_at" in data_dict
    assert "stage_name" in data_dict
    assert "input_hash" in data_dict
    assert "extra_params" in data_dict

    # Test deserialization
    new_metadata = CacheMetadata.from_dict(data_dict)
    assert new_metadata.created_at == metadata.created_at
    assert new_metadata.stage_name == metadata.stage_name
    assert new_metadata.input_hash == metadata.input_hash
    assert new_metadata.extra_params == metadata.extra_params

    # Test expiry
    assert not metadata.is_expired(expiry_days=7)  # Should not be expired immediately

    # Mock time to be 8 days in the future
    future_time = current_time + 8 * 24 * 60 * 60
    monkeypatch.setattr(time, "time", lambda: future_time)

    # Set creation time to 10 days ago relative to future_time
    metadata.created_at = future_time - (10 * 24 * 60 * 60)
    assert metadata.is_expired(expiry_days=7)  # Should be expired after 7 days
