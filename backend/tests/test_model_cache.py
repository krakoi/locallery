"""Local model files take precedence over Hub requests, including mutable refs."""

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from locallery.embedding import load_processor, local_first


def missing_config():
    error = OSError("Required configuration is not cached")
    error.__cause__ = LocalEntryNotFoundError("No cached config.json")
    return error


@pytest.mark.parametrize(
    "error",
    [
        missing_config(),
        LocalEntryNotFoundError("Missing processor configuration"),
        FileNotFoundError("Missing model shard"),
        OSError("google/model does not appear to have a file named model.safetensors"),
        OSError("Can't load image processor for 'google/model'"),
        OSError("Can't load video processor for 'google/model'"),
        OSError("Can't load tokenizer for 'google/model'"),
    ],
)
def test_missing_files_allow_one_download_with_same_revision_and_cache(error):
    calls = []

    def loader(model, *args, **kwargs):
        calls.append((model, args, kwargs))
        if kwargs["local_files_only"]:
            raise error
        return "downloaded"

    assert (
        local_first(
            loader, "google/model", "file.json", revision="pinned", cache_dir="cache"
        )
        == "downloaded"
    )
    assert [kwargs["local_files_only"] for _, _, kwargs in calls] == [True, False]
    assert all(
        model == "google/model"
        and args == ("file.json",)
        and kwargs["revision"] == "pinned"
        and kwargs["cache_dir"] == "cache"
        for model, args, kwargs in calls
    )


@pytest.mark.parametrize(
    "error",
    [
        OSError("Cached config.json is not a valid JSON file"),
        PermissionError("Cache is not readable"),
        ValueError("Wrong checkpoint architecture"),
        RuntimeError("CUDA out of memory"),
    ],
)
def test_loading_errors_do_not_trigger_download(error):
    calls = []

    def loader(model, **kwargs):
        calls.append(kwargs["local_files_only"])
        raise error

    with pytest.raises(type(error)) as raised:
        local_first(loader, "google/model")
    assert raised.value is error
    assert calls == [True]


def test_complete_cache_does_not_check_hub_and_loads_cached_mutable_revision():
    calls = []

    def loader(model, **kwargs):
        calls.append(kwargs)
        assert kwargs["local_files_only"] is True
        assert kwargs["revision"] == "main"
        return "cached"

    assert local_first(loader, "google/model", revision="main") == "cached"
    assert len(calls) == 1


def test_local_directory_never_falls_back_to_hub(tmp_path):
    calls = []

    def loader(model, **kwargs):
        calls.append(kwargs["local_files_only"])
        raise FileNotFoundError("Missing local weights")

    with pytest.raises(FileNotFoundError):
        local_first(loader, str(tmp_path))
    assert calls == [True]


def test_explicit_offline_mode_never_falls_back_to_hub(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "is_offline_mode", lambda: True)
    calls = []

    def loader(model, **kwargs):
        calls.append(kwargs["local_files_only"])
        raise missing_config()

    with pytest.raises(OSError):
        local_first(loader, "google/model")
    assert calls == [True]


def test_failed_download_is_not_retried():
    calls = []

    def loader(model, **kwargs):
        calls.append(kwargs["local_files_only"])
        if kwargs["local_files_only"]:
            raise missing_config()
        raise OSError("Hub is unavailable")

    with pytest.raises(OSError, match="Hub is unavailable"):
        local_first(loader, "google/model")
    assert calls == [True, False]


@pytest.mark.parametrize("missing", ["configuration", "template"])
def test_incomplete_processor_cache_can_download_missing_files(monkeypatch, missing):
    from types import SimpleNamespace

    import transformers
    from transformers.utils import hub

    calls = []

    def processor(model, **kwargs):
        calls.append(kwargs["local_files_only"])
        if kwargs["local_files_only"]:
            if missing == "configuration":
                raise ValueError("Unrecognized image processor in google/model")
            return SimpleNamespace(chat_template=None)
        return SimpleNamespace(chat_template="template")

    def config(model, filename, **kwargs):
        assert filename == "processor_config.json"
        assert kwargs["local_files_only"] is True
        raise missing_config()

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", processor)
    monkeypatch.setattr(hub, "cached_file", config)
    assert local_first(load_processor, "google/model").chat_template == "template"
    assert calls == [True, False]


def test_invalid_existing_processor_config_does_not_download(monkeypatch):
    import transformers
    from transformers.utils import hub

    calls = []

    def processor(model, **kwargs):
        calls.append(kwargs["local_files_only"])
        raise ValueError("Unrecognized image processor in google/model")

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", processor)
    monkeypatch.setattr(
        hub, "cached_file", lambda *args, **kwargs: "processor_config.json"
    )
    with pytest.raises(ValueError, match="Unrecognized image processor"):
        local_first(load_processor, "google/model")
    assert calls == [True]
