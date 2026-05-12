from serve_engine.backends.trtllm import TRTLLMBackend
from serve_engine.lifecycle.plan import DeploymentPlan


def _plan(**overrides):
    base = dict(
        model_name="llama-1b",
        hf_repo="meta-llama/Llama-3.2-1B-Instruct",
        revision="main",
        backend="trtllm",
        image_tag="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14",
        gpu_ids=[0],
        max_model_len=8192,
        target_concurrency=8,
    )
    base.update(overrides)
    return DeploymentPlan(**base)


def test_build_argv_minimum():
    argv = TRTLLMBackend().build_argv(_plan(), local_model_path="/models/llama-1b")
    # trtllm-serve takes the model path positionally, immediately after the
    # launcher. The release image does not embed the launcher, so we prepend it.
    assert argv[0] == "trtllm-serve"
    assert argv[1] == "/models/llama-1b"
    assert "--backend" in argv
    assert argv[argv.index("--backend") + 1] == "pytorch"
    assert "--tp_size" in argv
    assert argv[argv.index("--tp_size") + 1] == "1"
    assert "--max_seq_len" in argv
    assert argv[argv.index("--max_seq_len") + 1] == "8192"
    assert "--max_batch_size" in argv
    assert argv[argv.index("--max_batch_size") + 1] == "8"  # plan default
    assert "--kv_cache_free_gpu_memory_fraction" in argv
    assert "--host" in argv and argv[argv.index("--host") + 1] == "0.0.0.0"
    assert "--port" in argv and argv[argv.index("--port") + 1] == "8000"
    assert "--trust_remote_code" in argv


def test_build_argv_max_batch_size_from_target_concurrency():
    argv = TRTLLMBackend().build_argv(
        _plan(target_concurrency=128), local_model_path="/m",
    )
    assert argv[argv.index("--max_batch_size") + 1] == "128"


def test_build_argv_tp_4():
    argv = TRTLLMBackend().build_argv(
        _plan(gpu_ids=[0, 1, 2, 3], tensor_parallel=4),
        local_model_path="/models/x",
    )
    assert argv[argv.index("--tp_size") + 1] == "4"


def test_container_kwargs_gpu_request():
    kw = TRTLLMBackend().container_kwargs(_plan(gpu_ids=[2, 3], tensor_parallel=2))
    assert kw["device_requests"][0]["device_ids"] == ["2", "3"]
    assert kw["ipc_mode"] == "host"
    assert kw["shm_size"] == "2g"
    assert kw["ulimits"][0].name == "memlock"


def test_default_image():
    assert TRTLLMBackend().image_default.startswith("nvcr.io/nvidia/tensorrt-llm/release:")


def test_internal_port():
    assert TRTLLMBackend().internal_port == 8000


def test_build_argv_extra_args_keyvalue():
    argv = TRTLLMBackend().build_argv(
        _plan(extra_args={"--ep_size": "2", "--max_num_tokens": "8192"}),
        local_model_path="/models/x",
    )
    assert argv[argv.index("--ep_size") + 1] == "2"
    assert argv[argv.index("--max_num_tokens") + 1] == "8192"


def test_build_argv_extra_args_bare_flag():
    argv = TRTLLMBackend().build_argv(
        _plan(extra_args={"--enable-chunked-context": ""}),
        local_model_path="/models/x",
    )
    idx = argv.index("--enable-chunked-context")
    if idx + 1 < len(argv):
        assert argv[idx + 1].startswith("--")
    assert "" not in argv
