from serve_engine.backends.vllm import VLLMBackend
from serve_engine.lifecycle.plan import DeploymentPlan


def _plan(**overrides):
    base = dict(
        model_name="llama-1b",
        hf_repo="meta-llama/Llama-3.2-1B-Instruct",
        revision="main",
        backend="vllm",
        image_tag="vllm/vllm-openai:v0.7.3",
        gpu_ids=[0],
        max_model_len=8192,
    )
    base.update(overrides)
    return DeploymentPlan(**base)


def test_build_argv_minimum():
    argv = VLLMBackend().build_argv(_plan(), local_model_path="/models/llama-1b")
    assert argv[0] == "--model"
    assert argv[1] == "/models/llama-1b"
    assert "--tensor-parallel-size" in argv
    assert argv[argv.index("--tensor-parallel-size") + 1] == "1"
    assert "--max-model-len" in argv
    assert argv[argv.index("--max-model-len") + 1] == "8192"
    assert "--max-num-seqs" in argv
    assert argv[argv.index("--max-num-seqs") + 1] == "8"  # plan default
    assert "--enable-prefix-caching" in argv
    assert "--enable-chunked-prefill" in argv
    assert "--host" in argv and argv[argv.index("--host") + 1] == "0.0.0.0"
    assert "--port" in argv and argv[argv.index("--port") + 1] == "8000"


def test_build_argv_max_num_seqs_from_target_concurrency():
    argv = VLLMBackend().build_argv(
        _plan(target_concurrency=64), local_model_path="/m",
    )
    assert argv[argv.index("--max-num-seqs") + 1] == "64"


def test_build_argv_tp_4():
    argv = VLLMBackend().build_argv(
        _plan(gpu_ids=[0, 1, 2, 3], tensor_parallel=4),
        local_model_path="/models/x",
    )
    assert argv[argv.index("--tensor-parallel-size") + 1] == "4"


def test_container_kwargs_gpu_request():
    kw = VLLMBackend().container_kwargs(_plan(gpu_ids=[2, 3], tensor_parallel=2))
    assert kw["device_requests"][0]["device_ids"] == ["2", "3"]
    assert kw["ipc_mode"] == "host"
    assert kw["shm_size"] == "2g"
    assert kw["ulimits"][0].name == "memlock"


def test_default_image():
    assert VLLMBackend().image_default.startswith("vllm/vllm-openai:")


def test_build_argv_extra_args_keyvalue():
    argv = VLLMBackend().build_argv(
        _plan(extra_args={"--kv-cache-dtype": "fp8_e4m3", "--reasoning-parser": "qwen3"}),
        local_model_path="/models/x",
    )
    assert argv[argv.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
    assert argv[argv.index("--reasoning-parser") + 1] == "qwen3"


def test_build_argv_extra_args_bare_flag():
    # Empty value = bare boolean flag, no following value token.
    argv = VLLMBackend().build_argv(
        _plan(extra_args={"--enable-expert-parallel": ""}),
        local_model_path="/models/x",
    )
    idx = argv.index("--enable-expert-parallel")
    # Next token (if any) must be another flag, not an empty string.
    if idx + 1 < len(argv):
        assert argv[idx + 1].startswith("--")
    assert "" not in argv
