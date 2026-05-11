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
    assert "--enable-prefix-caching" in argv
    assert "--enable-chunked-prefill" in argv
    assert "--host" in argv and argv[argv.index("--host") + 1] == "0.0.0.0"
    assert "--port" in argv and argv[argv.index("--port") + 1] == "8000"


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
    assert VLLMBackend.image_default.startswith("vllm/vllm-openai:")
