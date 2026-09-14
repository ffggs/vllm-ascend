from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.compilation.passes.qknorm_rope_fusion_pass as fusion_module


def _make_config():
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            splitting_ops=None,
            use_inductor_graph_partition=False,
            pass_config=SimpleNamespace(),
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            architectures=["Gemma4ForConditionalGeneration"],
            hf_text_config=SimpleNamespace(attention_bias=False),
        ),
        device_config=None,
    )


def test_qknorm_rope_pass_registers_all_gemma4_attention_shapes(monkeypatch):
    registrations = []

    class RecordingPattern:
        def __init__(self, **kwargs):
            registrations.append((self.__class__.__name__, kwargs))

        def register(self, pattern_match_passes):
            pass

    full_attention = SimpleNamespace(head_size=512, num_heads=16, num_kv_heads=2)
    sliding_attention = SimpleNamespace(head_size=256, num_heads=16, num_kv_heads=8)
    monkeypatch.setattr(
        fusion_module,
        "get_layers_from_vllm_config",
        lambda config, layer_type: {
            "model.layers.0.self_attn.attn": sliding_attention,
            "model.layers.1.self_attn.attn": full_attention,
            "model.layers.2.self_attn.attn": sliding_attention,
        },
    )
    monkeypatch.setattr(fusion_module, "QKNormRopeFusionPattern", RecordingPattern)
    monkeypatch.setattr(fusion_module, "QKNormRopeFusionPatternWithBias", RecordingPattern)

    fusion_module.QKNormRopeFusionPass(_make_config())

    assert len(registrations) == 4
    assert {kwargs["head_dim"] for _, kwargs in registrations} == {256, 512}
    assert {(kwargs["head_dim"], kwargs["rope_dim"]) for _, kwargs in registrations} == {
        (256, 256),
        (512, 512),
    }
    assert {kwargs["num_heads"] for _, kwargs in registrations} == {16}
    assert {kwargs["num_kv_heads"] for _, kwargs in registrations} == {2, 8}


def test_qknorm_rope_pass_skips_bias_patterns_for_biasless_gemma4(monkeypatch):
    registered_pattern_types = []

    class RecordingPattern:
        def __init__(self, **kwargs):
            registered_pattern_types.append(self.__class__.__name__)

        def register(self, pattern_match_passes):
            pass

    monkeypatch.setattr(
        fusion_module,
        "get_layers_from_vllm_config",
        lambda config, layer_type: {
            "model.layers.0.self_attn.attn": SimpleNamespace(
                head_size=256,
                num_heads=16,
                num_kv_heads=8,
            )
        },
    )
    monkeypatch.setattr(fusion_module, "QKNormRopeFusionPattern", RecordingPattern)
    monkeypatch.setattr(fusion_module, "QKNormRopeFusionPatternWithBias", RecordingPattern)

    fusion_module.QKNormRopeFusionPass(_make_config())

    assert registered_pattern_types == ["RecordingPattern", "RecordingPattern"]


def test_qknorm_rope_patterns_have_shape_specific_registration_ids():
    config = _make_config()
    sliding = fusion_module.QKNormRopeFusionPattern(
        vllm_config=config,
        head_dim=256,
        rope_dim=256,
        num_heads=16,
        num_kv_heads=8,
    )
    full = fusion_module.QKNormRopeFusionPattern(
        vllm_config=config,
        head_dim=512,
        rope_dim=512,
        num_heads=16,
        num_kv_heads=2,
    )

    assert sliding.get_pattern_id() != full.get_pattern_id()


@pytest.mark.parametrize(
    "pattern_cls",
    [fusion_module.QKNormRopeFusionPattern, fusion_module.QKNormRopeFusionPatternWithBias],
)
def test_qknorm_rope_pattern_rejects_mismatched_qkv_width(pattern_cls):
    pattern = pattern_cls(
        vllm_config=_make_config(),
        head_dim=256,
        rope_dim=256,
        num_heads=16,
        num_kv_heads=8,
    )

    with pytest.raises(RuntimeError, match="does not match pattern size"):
        pattern._check_qkv_size(torch.empty(1, 10240))
