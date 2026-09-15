# SPDX-License-Identifier: Apache-2.0

import asyncio
import itertools
from types import SimpleNamespace

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment
from vllm.renderers.params import ChatParams

from vllm_ascend.patch.platform import patch_gemma4_template
from vllm_ascend.patch.platform.patch_gemma4_template import _NEXT_ROLE_SCAN, optimize_gemma4_template


def role_template():
    return (
        "{% for message in loop_messages %}"
        "{% set next_nt = namespace(role=None, found=false) %}"
        + _NEXT_ROLE_SCAN
        + "{{ next_nt.role }}:{{ next_nt.found }};{% endfor %}"
    )


def test_next_role_matches_with_tool_runs_and_assistant_continuations():
    env = ImmutableSandboxedEnvironment(extensions=["jinja2.ext.loopcontrols"])
    original = env.from_string(role_template())
    optimized = env.from_string(optimize_gemma4_template(role_template()))
    for length in range(6):
        for roles in itertools.product(("user", "assistant", "tool"), repeat=length):
            messages = [{"role": role} for role in roles]
            assert optimized.render(loop_messages=messages) == original.render(loop_messages=messages)


def test_resolved_role_stops_visiting_the_remaining_suffix():
    class CountingEnvironment(ImmutableSandboxedEnvironment):
        checks = 0

        def getattr(self, obj, attribute):
            if attribute == "found":
                self.checks += 1
            return super().getattr(obj, attribute)

    messages = [{"role": "user"}, {"role": "assistant"}] * 40
    env = CountingEnvironment(extensions=["jinja2.ext.loopcontrols"])
    original = env.from_string(role_template()).render(loop_messages=messages)
    before = env.checks
    env.checks = 0
    optimized = env.from_string(optimize_gemma4_template(role_template())).render(loop_messages=messages)
    assert optimized == original
    assert env.checks < before / 10


def test_unknown_or_already_optimized_template_is_unchanged():
    template = role_template()
    unknown = template.replace("next_nt.found", "next_message.found")
    assert optimize_gemma4_template(unknown) == unknown
    assert optimize_gemma4_template(template + template) == template + template
    optimized = optimize_gemma4_template(template)
    assert optimized != template
    assert optimize_gemma4_template(optimized) == optimized
    assert optimize_gemma4_template(None) is None
    templates = {"default": template}
    assert optimize_gemma4_template(templates) is templates


@pytest.mark.parametrize(
    "enabled,model_type,known_template,changes",
    [
        (True, "gemma4", True, True),
        (False, "gemma4", True, False),
        (True, "other", True, False),
        (True, "gemma4", False, False),
    ],
)
def test_activation_is_opt_in_and_isolates_shared_tokenizers(monkeypatch, enabled, model_type, known_template, changes):
    class Renderer:
        def __init__(self, config, tokenizer):
            self.tokenizer = tokenizer
            self.template_at_init = tokenizer.chat_template

        def render_messages(self, messages, params):
            return params

        async def render_messages_async(self, messages, params):
            return params

    monkeypatch.setattr(patch_gemma4_template, "HfRenderer", Renderer)
    monkeypatch.setattr(
        patch_gemma4_template,
        "get_ascend_config",
        lambda: SimpleNamespace(gemma4_template_early_exit=enabled),
    )
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)))
    template = role_template() if known_template else "{{ messages }}"
    tokenizer = SimpleNamespace(chat_template=template)
    patch_gemma4_template.install_gemma4_template_patch()
    wrapped_init = Renderer.__init__
    patch_gemma4_template.install_gemma4_template_patch()
    assert Renderer.__init__ is wrapped_init
    renderer = Renderer(config, tokenizer)
    assert tokenizer.chat_template == template
    assert renderer.tokenizer is tokenizer
    assert renderer.template_at_init == template
    params = ChatParams(chat_template_kwargs={"enable_thinking": True})
    resolved = renderer.render_messages([], params)
    assert resolved.chat_template == (optimize_gemma4_template(template) if changes else None)
    assert resolved.chat_template_kwargs == params.chat_template_kwargs
    assert params.chat_template is None
    assert asyncio.run(renderer.render_messages_async([], params)) == resolved
    for custom_template in ("explicit template", "", template):
        custom_params = ChatParams(chat_template=custom_template)
        assert renderer.render_messages([], custom_params) is custom_params
        assert asyncio.run(renderer.render_messages_async([], custom_params)) is custom_params
    tokenizer.chat_template = "updated template"
    assert renderer.render_messages([], params) is params


def test_renderer_without_tokenizer_is_unchanged(monkeypatch):
    class Renderer:
        def __init__(self, config, tokenizer):
            self.tokenizer = tokenizer

        def render_messages(self, messages, params):
            return params

        async def render_messages_async(self, messages, params):
            return params

    monkeypatch.setattr(patch_gemma4_template, "HfRenderer", Renderer)
    patch_gemma4_template.install_gemma4_template_patch()
    assert Renderer(None, None).tokenizer is None
