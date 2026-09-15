# SPDX-License-Identifier: Apache-2.0
"""Stop Gemma4's next-role search when its result is already known."""

from dataclasses import replace
from functools import wraps

from vllm.renderers.hf import HfRenderer

from vllm_ascend.ascend_config import get_ascend_config

_NEXT_ROLE_SCAN = """        {%- for j in range(loop.index0 + 1, loop_messages | length) -%}
            {%- if not next_nt.found -%}
                {%- if loop_messages[j]['role'] != 'tool' -%}
                    {%- set next_nt.role = loop_messages[j]['role'] -%}
                    {%- set next_nt.found = true -%}
                {%- endif -%}
            {%- endif -%}
        {%- endfor -%}"""
_EARLY_EXIT_SCAN = _NEXT_ROLE_SCAN.replace(
    "{%- set next_nt.found = true -%}", "{%- set next_nt.found = true -%}\n                    {%- break -%}"
)


def optimize_gemma4_template(template):
    """Only transform the known, output-free role-search loop."""
    if not isinstance(template, str) or template.count(_NEXT_ROLE_SCAN) != 1:
        return template
    return template.replace(_NEXT_ROLE_SCAN, _EARLY_EXIT_SCAN)


def install_gemma4_template_patch():
    original = HfRenderer.__init__
    if getattr(original, "_ascend_gemma4_template_patch", False):
        return

    @wraps(original)
    def initialize(self, config, tokenizer):
        original(self, config, tokenizer)
        self._ascend_gemma4_template = None
        if (
            tokenizer is not None
            and config.model_config.hf_config.model_type == "gemma4"
            and get_ascend_config().gemma4_template_early_exit
        ):
            optimized = optimize_gemma4_template(tokenizer.chat_template)
            if optimized != tokenizer.chat_template:
                # Cached HF tokenizers reconstruct the original on copy. Pass
                # the template explicitly instead of mutating their copies.
                self._ascend_gemma4_template = (tokenizer.chat_template, optimized)

    def resolve_params(self, params):
        templates = self._ascend_gemma4_template
        if templates is not None and params.chat_template is None and self.tokenizer.chat_template == templates[0]:
            return replace(params, chat_template=templates[1])
        return params

    original_sync = HfRenderer.render_messages
    original_async = HfRenderer.render_messages_async

    @wraps(original_sync)
    def render_messages(self, messages, params):
        return original_sync(self, messages, resolve_params(self, params))

    @wraps(original_async)
    async def render_messages_async(self, messages, params):
        return await original_async(self, messages, resolve_params(self, params))

    initialize._ascend_gemma4_template_patch = True
    HfRenderer.__init__ = initialize
    HfRenderer.render_messages = render_messages
    HfRenderer.render_messages_async = render_messages_async


install_gemma4_template_patch()
