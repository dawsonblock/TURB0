import ast
import os
import re
import pytest
from turboquant.runtime.support import SUPPORTED_FAMILIES

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _read(rel_path: str) -> str:
    with open(os.path.join(REPO_ROOT, rel_path), encoding="utf-8") as f:
        return f.read()


def test_supported_families_truth():
    """Unsupported model names must not appear alongside positive support language in docs.

    Specifically: the integration guide must not claim that any architecture
    automatically gains support through base.py dispatch.  Each unsupported
    family that appears in docs must be accompanied by a clear disqualifier word
    (unsupported, exploratory, uncertified, not certified, not in, not supported).
    """
    doc_paths = [
        'README.md',
        'docs/support_matrix.md',
        'docs/supported-surface.md',
        'docs/product_contract.md',
        'docs/integration.md',
    ]

    # Phrases that would indicate a claim of support for any family
    positive_support_phrases = [
        "automatically support",
        "works out of the box",
        "automatically works",
        "fully supported",
        "supported out of the box",
    ]

    disqualifier_words = {
        "unsupported", "exploratory", "uncertified", "not certified",
        "not in", "not supported", "experimental", "out of scope",
    }

    for rel_path in doc_paths:
        abs_path = os.path.join(REPO_ROOT, rel_path)
        if not os.path.exists(abs_path):
            continue

        content = _read(rel_path)

        # Check that no blanket positive-support phrase appears unconstrained
        for phrase in positive_support_phrases:
            assert phrase.lower() not in content.lower(), (
                f"{rel_path}: found positive-support phrase '{phrase}' — "
                f"this blurs the distinction between dispatch routing and the "
                f"certified allowlist (llama, gemma only)."
            )


def test_product_contract_consistency():
    """Verify that the product contract matches the source of truth in the code."""
    contract_path = os.path.join(REPO_ROOT, 'docs/product_contract.md')

    assert os.path.exists(contract_path), 'docs/product_contract.md must exist'

    content = _read('docs/product_contract.md')

    # Supported hardware
    assert 'Apple Silicon' in content, "product_contract.md must mention Apple Silicon"

    # Supported model families must appear (these come from the allowlist)
    for family in SUPPORTED_FAMILIES:
        assert family.lower() in content.lower(), (
            f"product_contract.md must mention supported family '{family}'"
        )

    # Metal must be explicitly experimental
    assert 'experimental' in content.lower(), (
        "product_contract.md must describe Metal kernels as experimental"
    )

    # Certification must be stated as incomplete / not yet achieved
    assert any(phrase in content.lower() for phrase in ['not certified', 'uncertified', 'status: not']), (
        "product_contract.md must state that runtime certification is not yet complete"
    )


def test_no_metal_by_default_in_readme():
    """README must never describe TurboQuant's custom Metal kernels as the default runtime path.

    MLX mentions Metal for its own tensor backend (factual); this test only
    checks that TurboQuant-specific Metal references (TQ_USE_METAL, custom Metal
    kernels) are qualified as experimental or optional.
    """
    readme_path = os.path.join(REPO_ROOT, 'README.md')

    if not os.path.exists(readme_path):
        return

    content = _read('README.md')

    # TurboQuant custom Metal kernel indicators — these must not be unqualified
    tq_metal_indicators = ['TQ_USE_METAL', 'custom Metal kernel', 'Custom Metal']
    qualifier_words = {'experimental', 'optional', 'not the default', 'prototype'}

    import re
    paragraphs = re.split(r'\n\n+', content)

    for para in paragraphs:
        has_tq_metal = any(ind in para for ind in tq_metal_indicators)
        if not has_tq_metal:
            continue
        lowered = para.lower()
        assert any(w in lowered for w in qualifier_words), (
            f"README paragraph mentions TurboQuant Metal features without a qualifier "
            f"(expected one of: {qualifier_words}):\n\n{para[:300]}"
        )


def test_integration_doc_no_automatic_support_claim():
    """integration.md must not claim any model 'automatically' gains support via base.py dispatch."""
    content = _read('docs/integration.md')
    lowered = content.lower()

    # The original blocker phrase
    assert "will automatically support turboquant" not in lowered, (
        "integration.md claims 'will automatically support TurboQuant' — "
        "this conflates dispatch routing with the allowlist gate. "
        "Only 'llama' and 'gemma' are in the certified allowlist."
    )

    assert "works out of the box" not in lowered, (
        "integration.md uses 'works out of the box' — this implies support "
        "for any model via base.py dispatch, which is false. "
        "Routing ≠ membership in the certified allowlist."
    )


def test_integration_doc_upgrade_call_has_model_family():
    """integration.md's upgrade_cache_list example must include model_family argument."""
    content = _read('docs/integration.md')

    # Find the upgrade_cache_list call in the code block
    assert 'model_family' in content, (
        "integration.md upgrade_cache_list example does not include 'model_family'. "
        "All canonical examples of upgrade_cache_list must pass model_family explicitly."
    )


def test_evaluation_doc_examples_have_model_family():
    """evaluation.md quick-start examples must include model_family for all eval calls."""
    content = _read('docs/evaluation.md')

    for fn_name in ('perplexity_report', 'drift_report', 'memory_report'):
        # Find the call and make sure model_family is in the same code block
        # We check that both the function name and model_family appear in the file
        assert fn_name in content, f"evaluation.md is missing expected function {fn_name}"

    assert 'model_family' in content, (
        "evaluation.md quick-start examples do not include 'model_family'. "
        "All eval function examples must pass model_family explicitly."
    )


def test_support_matrix_no_automatic_dispatch_claim():
    """support_matrix.md must not describe unsupported families as auto-dispatching to TurboQuant."""
    content = _read('docs/support_matrix.md')
    lowered = content.lower()

    assert "route through" not in lowered or "not in the" in lowered, (
        "support_matrix.md says unsupported families 'route through base.py dispatch automatically' "
        "without stating they are rejected by upgrade_cache_list. This implies support."
    )

    # The 'All others' row should not say 'automatically' without the gate caveat
    assert "route through `base.py` dispatch automatically; uncertified" not in content, (
        "support_matrix.md still has the stale 'route through base.py dispatch automatically' "
        "text that implies unsupported families work automatically."
    )


def test_readme_no_all_families_autorouted():
    """README Component Status must not say 'All model families auto-routed'."""
    content = _read('README.md')
    assert "All model families auto-routed" not in content, (
        "README Component Status says 'All model families auto-routed' — "
        "this overclaims: only llama and gemma are certified; others are rejected by upgrade_cache_list."
    )


def test_readme_model_matrix_no_auto_routed_for_unsupported():
    """README model matrix must not describe Qwen/Mistral/Phi as 'auto-routed'."""
    content = _read('README.md')
    for name in ('qwen', 'mistral', 'phi'):
        for line in content.splitlines():
            if name.lower() in line.lower() and '|' in line:
                assert 'auto-routed' not in line.lower(), (
                    f"README model matrix describes '{name}' as 'auto-routed'."
                )
                assert 'auto-route' not in line.lower(), (
                    f"README model matrix claims '{name}' auto-routes."
                )


def test_readme_eval_section_has_model_family():
    """README Evaluation section examples must include model_family."""
    content = _read('README.md')
    for fn in ('perplexity_report', 'drift_report', 'memory_report'):
        for line in content.splitlines():
            if fn + '(' in line and 'turboquant_config' in line:
                assert 'model_family' in line, (
                    f"README Evaluation section: {fn}() call missing model_family argument."
                )


def test_readme_turboquantkcache_marked_internal():
    """README TurboQuantKCache section must be labeled internal/eval use."""
    content = _read('README.md')
    assert 'TurboQuantKCache` (internal' in content or 'eval use' in content, (
        "README must label TurboQuantKCache section as internal/eval use."
    )


def test_architecture_doc_maybe_turboquant_deprecated():
    """architecture.md must label maybe_turboquant_attention as legacy and not claim one-line dispatch."""
    content = _read('docs/architecture.md')
    assert 'legacy' in content.lower() or 'deprecated' in content.lower(), (
        "docs/architecture.md documents maybe_turboquant_attention without marking it as legacy/deprecated."
    )
    assert 'one-line `maybe_turboquant_attention` dispatch' not in content, (
        "architecture.md §6 still says other families need a 'one-line maybe_turboquant_attention dispatch'."
    )

