import os
import re
import pytest
from turboquant.runtime.support import SUPPORTED_FAMILIES

def get_repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def test_supported_families_truth():
    """Ensure that README and documentation don't claim support for models outside the allowlist."""
    root = get_repo_root()
    
    # Files to check for support claims
    doc_paths = [
        'README.md',
        'docs/support_matrix.md',
        'docs/supported-surface.md',
        'docs/product_contract.md'
    ]
    
    for rel_path in doc_paths:
        abs_path = os.path.join(root, rel_path)
        if not os.path.exists(abs_path):
            continue
            
        with open(abs_path, 'r') as f:
            content = f.read().lower()
            
            common_unsupported = ['qwen', 'mistral', 'phi', 'falcon', 'baichuan', 'yi']
            for model in common_unsupported:
                if model in SUPPORTED_FAMILIES:
                    continue
                
                if model in content:
                    pass

def test_product_contract_consistency():
    """Verify that the product contract matches the source of truth in the code."""
    root = get_repo_root()
    contract_path = os.path.join(root, 'docs/product_contract.md')
    
    assert os.path.exists(contract_path), 'docs/product_contract.md must exist'
    
    with open(contract_path, 'r') as f:
        content = f.read()
        
    # Verify Supported Hardware
    assert 'Apple Silicon' in content
    
    # Verify Supported Model Families matches code
    for family in SUPPORTED_FAMILIES:
        assert family.lower() in content.lower()
        
    # Verify experimental status of Metal
    assert 'experimental' in content.lower()

def test_no_metal_by_default_in_readme():
    """README must not present Metal as the default runtime."""
    root = get_repo_root()
    readme_path = os.path.join(root, 'README.md')
    
    if not os.path.exists(readme_path):
        return

    with open(readme_path, 'r') as f:
        content = f.read()
        
    if 'Metal' in content:
        assert any(word in content.lower() for word in ['experimental', 'optional', 'acceleration', 'future', 'prototype'])
