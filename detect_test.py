import re

from utils import defect_type_detection

def extract_hunk_changes(diff_text: str) -> list[str]:
    """
    从 diff 文本中提取所有以单个 '+' 或 '-' 开头的行（不含 '+++'/'---' 文件头）。
    返回一个字符串列表，每个元素都是一条变更行。
    """
    lines = diff_text.splitlines()
    changes = []
    for ln in lines:
        # 以单独 '+' 或 '-' 开头，后面紧接空格或代码
        if re.match(r'^[+-][^+-]', ln):
            changes.append(ln)
    return changes

# Defect categories
DEFECT_TYPES = {
    'mem_leak':          'Memory Leak',
    'null_deref':        'Null Pointer Dereference',
    'out_of_bound':      'Array Out-of-Bounds Access',
    'use_after_free':    'Use-After-Free',
    'data_race':         'Concurrency Data Race',
    'logic_error':       'Logic Error / Incomplete Implementation',
    'vuln_api_use':      'Use of Vulnerable/Unsafe API',
    'resource_leak':     'Resource Leak - File/Handle',
    'assertion_fail':    'Assertion Trigger',
    'update':            'Update for Read or Other Reasons',
}