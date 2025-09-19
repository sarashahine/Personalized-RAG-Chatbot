#!/usr/bin/env python3
"""
Static code analysis for the Sayed Hashem Safieddine chatbot pipeline.
Analyzes modules, functions, complexity, and call graphs.
"""

import ast
import json
import os
import subprocess
import sys
from typing import Dict, List, Any

def analyze_python_file(file_path: str) -> Dict[str, Any]:
    """Analyze a Python file using AST and radon."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse AST
    tree = ast.parse(content)

    functions = []
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_info = {
                "name": node.name,
                "line_start": node.lineno,
                "line_end": getattr(node, 'end_lineno', node.lineno),
                "args": [arg.arg for arg in node.args.args],
                "docstring": ast.get_docstring(node) or "",
                "async": isinstance(node, ast.AsyncFunctionDef)
            }
            functions.append(func_info)
        elif isinstance(node, ast.ClassDef):
            class_info = {
                "name": node.name,
                "line_start": node.lineno,
                "methods": []
            }
            # Get methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    class_info["methods"].append({
                        "name": item.name,
                        "line_start": item.lineno,
                        "async": isinstance(item, ast.AsyncFunctionDef)
                    })
            classes.append(class_info)

    # Get complexity using radon
    try:
        result = subprocess.run(
            ['radon', 'cc', '-s', file_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(file_path) if os.path.dirname(file_path) else '.'
        )
        complexity_data = result.stdout.strip()
    except Exception as e:
        complexity_data = f"Error getting complexity: {e}"

    # Count lines
    lines_count = len(content.split('\n'))

    return {
        "file_path": file_path,
        "lines_count": lines_count,
        "functions": functions,
        "classes": classes,
        "complexity_analysis": complexity_data,
        "imports": [node.names[0].name if hasattr(node.names[0], 'name') else str(node.names[0])
                   for node in ast.walk(tree) if isinstance(node, ast.Import)],
        "from_imports": [f"{node.module}.{node.names[0].name}" if node.module else node.names[0].name
                        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    }

def analyze_module_functions(functions: List[Dict], file_path: str) -> List[Dict]:
    """Analyze individual functions with more detail."""
    detailed_functions = []

    for func in functions:
        # Read function source
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        func_lines = lines[func['line_start']-1:func['line_end']] if func['line_end'] else lines[func['line_start']-1:]
        func_source = ''.join(func_lines).strip()

        # Simple call graph analysis
        callees = []
        for line in func_lines:
            # Look for function calls (very basic)
            if '(' in line and not line.strip().startswith('def ') and not line.strip().startswith('class '):
                # Extract potential function names
                words = line.replace('(', ' ').split()
                for word in words:
                    if word.isidentifier() and len(word) > 2:
                        callees.append(word)

        detailed_func = {
            **func,
            "loc": len(func_lines),
            "source_preview": func_source[:200] + "..." if len(func_source) > 200 else func_source,
            "callees": list(set(callees)),  # Remove duplicates
            "complexity_estimate": "medium" if len(func_lines) > 20 else "low"
        }
        detailed_functions.append(detailed_func)

    return detailed_functions

def main():
    """Main analysis function."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(base_dir, 'src')

    modules_to_analyze = [
        'bot.py',
        'prompts.py',
        'arabic_utils.py',
        'shared_redis.py'
    ]

    analysis_results = {}

    for module in modules_to_analyze:
        file_path = os.path.join(src_dir, module)
        if os.path.exists(file_path):
            print(f"Analyzing {module}...")
            basic_analysis = analyze_python_file(file_path)
            detailed_functions = analyze_module_functions(basic_analysis['functions'], file_path)

            analysis_results[module] = {
                **basic_analysis,
                "detailed_functions": detailed_functions
            }

            # Save individual module analysis
            output_file = os.path.join(base_dir, 'audit_report', 'static_analysis', f'{module}.json')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_results[module], f, ensure_ascii=False, indent=2)

            # Create markdown summary
            md_file = os.path.join(base_dir, 'audit_report', 'static_analysis', f'{module}_functions.md')
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(f"# {module} - Function Analysis\n\n")
                f.write(f"**Total Lines:** {basic_analysis['lines_count']}\n\n")
                f.write(f"**Complexity Analysis:**\n```\n{basic_analysis['complexity_analysis']}\n```\n\n")

                f.write("## Functions\n\n")
                for func in detailed_functions:
                    f.write(f"### {func['name']}()\n")
                    f.write(f"- **Lines:** {func['loc']}\n")
                    f.write(f"- **Async:** {func['async']}\n")
                    f.write(f"- **Args:** {', '.join(func['args'])}\n")
                    f.write(f"- **Complexity:** {func['complexity_estimate']}\n")
                    f.write(f"- **Callees:** {', '.join(func['callees'][:5])}{'...' if len(func['callees']) > 5 else ''}\n")
                    if func['docstring']:
                        f.write(f"- **Docstring:** {func['docstring'][:100]}...\n")
                    f.write("\n")

                f.write("## Classes\n\n")
                for cls in basic_analysis['classes']:
                    f.write(f"### {cls['name']}\n")
                    f.write(f"- **Methods:** {len(cls['methods'])}\n")
                    for method in cls['methods'][:5]:
                        f.write(f"  - {method['name']}() ({'async' if method['async'] else 'sync'})\n")
                    f.write("\n")

    # Save overall analysis
    overall_file = os.path.join(base_dir, 'audit_report', 'static_analysis.json')
    with open(overall_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)

    print("Static analysis complete!")

if __name__ == '__main__':
    main()