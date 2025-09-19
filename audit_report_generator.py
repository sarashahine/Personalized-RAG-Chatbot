#!/usr/bin/env python3
"""
Final audit report generator for the Sayed Hashem Safieddine chatbot pipeline.
Compiles all audit data into a comprehensive report with findings and recommendations.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

class AuditReportGenerator:
    """Generate comprehensive audit reports from collected data."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.report_dir = os.path.join(base_dir, "audit_report")
        self.discovery_file = os.path.join(self.report_dir, "discovery.json")
        self.static_analysis_file = os.path.join(self.report_dir, "static_analysis.json")
        self.audit_results_file = os.path.join(self.report_dir, "runs", "audit_results.json")

    def load_json_file(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load JSON file safely."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def generate_findings(self, discovery: Dict, static: Dict, runtime: Dict) -> Dict[str, Any]:
        """Generate audit findings from all data sources."""

        findings = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "info": []
        }

        # Analyze diacritics coverage
        if runtime and "summary" in runtime:
            summary = runtime["summary"]
            diacritics_coverage = summary.get("avg_diacritics_coverage", 0)
            pass_rate = summary.get("overall_pass_rate", 0)

            if diacritics_coverage < 0.95:
                findings["critical"].append({
                    "title": "Insufficient Arabic Diacritization Coverage",
                    "description": ".2f",
                    "impact": "Final answers may not meet the 95% diacritization requirement",
                    "recommendation": "Enhance ArabicDiacritizer class and persona lexicon application"
                })

            if pass_rate < 0.8:
                findings["high"].append({
                    "title": "Low Test Pass Rate",
                    "description": ".1f",
                    "impact": "Pipeline reliability concerns for production deployment",
                    "recommendation": "Review and fix failing test cases, improve error handling"
                })

        # Analyze code complexity
        if static and "complexity_analysis" in static:
            complexity = static["complexity_analysis"]
            high_complexity_functions = [
                func for func in complexity.get("functions", [])
                if func.get("complexity", 0) > 15
            ]

            if high_complexity_functions:
                findings["medium"].append({
                    "title": "High Code Complexity Detected",
                    "description": f"{len(high_complexity_functions)} functions with complexity > 15",
                    "impact": "Increased maintenance burden and potential bug introduction",
                    "recommendation": "Refactor complex functions into smaller, more manageable units"
                })

        # Analyze pipeline components
        if discovery and "pipeline_components" in discovery:
            components = discovery["pipeline_components"]
            missing_components = []

            expected_agents = ["Planner", "Classifier", "Reformulator", "Answer Generator", "Persona Styler", "Validator", "Monitor"]
            # Convert component names to match expected format
            found_agents = []
            for comp_name in components.keys():
                # Map prompt names to agent names
                if "PLANNER" in comp_name.upper():
                    found_agents.append("Planner")
                elif "CLASSIFIER" in comp_name.upper() or "ANSWER_GENERATION" in comp_name.upper():
                    found_agents.append("Classifier")
                elif "REFORMULATOR" in comp_name.upper():
                    found_agents.append("Reformulator")
                elif "ANSWER_GENERATION" in comp_name.upper():
                    found_agents.append("Answer Generator")
                elif "PERSONA_STYLE" in comp_name.upper():
                    found_agents.append("Persona Styler")
                elif "VALIDATOR" in comp_name.upper():
                    found_agents.append("Validator")

            for agent in expected_agents:
                if agent not in found_agents:
                    missing_components.append(agent)

            if missing_components:
                findings["high"].append({
                    "title": "Missing Pipeline Components",
                    "description": f"Missing agents: {', '.join(missing_components)}",
                    "impact": "Incomplete multi-agent architecture may affect response quality",
                    "recommendation": "Implement missing agent components or update architecture documentation"
                })

        # Check for error handling
        if static and "function_analysis" in static:
            functions = static["function_analysis"]
            functions_without_error_handling = [
                func for func in functions
                if not any(keyword in func.get("code", "").lower() for keyword in ["try:", "except", "catch"])
            ]

            if len(functions_without_error_handling) > len(functions) * 0.3:  # More than 30%
                findings["medium"].append({
                    "title": "Insufficient Error Handling",
                    "description": ".1f",
                    "impact": "Pipeline may fail unexpectedly under edge cases",
                    "recommendation": "Add comprehensive try-except blocks and error recovery mechanisms"
                })

        # Performance analysis
        if runtime and "test_results" in runtime:
            test_results = runtime["test_results"]
            slow_tests = [
                test for test in test_results
                if test["summary"]["avg_duration_ms"] > 10000  # > 10 seconds
            ]

            if slow_tests:
                findings["medium"].append({
                    "title": "Performance Issues Detected",
                    "description": f"{len(slow_tests)} test cases exceed 10-second response time",
                    "impact": "Poor user experience with slow response times",
                    "recommendation": "Optimize pipeline components, consider caching strategies, and implement response time monitoring"
                })

        return findings

    def generate_recommendations(self, findings: Dict[str, Any]) -> List[str]:
        """Generate prioritized recommendations based on findings."""

        recommendations = []

        # Critical issues first
        if findings["critical"]:
            recommendations.extend([
                "🔴 PRIORITY 1: Address all critical findings immediately before production deployment",
                "• Implement enhanced Arabic diacritization to achieve ≥95% coverage",
                "• Add comprehensive validation for diacritization requirements"
            ])

        # High priority
        if findings["high"]:
            recommendations.extend([
                "🟠 PRIORITY 2: Resolve high-priority issues for system reliability",
                "• Complete missing pipeline components",
                "• Improve test pass rates through debugging and error handling"
            ])

        # Medium priority
        if findings["medium"]:
            recommendations.extend([
                "🟡 PRIORITY 3: Address medium-priority issues for maintainability",
                "• Refactor high-complexity functions",
                "• Enhance error handling throughout the pipeline",
                "• Optimize performance bottlenecks"
            ])

        # General recommendations
        recommendations.extend([
            "📋 GENERAL RECOMMENDATIONS:",
            "• Implement continuous monitoring and alerting for diacritization coverage",
            "• Add comprehensive logging and tracing for production debugging",
            "• Create automated regression tests for critical functionality",
            "• Document all pipeline components and their interactions",
            "• Set up performance benchmarks and monitoring dashboards"
        ])

        return recommendations

    def generate_code_annotations(self, static: Dict, runtime: Dict) -> Dict[str, Any]:
        """Generate code annotations for improvement suggestions."""

        annotations = {
            "functions_to_refactor": [],
            "missing_error_handling": [],
            "performance_bottlenecks": [],
            "test_coverage_gaps": []
        }

        # Extract from static analysis
        if static and "complexity_analysis" in static:
            complex_functions = [
                func for func in static["complexity_analysis"].get("functions", [])
                if func.get("complexity", 0) > 15
            ]
            annotations["functions_to_refactor"] = [
                {
                    "function": func["name"],
                    "file": func["file"],
                    "complexity": func["complexity"],
                    "suggestion": "Break down into smaller functions with single responsibilities"
                }
                for func in complex_functions
            ]

        # Extract performance issues from runtime
        if runtime and "test_results" in runtime:
            slow_tests = [
                test for test in runtime["test_results"]
                if test["summary"]["avg_duration_ms"] > 5000  # > 5 seconds
            ]
            annotations["performance_bottlenecks"] = [
                {
                    "test_case": test["test_case"]["name"],
                    "avg_duration_ms": test["summary"]["avg_duration_ms"],
                    "suggestion": "Profile and optimize pipeline stages, consider caching"
                }
                for test in slow_tests
            ]

        return annotations

    def generate_final_report(self) -> Dict[str, Any]:
        """Generate the complete audit report."""

        # Load all data sources
        discovery = self.load_json_file(self.discovery_file)
        static = self.load_json_file(self.static_analysis_file)
        runtime = self.load_json_file(self.audit_results_file)

        # Generate components
        findings = self.generate_findings(discovery, static, runtime)
        recommendations = self.generate_recommendations(findings)
        annotations = self.generate_code_annotations(static, runtime)

        # Calculate overall scores
        overall_score = 100
        penalty_weights = {"critical": 25, "high": 15, "medium": 5, "low": 1}

        for severity, issues in findings.items():
            if severity in penalty_weights:
                overall_score -= len(issues) * penalty_weights[severity]

        overall_score = max(0, overall_score)  # Don't go below 0

        # Runtime metrics summary
        runtime_summary = {}
        if runtime and "summary" in runtime:
            rt_sum = runtime["summary"]
            runtime_summary = {
                "total_tests": rt_sum.get("total_tests", 0),
                "pass_rate": rt_sum.get("overall_pass_rate", 0),
                "avg_diacritics_coverage": rt_sum.get("avg_diacritics_coverage", 0),
                "avg_response_time_ms": sum(
                    test["summary"]["avg_duration_ms"]
                    for test in runtime.get("test_results", [])
                ) / len(runtime.get("test_results", [])) if runtime.get("test_results") else 0
            }

        # Create final report
        report = {
            "audit_metadata": {
                "timestamp": datetime.now().isoformat(),
                "auditor": "Automated QA Pipeline",
                "target_system": "Sayed Hashem Safieddine Arabic Diacritization Chatbot",
                "audit_version": "1.0",
                "overall_score": overall_score,
                "grade": "A" if overall_score >= 90 else "B" if overall_score >= 80 else "C" if overall_score >= 70 else "D" if overall_score >= 60 else "F"
            },
            "executive_summary": {
                "system_overview": "Multi-agent Arabic chatbot with GPT-4o and comprehensive diacritization pipeline",
                "audit_scope": "Code quality, Arabic NLP accuracy, performance, and reliability assessment",
                "key_findings": {
                    "total_findings": sum(len(issues) for issues in findings.values()),
                    "critical_issues": len(findings["critical"]),
                    "high_issues": len(findings["high"]),
                    "production_readiness": "Ready" if overall_score >= 80 else "Needs Work"
                },
                "runtime_metrics": runtime_summary
            },
            "detailed_findings": findings,
            "recommendations": recommendations,
            "code_annotations": annotations,
            "data_sources": {
                "discovery_available": discovery is not None,
                "static_analysis_available": static is not None,
                "runtime_tests_available": runtime is not None
            },
            "raw_data": {
                "discovery": discovery,
                "static_analysis": static,
                "runtime_results": runtime
            }
        }

        return report

    def save_report(self, report: Dict[str, Any], output_file: str = "final_audit_report.json"):
        """Save the final report to file."""
        output_path = os.path.join(self.report_dir, output_file)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"📄 Final audit report saved to: {output_path}")
        return output_path

def main():
    """Main report generation function."""
    generator = AuditReportGenerator()

    print("📊 Generating Final Audit Report")
    print("=" * 50)

    # Generate report
    report = generator.generate_final_report()

    # Save report
    output_file = generator.save_report(report)

    # Print summary
    metadata = report["audit_metadata"]
    summary = report["executive_summary"]

    print("\n🎯 AUDIT SUMMARY")
    print("=" * 50)
    print(f"Overall Score: {metadata['overall_score']}/100 ({metadata['grade']})")
    print(f"Production Readiness: {summary['key_findings']['production_readiness']}")
    print(f"Total Findings: {summary['key_findings']['total_findings']}")
    print(f"Critical Issues: {summary['key_findings']['critical_issues']}")
    print(f"High Priority Issues: {summary['key_findings']['high_issues']}")

    if "runtime_metrics" in summary and summary["runtime_metrics"]:
        rt = summary["runtime_metrics"]
        print("\n📈 RUNTIME METRICS")
        print(f"Tests Run: {rt['total_tests']}")
        print(f"Pass Rate: {rt['pass_rate']:.1%}")
        print(f"Avg Diacritics Coverage: {rt['avg_diacritics_coverage']:.2%}")
        print(f"Avg Response Time: {rt['avg_response_time_ms']:.0f}ms")

    print("\n📋 TOP RECOMMENDATIONS")
    recommendations = report["recommendations"][:5]  # First 5
    for rec in recommendations:
        print(f"• {rec}")

    print(f"\n💾 Full report available at: {output_file}")

if __name__ == '__main__':
    main()