#!/usr/bin/env python3

"""
Analyze markdown file for redundant content patterns to inform condensation strategy.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import argparse


def analyze_repetitive_patterns(content: str) -> Dict[str, int]:
    """Find repetitive text patterns that appear frequently."""

    patterns = {
        # Navigation/UI patterns
        "click_patterns": len(
            re.findall(r"Click\s+\w+\s+to\s+\w+", content, re.IGNORECASE)
        ),
        "screen_displays": len(
            re.findall(r"The\s+\w+\s+screen\s+displays", content, re.IGNORECASE)
        ),
        "dialog_patterns": len(
            re.findall(
                r"dialog\s+(box\s+)?(displays|opens|closes)", content, re.IGNORECASE
            )
        ),
        "access_patterns": len(
            re.findall(r"To\s+access\s+.*?:", content, re.IGNORECASE)
        ),
        # Common Epic-specific patterns
        "epic_help_file": len(
            re.findall(r"Applied Epic.*?Help File", content, re.IGNORECASE)
        ),
        "full_context": len(
            re.findall(
                r"Click here to see this page in full context", content, re.IGNORECASE
            )
        ),
        "learn_more": len(re.findall(r"To learn more about", content, re.IGNORECASE)),
        "launch_video": len(re.findall(r"Launch video", content, re.IGNORECASE)),
        # Redundant instructions
        "use_following": len(
            re.findall(
                r"Use the following (steps|information) to", content, re.IGNORECASE
            )
        ),
        "when_finished": len(
            re.findall(
                r"When (you are|you have) (finished|completed)", content, re.IGNORECASE
            )
        ),
        "system_displays": len(
            re.findall(r"The system displays", content, re.IGNORECASE)
        ),
        # Low-value descriptions
        "this_screen": len(
            re.findall(
                r"This (screen|window|dialog|area) (displays|shows|allows)",
                content,
                re.IGNORECASE,
            )
        ),
        "purpose_of": len(re.findall(r"The purpose of this", content, re.IGNORECASE)),
        "when_you_click": len(
            re.findall(r"When you click.*?, the system", content, re.IGNORECASE)
        ),
        # Error patterns
        "access_denied": len(re.findall(r"Access Denied", content, re.IGNORECASE)),
        "restricted_access": len(
            re.findall(r"restricted access", content, re.IGNORECASE)
        ),
        "untitled_page": len(re.findall(r"# Untitled", content, re.IGNORECASE)),
    }

    return {k: v for k, v in patterns.items() if v > 0}


def find_common_phrases(
    content: str, min_length: int = 10, min_occurrences: int = 5
) -> List[Tuple[str, int]]:
    """Find commonly repeated phrases."""

    # Extract sentences and normalize
    sentences = re.findall(r"[.!?]+\s*([^.!?]+)", content)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= min_length]

    # Count occurrences
    phrase_counts = Counter(sentences)

    # Return phrases that occur multiple times
    return [
        (phrase, count)
        for phrase, count in phrase_counts.most_common()
        if count >= min_occurrences
    ]


def analyze_headings(content: str) -> Dict[str, int]:
    """Analyze heading patterns and redundancy."""

    headings = re.findall(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE)

    analysis = {
        "total_headings": len(headings),
        "h1_count": len([h for h in headings if len(h[0]) == 1]),
        "h2_count": len([h for h in headings if len(h[0]) == 2]),
        "h3_count": len([h for h in headings if len(h[0]) == 3]),
        "h4_plus_count": len([h for h in headings if len(h[0]) >= 4]),
    }

    # Find duplicate heading text
    heading_text = [h[1] for h in headings]
    duplicates = Counter(heading_text)
    analysis["duplicate_headings"] = {
        text: count for text, count in duplicates.items() if count > 1
    }

    return analysis


def analyze_list_patterns(content: str) -> Dict[str, int]:
    """Analyze list formatting and potential issues."""

    patterns = {
        "numbered_lists": len(re.findall(r"^\d+\.\s+", content, re.MULTILINE)),
        "bullet_lists": len(re.findall(r"^\s*[-*]\s+", content, re.MULTILINE)),
        "broken_images": len(re.findall(r"^\s*!\s*$", content, re.MULTILINE)),
        "lone_exclamations": len(re.findall(r"^\s*!\s*$", content, re.MULTILINE)),
        "empty_lines": len(re.findall(r"^\s*$", content, re.MULTILINE)),
    }

    return patterns


def analyze_section_sizes(content: str) -> Dict[str, List[int]]:
    """Analyze section sizes to identify very long sections."""

    # Split by main headings (##)
    sections = re.split(r"\n(?=## )", content)

    section_analysis = {
        "section_count": len(sections),
        "section_lengths": [len(section) for section in sections],
        "very_long_sections": len(
            [s for s in sections if len(s) > 5000]
        ),  # >5KB sections
        "short_sections": len(
            [s for s in sections if len(s) < 200]
        ),  # <200 char sections
    }

    # Find sections with excessive numbered steps
    excessive_steps = []
    for i, section in enumerate(sections):
        step_count = len(re.findall(r"^\d+\.\s+", section, re.MULTILINE))
        if step_count > 15:  # More than 15 steps
            excessive_steps.append((i, step_count))

    section_analysis["excessive_step_sections"] = excessive_steps

    return section_analysis


def estimate_token_savings(patterns: Dict[str, int]) -> Dict[str, int]:
    """Estimate potential token savings from removing patterns."""

    # Rough estimates based on pattern analysis
    savings_estimates = {
        "epic_help_file": patterns.get("epic_help_file", 0)
        * 15,  # ~15 tokens per occurrence
        "full_context": patterns.get("full_context", 0) * 20,
        "learn_more": patterns.get("learn_more", 0) * 10,
        "launch_video": patterns.get("launch_video", 0) * 8,
        "click_patterns": patterns.get("click_patterns", 0) * 8,
        "screen_displays": patterns.get("screen_displays", 0) * 12,
        "dialog_patterns": patterns.get("dialog_patterns", 0) * 10,
        "this_screen": patterns.get("this_screen", 0) * 15,
        "when_finished": patterns.get("when_finished", 0) * 12,
        "system_displays": patterns.get("system_displays", 0) * 8,
        "access_denied": patterns.get("access_denied", 0) * 50,  # Error pages are long
    }

    return {k: v for k, v in savings_estimates.items() if v > 0}


def main():
    parser = argparse.ArgumentParser(
        description="Analyze markdown for redundant content"
    )
    parser.add_argument("file", help="Markdown file to analyze")
    parser.add_argument(
        "--phrases", action="store_true", help="Analyze common phrases (slower)"
    )

    args = parser.parse_args()

    print(f"Analyzing {args.file} for redundant content patterns...")
    print("=" * 60)

    with open(args.file, "r", encoding="utf-8") as f:
        content = f.read()

    file_size = len(content)
    estimated_tokens = file_size // 3.2  # Rough estimate

    print(f"File size: {file_size:,} chars (~{estimated_tokens:,.0f} tokens)")
    print()

    # Analyze repetitive patterns
    print("REPETITIVE PATTERNS:")
    print("-" * 30)
    patterns = analyze_repetitive_patterns(content)
    for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"{pattern.replace('_', ' ').title()}: {count:,} occurrences")

    print()

    # Analyze headings
    print("HEADING ANALYSIS:")
    print("-" * 30)
    heading_analysis = analyze_headings(content)
    for key, value in heading_analysis.items():
        if key != "duplicate_headings":
            print(f"{key.replace('_', ' ').title()}: {value:,}")

    if heading_analysis["duplicate_headings"]:
        print("\nDuplicate headings:")
        for heading, count in heading_analysis["duplicate_headings"].items():
            print(f"  '{heading}': {count} times")

    print()

    # Analyze lists and formatting
    print("LIST & FORMATTING ANALYSIS:")
    print("-" * 30)
    list_analysis = analyze_list_patterns(content)
    for pattern, count in list_analysis.items():
        print(f"{pattern.replace('_', ' ').title()}: {count:,}")

    print()

    # Analyze section sizes
    print("SECTION SIZE ANALYSIS:")
    print("-" * 30)
    section_analysis = analyze_section_sizes(content)
    for key, value in section_analysis.items():
        if key not in ["section_lengths", "excessive_step_sections"]:
            print(f"{key.replace('_', ' ').title()}: {value:,}")

    if section_analysis["section_lengths"]:
        avg_length = sum(section_analysis["section_lengths"]) // len(
            section_analysis["section_lengths"]
        )
        max_length = max(section_analysis["section_lengths"])
        print(f"Average section length: {avg_length:,} chars")
        print(f"Largest section: {max_length:,} chars")

    if section_analysis["excessive_step_sections"]:
        print(f"\nSections with >15 steps:")
        for section_idx, step_count in section_analysis["excessive_step_sections"]:
            print(f"  Section {section_idx}: {step_count} steps")

    print()

    # Estimate potential savings
    print("ESTIMATED TOKEN SAVINGS:")
    print("-" * 30)
    savings = estimate_token_savings(patterns)
    total_savings = sum(savings.values())

    for pattern, tokens in sorted(savings.items(), key=lambda x: x[1], reverse=True):
        percentage = (tokens / estimated_tokens) * 100
        print(
            f"{pattern.replace('_', ' ').title()}: ~{tokens:,} tokens ({percentage:.1f}%)"
        )

    print(
        f"\nTotal estimated savings: ~{total_savings:,} tokens ({(total_savings/estimated_tokens)*100:.1f}%)"
    )
    print(
        f"Estimated file size after cleanup: ~{estimated_tokens - total_savings:,.0f} tokens"
    )

    # Common phrases analysis (optional)
    if args.phrases:
        print()
        print("COMMON PHRASES (>5 occurrences):")
        print("-" * 30)
        common_phrases = find_common_phrases(content)
        for phrase, count in common_phrases[:10]:  # Top 10
            print(f"{count}x: {phrase[:80]}...")


if __name__ == "__main__":
    main()

