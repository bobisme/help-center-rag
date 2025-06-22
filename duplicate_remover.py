#!/usr/bin/env python3
"""
Duplicate content remover for Epic documentation.
Removes or merges duplicate sections identified by semantic analysis.
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass

from semantic_analyzer import SemanticAnalyzer, DocumentSection


@dataclass
class MergeAction:
    """Represents an action to merge duplicate sections."""
    keep_section_id: int
    remove_section_ids: List[int]
    similarity_scores: List[float]
    estimated_savings: int  # tokens saved


class DuplicateRemover:
    """Removes duplicate content from markdown documentation."""
    
    def __init__(self, analyzer: SemanticAnalyzer):
        """Initialize with a semantic analyzer."""
        self.analyzer = analyzer
        self.merge_actions: List[MergeAction] = []
        
    def find_merge_groups(self, similarity_threshold: float = 0.95) -> List[MergeAction]:
        """Find groups of sections that should be merged together."""
        duplicates = self.analyzer.find_duplicates(threshold=similarity_threshold)
        
        # Build adjacency list of similar sections
        similar_sections: Dict[int, Set[int]] = {}
        for idx1, idx2, similarity in duplicates:
            if idx1 not in similar_sections:
                similar_sections[idx1] = set()
            if idx2 not in similar_sections:
                similar_sections[idx2] = set()
            similar_sections[idx1].add(idx2)
            similar_sections[idx2].add(idx1)
        
        # Find connected components (groups of mutually similar sections)
        visited = set()
        merge_groups = []
        
        for section_id in similar_sections:
            if section_id in visited:
                continue
            
            # BFS to find all connected sections
            group = set()
            queue = [section_id]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                
                visited.add(current)
                group.add(current)
                
                # Add all similar sections to queue
                if current in similar_sections:
                    for neighbor in similar_sections[current]:
                        if neighbor not in visited:
                            queue.append(neighbor)
            
            if len(group) > 1:
                merge_groups.append(sorted(group))
        
        # Convert groups to merge actions
        merge_actions = []
        for group in merge_groups:
            # Keep the section with the most content
            group_sections = [self.analyzer.sections[i] for i in group]
            keep_section = max(group_sections, key=lambda s: s.char_count)
            remove_sections = [s.id for s in group_sections if s.id != keep_section.id]
            
            # Calculate estimated savings
            estimated_savings = sum(self.analyzer.sections[sid].token_count for sid in remove_sections)
            
            # Get similarity scores for this group
            similarities = []
            for remove_id in remove_sections:
                sim_score = self.analyzer.similarity_matrix[keep_section.id, remove_id]
                similarities.append(sim_score)
            
            merge_action = MergeAction(
                keep_section_id=keep_section.id,
                remove_section_ids=remove_sections,
                similarity_scores=similarities,
                estimated_savings=estimated_savings
            )
            merge_actions.append(merge_action)
        
        # Sort by estimated savings (descending)
        merge_actions.sort(key=lambda x: x.estimated_savings, reverse=True)
        
        self.merge_actions = merge_actions
        print(f"Found {len(merge_actions)} merge groups")
        print(f"Total estimated savings: {sum(a.estimated_savings for a in merge_actions):,} tokens")
        
        return merge_actions
    
    def enhance_sections(self, keep_similar: bool = True) -> None:
        """Enhance sections that will be kept by incorporating unique content from duplicates."""
        for action in self.merge_actions:
            keep_section = self.analyzer.sections[action.keep_section_id]
            
            if not keep_similar:
                continue  # Skip enhancement, just remove duplicates
            
            # Collect unique content from sections being removed
            unique_content = []
            keep_lines = set(keep_section.content.strip().split('\n'))
            
            for remove_id in action.remove_section_ids:
                remove_section = self.analyzer.sections[remove_id]
                remove_lines = remove_section.content.strip().split('\n')
                
                # Find lines in remove_section that aren't in keep_section
                for line in remove_lines:
                    line = line.strip()
                    if line and line not in keep_lines and len(line) > 10:
                        unique_content.append(line)
            
            # Add unique content to keep section if any found
            if unique_content:
                enhanced_content = keep_section.content + "\n\n### Additional Information\n"
                for content in unique_content[:5]:  # Limit to avoid bloat
                    enhanced_content += f"- {content}\n"
                
                self.analyzer.sections[action.keep_section_id].content = enhanced_content
    
    def remove_duplicates(self, input_file: str, output_file: str, 
                         dry_run: bool = False) -> Dict[str, int]:
        """Remove duplicate sections from markdown file."""
        print(f"Processing {input_file} -> {output_file}")
        
        # Read original file
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Create set of section IDs to remove
        sections_to_remove = set()
        for action in self.merge_actions:
            sections_to_remove.update(action.remove_section_ids)
        
        print(f"Removing {len(sections_to_remove)} duplicate sections")
        
        if dry_run:
            stats = {
                'original_sections': len(self.analyzer.sections),
                'removed_sections': len(sections_to_remove),
                'remaining_sections': len(self.analyzer.sections) - len(sections_to_remove),
                'estimated_token_savings': sum(a.estimated_savings for a in self.merge_actions),
                'original_tokens': sum(s.token_count for s in self.analyzer.sections),
            }
            stats['final_tokens'] = stats['original_tokens'] - stats['estimated_token_savings']
            return stats
        
        # Filter out lines from sections to be removed
        output_lines = []
        current_section_id = None
        skip_section = False
        
        for line_num, line in enumerate(lines, 1):
            # Check if this line starts a new section
            heading_match = re.match(r'^(#+)\s+(.+)$', line.strip())
            
            if heading_match:
                # Find which section this heading belongs to
                current_section_id = None
                skip_section = False
                
                for section in self.analyzer.sections:
                    if section.start_line == line_num:
                        current_section_id = section.id
                        skip_section = section.id in sections_to_remove
                        break
            
            # Add line to output unless we're skipping this section
            if not skip_section:
                output_lines.append(line)
        
        # Write cleaned file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)
        
        # Calculate actual savings
        original_size = sum(len(line) for line in lines)
        new_size = sum(len(line) for line in output_lines)
        
        stats = {
            'original_sections': len(self.analyzer.sections),
            'removed_sections': len(sections_to_remove),
            'remaining_sections': len(self.analyzer.sections) - len(sections_to_remove),
            'original_chars': original_size,
            'new_chars': new_size,
            'char_savings': original_size - new_size,
            'estimated_token_savings': sum(a.estimated_savings for a in self.merge_actions),
            'original_tokens': sum(s.token_count for s in self.analyzer.sections),
        }
        stats['final_tokens'] = stats['original_tokens'] - stats['estimated_token_savings']
        
        print(f"Removed {stats['removed_sections']} sections")
        print(f"Estimated token reduction: {stats['estimated_token_savings']:,} tokens")
        print(f"Final estimated tokens: {stats['final_tokens']:,}")
        
        return stats
    
    def generate_removal_report(self, output_path: str = "duplicate_removal_report.md") -> str:
        """Generate detailed report of what will be removed."""
        report = f"""# Duplicate Removal Report

## Summary
- **Total merge groups**: {len(self.merge_actions)}
- **Sections to remove**: {sum(len(a.remove_section_ids) for a in self.merge_actions)}
- **Estimated token savings**: {sum(a.estimated_savings for a in self.merge_actions):,}

## Merge Actions (Top 20 by savings)

"""
        
        for i, action in enumerate(self.merge_actions[:20]):
            keep_section = self.analyzer.sections[action.keep_section_id]
            
            report += f"""### Merge Group {i+1}
**Keep**: Section {action.keep_section_id} - "{keep_section.title}" ({keep_section.token_count:,} tokens)

**Remove**:
"""
            
            for j, (remove_id, similarity) in enumerate(zip(action.remove_section_ids, action.similarity_scores)):
                remove_section = self.analyzer.sections[remove_id]
                report += f"- Section {remove_id}: \"{remove_section.title}\" ({remove_section.token_count:,} tokens, {similarity:.3f} similarity)\n"
            
            report += f"""
**Estimated savings**: {action.estimated_savings:,} tokens
**Lines**: {keep_section.start_line}-{keep_section.end_line}

---
"""
        
        # Add detailed analysis
        if len(self.merge_actions) > 20:
            remaining_savings = sum(a.estimated_savings for a in self.merge_actions[20:])
            report += f"""
## Additional {len(self.merge_actions) - 20} merge groups
**Additional estimated savings**: {remaining_savings:,} tokens

"""
        
        # Add section analysis
        perfect_matches = [a for a in self.merge_actions if all(s >= 0.99 for s in a.similarity_scores)]
        near_perfect = [a for a in self.merge_actions if all(s >= 0.95 for s in a.similarity_scores) and a not in perfect_matches]
        
        report += f"""## Analysis by Similarity

### Perfect Matches (≥99% similarity)
- **Count**: {len(perfect_matches)} groups
- **Token savings**: {sum(a.estimated_savings for a in perfect_matches):,}

### Near-Perfect Matches (95-99% similarity)  
- **Count**: {len(near_perfect)} groups
- **Token savings**: {sum(a.estimated_savings for a in near_perfect):,}

### All Other Matches
- **Count**: {len(self.merge_actions) - len(perfect_matches) - len(near_perfect)} groups
- **Token savings**: {sum(a.estimated_savings for a in self.merge_actions) - sum(a.estimated_savings for a in perfect_matches) - sum(a.estimated_savings for a in near_perfect):,}

## Recommendations

1. **Start with perfect matches**: Remove {len(perfect_matches)} groups with 99%+ similarity for {sum(a.estimated_savings for a in perfect_matches):,} token savings
2. **Consider near-perfect matches**: Additional {len(near_perfect)} groups for {sum(a.estimated_savings for a in near_perfect):,} more tokens  
3. **Review remaining manually**: {len(self.merge_actions) - len(perfect_matches) - len(near_perfect)} groups need human review

Total potential savings: {sum(a.estimated_savings for a in self.merge_actions):,} tokens
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Removal report saved to {output_path}")
        return report


def main():
    """Main duplicate removal pipeline."""
    parser = argparse.ArgumentParser(description="Remove duplicate content from Epic documentation")
    parser.add_argument("input_file", help="Path to markdown file to clean")
    parser.add_argument("--output", help="Output file path", 
                       default=None)
    parser.add_argument("--threshold", type=float, default=0.95,
                       help="Similarity threshold for duplicate detection")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be removed without actually doing it")
    parser.add_argument("--enhance", action="store_true",
                       help="Enhance kept sections with unique content from duplicates")
    parser.add_argument("--report-only", action="store_true",
                       help="Only generate removal report, don't process file")
    
    args = parser.parse_args()
    
    # Set default output filename
    if args.output is None:
        input_path = Path(args.input_file)
        args.output = str(input_path.parent / f"{input_path.stem}_deduplicated{input_path.suffix}")
    
    # Initialize analyzer and find duplicates
    print("Loading semantic analyzer...")
    analyzer = SemanticAnalyzer()
    analyzer.parse_markdown(args.input_file)
    analyzer.generate_embeddings()
    analyzer.calculate_similarity_matrix()
    
    # Initialize remover and find merge groups
    print(f"Finding duplicate groups (threshold={args.threshold})...")
    remover = DuplicateRemover(analyzer)
    merge_actions = remover.find_merge_groups(similarity_threshold=args.threshold)
    
    # Generate removal report
    report_path = Path(args.output).parent / "duplicate_removal_report.md"
    remover.generate_removal_report(str(report_path))
    
    if args.report_only:
        print("Report-only mode: removal report generated")
        return
    
    # Enhance sections if requested
    if args.enhance:
        print("Enhancing kept sections with unique content...")
        remover.enhance_sections(keep_similar=True)
    
    # Remove duplicates
    print(f"Removing duplicates...")
    stats = remover.remove_duplicates(args.input_file, args.output, dry_run=args.dry_run)
    
    print(f"\n=== RESULTS ===")
    if args.dry_run:
        print("DRY RUN - No files modified")
    print(f"Original sections: {stats['original_sections']}")
    print(f"Sections removed: {stats['removed_sections']}")
    print(f"Remaining sections: {stats['remaining_sections']}")
    print(f"Original tokens: {stats['original_tokens']:,}")
    print(f"Estimated final tokens: {stats['final_tokens']:,}")
    print(f"Token reduction: {stats['estimated_token_savings']:,} ({stats['estimated_token_savings']/stats['original_tokens']*100:.1f}%)")
    
    if not args.dry_run:
        print(f"Cleaned file saved to: {args.output}")


if __name__ == "__main__":
    main()