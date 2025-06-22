#!/usr/bin/env python3
"""
Semantic analyzer for Epic documentation compaction.
Identifies content overlap, duplicates, and optimization opportunities using embeddings.
"""

import argparse
import re
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tiktoken


@dataclass
class DocumentSection:
    """Represents a section of the documentation."""

    id: int
    title: str
    content: str
    level: int
    start_line: int
    end_line: int
    token_count: int
    char_count: int


class SemanticAnalyzer:
    """Analyzes Epic documentation for semantic overlap and optimization opportunities."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with sentence transformer model."""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        self.sections: List[DocumentSection] = []
        self.embeddings: Optional[np.ndarray] = None
        self.similarity_matrix: Optional[np.ndarray] = None

    def parse_markdown(self, file_path: str) -> List[DocumentSection]:
        """Parse markdown file into structured sections."""
        print(f"Parsing markdown file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        sections = []
        current_section = None
        section_id = 0

        for line_num, line in enumerate(lines, 1):
            # Detect headings
            heading_match = re.match(r"^(#+)\s+(.+)$", line.strip())

            if heading_match:
                # Save previous section
                if current_section is not None:
                    current_section.end_line = line_num - 1
                    sections.append(current_section)

                # Start new section
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                current_section = DocumentSection(
                    id=section_id,
                    title=title,
                    content="",
                    level=level,
                    start_line=line_num,
                    end_line=line_num,
                    token_count=0,
                    char_count=0,
                )
                section_id += 1
            elif current_section is not None:
                # Add content to current section
                current_section.content += line

        # Don't forget the last section
        if current_section is not None:
            current_section.end_line = len(lines)
            sections.append(current_section)

        # Calculate metrics for each section
        for section in sections:
            section.char_count = len(section.content)
            section.token_count = len(self.tokenizer.encode(section.content))

        self.sections = sections
        print(f"Parsed {len(sections)} sections")
        return sections

    def generate_embeddings(self) -> np.ndarray:
        """Generate sentence embeddings for all sections."""
        print("Generating embeddings for sections...")

        # Combine title and content for better semantic representation
        texts = []
        for section in self.sections:
            # Use title + first 500 chars of content for embedding
            text = f"{section.title}\n{section.content[:500]}"
            texts.append(text)

        embeddings = self.model.encode(texts, show_progress_bar=True)
        self.embeddings = embeddings

        print(f"Generated embeddings: {embeddings.shape}")
        return embeddings

    def calculate_similarity_matrix(self) -> np.ndarray:
        """Calculate cosine similarity matrix between all sections."""
        if self.embeddings is None:
            self.generate_embeddings()

        print("Calculating similarity matrix...")
        similarity_matrix = cosine_similarity(self.embeddings)
        self.similarity_matrix = similarity_matrix

        return similarity_matrix

    def find_duplicates(self, threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """Find pairs of sections with high similarity."""
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix()

        duplicates = []
        n = len(self.sections)

        for i in range(n):
            for j in range(i + 1, n):
                similarity = self.similarity_matrix[i, j]
                if similarity >= threshold:
                    duplicates.append((i, j, similarity))

        # Sort by similarity descending
        duplicates.sort(key=lambda x: x[2], reverse=True)

        print(f"Found {len(duplicates)} potential duplicates (threshold={threshold})")
        return duplicates

    def calculate_information_density(self) -> List[float]:
        """Calculate information density (entropy) for each section."""
        print("Calculating information density scores...")

        densities = []
        for section in self.sections:
            if not section.content.strip():
                densities.append(0.0)
                continue

            # Calculate word frequency entropy
            words = re.findall(r"\w+", section.content.lower())
            if not words:
                densities.append(0.0)
                continue

            word_counts = Counter(words)
            total_words = len(words)

            # Calculate entropy
            entropy = 0.0
            for count in word_counts.values():
                prob = count / total_words
                if prob > 0:
                    entropy -= prob * math.log2(prob)

            # Normalize by content length for density score
            density = entropy / math.log2(len(words)) if len(words) > 1 else 0.0
            densities.append(density)

        return densities

    def cluster_sections(self, n_clusters: int = 20) -> np.ndarray:
        """Cluster sections using K-means on embeddings."""
        if self.embeddings is None:
            self.generate_embeddings()

        print(f"Clustering sections into {n_clusters} clusters...")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.embeddings)

        return cluster_labels

    def analyze_large_sections(
        self, size_threshold: int = 5000
    ) -> List[DocumentSection]:
        """Identify sections larger than threshold for detailed analysis."""
        large_sections = [s for s in self.sections if s.char_count > size_threshold]
        large_sections.sort(key=lambda x: x.char_count, reverse=True)

        print(
            f"Found {len(large_sections)} sections larger than {size_threshold} chars"
        )
        return large_sections

    def generate_report(self, output_path: str = "semantic_analysis_report.md"):
        """Generate comprehensive analysis report."""
        print(f"Generating analysis report: {output_path}")

        # Calculate metrics
        duplicates = self.find_duplicates()
        densities = self.calculate_information_density()
        clusters = self.cluster_sections()
        large_sections = self.analyze_large_sections()

        # Calculate statistics
        total_tokens = sum(s.token_count for s in self.sections)
        total_chars = sum(s.char_count for s in self.sections)
        avg_similarity = np.mean(
            self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)]
        )

        report = f"""# Epic Documentation Semantic Analysis Report

## Overview
- **Total sections**: {len(self.sections)}
- **Total tokens**: {total_tokens:,}
- **Total characters**: {total_chars:,}
- **Average inter-section similarity**: {avg_similarity:.3f}

## Section Size Distribution
- **Largest section**: {max(s.char_count for s in self.sections):,} chars ({max(s.token_count for s in self.sections):,} tokens)
- **Average section size**: {total_chars // len(self.sections):,} chars ({total_tokens // len(self.sections):,} tokens)
- **Sections >5KB**: {len([s for s in self.sections if s.char_count > 5000])}
- **Sections >10KB**: {len([s for s in self.sections if s.char_count > 10000])}
- **Sections >50KB**: {len([s for s in self.sections if s.char_count > 50000])}

## Content Overlap Analysis

### High Similarity Pairs (>0.8 similarity)
"""

        for i, (idx1, idx2, sim) in enumerate(duplicates[:10]):
            s1, s2 = self.sections[idx1], self.sections[idx2]
            report += f"""
**Pair {i+1}**: Similarity {sim:.3f}
- Section {idx1}: "{s1.title}" ({s1.token_count:,} tokens)
- Section {idx2}: "{s2.title}" ({s2.token_count:,} tokens)
"""

        report += f"""
### Largest Sections (Potential Optimization Targets)
"""

        for i, section in enumerate(large_sections[:10]):
            report += f"""
**#{i+1}**: "{section.title}"
- Size: {section.char_count:,} chars ({section.token_count:,} tokens)
- Lines: {section.start_line}-{section.end_line}
- Info density: {densities[section.id]:.3f}
"""

        # Cluster analysis
        cluster_sizes = Counter(clusters)
        report += f"""
### Content Clustering
- **Number of clusters**: {len(cluster_sizes)}
- **Largest cluster**: {max(cluster_sizes.values())} sections
- **Average cluster size**: {len(self.sections) / len(cluster_sizes):.1f} sections
"""

        # Optimization recommendations
        potential_savings = 0
        for idx1, idx2, sim in duplicates:
            if sim > 0.9:  # Very high similarity
                smaller_section = min(
                    self.sections[idx1],
                    self.sections[idx2],
                    key=lambda x: x.token_count,
                )
                potential_savings += smaller_section.token_count

        report += f"""
## Optimization Opportunities

### Immediate Actions
1. **Merge near-duplicates**: {len([d for d in duplicates if d[2] > 0.9])} pairs with >90% similarity
   - Potential token savings: ~{potential_savings:,} tokens
   
2. **Condense large sections**: Focus on {len(large_sections)} sections >5KB
   - Combined size: {sum(s.token_count for s in large_sections):,} tokens ({sum(s.token_count for s in large_sections)/total_tokens*100:.1f}% of total)

3. **Low-density content**: {len([d for d in densities if d < 0.3])} sections with low information density
   - May contain repetitive or boilerplate content

### Estimated Impact
- **Conservative compression**: 15-20% token reduction
- **Aggressive optimization**: 25-35% token reduction
- **Target achievable**: {total_tokens - 2000000:,} tokens to remove for <2M token goal
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"Report saved to {output_path}")
        return report

    def visualize_similarity_matrix(self, output_path: str = "similarity_heatmap.png"):
        """Create heatmap visualization of similarity matrix."""
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix()

        plt.figure(figsize=(12, 10))

        # Sample matrix if too large for visualization
        matrix = self.similarity_matrix
        labels = [f"{i}: {s.title[:30]}..." for i, s in enumerate(self.sections)]

        if len(self.sections) > 100:
            # Sample 100 largest sections for visualization
            large_indices = sorted(
                range(len(self.sections)),
                key=lambda i: self.sections[i].char_count,
                reverse=True,
            )[:100]
            matrix = self.similarity_matrix[np.ix_(large_indices, large_indices)]
            labels = [labels[i] for i in large_indices]

        sns.heatmap(
            matrix,
            xticklabels=labels,
            yticklabels=labels,
            cmap="YlOrRd",
            center=0.5,
            square=True,
        )

        plt.title("Section Similarity Matrix")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Similarity heatmap saved to {output_path}")

    def visualize_embeddings_2d(self, output_path: str = "embeddings_2d.png"):
        """Create 2D visualization of section embeddings using PCA."""
        if self.embeddings is None:
            self.generate_embeddings()

        # Reduce to 2D using PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(self.embeddings)

        # Color by section size
        sizes = [s.char_count for s in self.sections]

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=sizes,
            cmap="viridis",
            alpha=0.6,
            s=30,
        )
        plt.colorbar(scatter, label="Section Size (chars)")
        plt.title("2D Visualization of Section Embeddings (colored by size)")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")

        # Annotate largest sections
        large_indices = sorted(
            range(len(self.sections)),
            key=lambda i: self.sections[i].char_count,
            reverse=True,
        )[:5]

        for idx in large_indices:
            plt.annotate(
                f"{idx}: {self.sections[idx].title[:20]}...",
                (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"2D embedding visualization saved to {output_path}")


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Semantic analysis of Epic documentation"
    )
    parser.add_argument("input_file", help="Path to markdown file to analyze")
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2", help="Sentence transformer model to use"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for duplicate detection",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for reports and visualizations",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Skip visualization generation"
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = SemanticAnalyzer(model_name=args.model)

    # Parse document
    sections = analyzer.parse_markdown(args.input_file)

    # Generate embeddings and similarity matrix
    analyzer.generate_embeddings()
    analyzer.calculate_similarity_matrix()

    # Generate comprehensive report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    report_path = output_dir / "semantic_analysis_report.md"
    analyzer.generate_report(str(report_path))

    # Generate visualizations unless disabled
    if not args.no_viz:
        heatmap_path = output_dir / "similarity_heatmap.png"
        embeddings_path = output_dir / "embeddings_2d.png"

        analyzer.visualize_similarity_matrix(str(heatmap_path))
        analyzer.visualize_embeddings_2d(str(embeddings_path))

    # Print key findings
    duplicates = analyzer.find_duplicates(threshold=args.threshold)
    large_sections = analyzer.analyze_large_sections()

    print(f"\n=== KEY FINDINGS ===")
    print(f"Total sections: {len(sections)}")
    print(f"Potential duplicates (>{args.threshold} similarity): {len(duplicates)}")
    print(f"Large sections (>5KB): {len(large_sections)}")
    print(f"Largest section: {max(s.char_count for s in sections):,} chars")
    print(f"Total tokens: {sum(s.token_count for s in sections):,}")

    if duplicates:
        print(f"\nTop duplicate pairs:")
        for i, (idx1, idx2, sim) in enumerate(duplicates[:3]):
            s1, s2 = sections[idx1], sections[idx2]
            print(f"  {i+1}. {sim:.3f}: '{s1.title[:40]}...' vs '{s2.title[:40]}...'")


if __name__ == "__main__":
    main()

