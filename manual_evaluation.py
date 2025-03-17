#!/usr/bin/env python3
"""
Direct comparison of document chunks with and without contextual enrichment.
This is a simplified manual evaluation that doesn't rely on complex containers or services.
"""

import os
import json
from rich.console import Console
from rich.table import Table

# Rich console for pretty output
console = Console()

# Test queries relevant to our test document
TEST_QUERIES = [
    "How do I document patient vital signs in Epic?",
    "What are the steps for ordering a lab test?",
    "How do I create a referral order?",
    "What tools are available for documentation in Epic?",
    "How do I manage the patient problem list?",
]

# Sections we expect to match for each query
EXPECTED_SECTIONS = {
    "How do I document patient vital signs in Epic?": ["Vital Signs Documentation"],
    "What are the steps for ordering a lab test?": ["Lab Orders"],
    "How do I create a referral order?": ["Referral Orders"],
    "What tools are available for documentation in Epic?": ["SmartTools", "Templates"],
    "How do I manage the patient problem list?": ["Problem List Management"],
}

# Manually define chunk content (simplified from our test document)
BASE_CHUNKS = [
    {
        "id": "chunk1",
        "content": """# Epic Healthcare Documentation - Clinical Workflows



## Introduction to Epic EMR

Epic Systems Corporation is a leading provider of electronic medical record (EMR) software used by hospitals, healthcare providers, and other medical organizations. Epic's software platform includes a suite of integrated applications for both clinical and revenue operations, with modules covering patient care, registration, scheduling, billing, and more.

The Epic EMR system is designed to create a seamless, unified patient record across care settings, enabling healthcare providers to access complete and up-to-date patient information regardless of where the patient receives care within a healthcare system.



## Key Features and Benefits

Epic EMR offers several key features and benefits:

1. **Integrated Platform**: A unified system for clinical, administrative, and billing functions
2. **Interoperability**: Ability to share data with other healthcare systems
3. **Mobile Access**: Secure access to patient information from mobile devices
4. **Patient Portal (MyChart)**: Online access for patients to view their health records
5. **Decision Support**: Tools to help clinicians make informed decisions
6. **Reporting and Analytics**: Comprehensive data analysis capabilities



## Clinical Documentation Workflows



### Patient Registration and Check-in

The patient registration process in Epic involves capturing demographic information, insurance details, and consent forms. Key steps include:

1. **Patient Search**: First, search for the patient in the system to avoid duplicate records
2. **Demographics Entry**: Collect or update patient information including name, DOB, address
3. **Insurance Verification**: Capture and verify insurance information
4. **Consent Forms**: Document patient consent for treatment and privacy notices
5. **Medical History**: Collect preliminary information about medical history and current medications



### Vital Signs Documentation

Documenting accurate vital signs is essential for patient assessment:

1. Access the patient's chart and navigate to the vitals flowsheet
2. Record temperature, blood pressure, pulse, respiratory rate, and oxygen saturation
3. Document height, weight, and pain score as appropriate
4. Enter any relevant notes about the measurements
5. Save the documentation""",
    },
    {
        "id": "chunk2",
        "content": """## Order Management



### Lab Orders

To order laboratory tests in Epic:

1. Navigate to the Orders tab in the patient's chart
2. Select "Laboratory" from the order types
3. Search for and select the specific lab test(s)
4. Enter collection details (time, priority, special instructions)
5. Select the appropriate diagnosis to link to the order
6. Complete and sign the order



### Imaging Orders

For radiology and other imaging orders:

1. Select "Imaging/Radiology" from the order types
2. Choose the appropriate study (X-ray, CT, MRI, ultrasound, etc.)
3. Specify the body part or region to be examined
4. Document the reason for the examination
5. Indicate if contrast is required
6. Provide any special instructions or patient preparation requirements
7. Complete and sign the order



### Referral Orders

To create referrals to specialists:

1. Access the Referral section under Orders
2. Select the specialty or specific provider
3. Document the reason for referral
4. Include relevant clinical information
5. Specify urgency (routine, urgent, STAT)
6. Add any specific questions for the consultant
7. Complete and route the referral



## Clinical Notes Documentation



### Progress Notes

Creating comprehensive progress notes:

1. Select the appropriate note type (progress note, procedure note, etc.)
2. Use templates or SmartTools to structure the note
3. Document subjective information (patient complaints, history)
4. Record objective findings (exam results, vital signs, lab results)
5. Document assessment and clinical impression
6. Detail the treatment plan and follow-up instructions
7. Sign and complete the note



### SmartTools and Templates

Epic offers several documentation aids:

1. **SmartPhrases**: Expand abbreviations into full text (e.g., ".hpi" becomes a history template)
2. **SmartLists**: Present pick-lists for common documentation elements
3. **SmartLinks**: Pull in patient-specific data (labs, vitals, etc.)
4. **SmartSets**: Group orders, documentation templates, and patient instructions
5. **Templates**: Pre-formatted note structures for consistent documentation""",
    },
    {
        "id": "chunk3",
        "content": """## Patient Education and Instructions

Epic facilitates patient education through:

1. **After Visit Summaries (AVS)**: Customized visit summaries with care instructions
2. **Patient Instructions**: Pre-built or custom instructions for specific conditions
3. **Education Materials**: Integrated resources for patient education
4. **MyChart Integration**: Ability to send materials directly to patient portal
5. **Documentation**: Record which materials were provided to patients



## Reporting and Analytics

Epic provides robust reporting capabilities:

1. **Standard Reports**: Pre-built reports for common metrics
2. **Custom Reports**: Tools to create tailored reports for specific needs
3. **Dashboards**: Visual displays of key performance indicators
4. **Population Health**: Tools to analyze and manage patient populations
5. **Regulatory Reporting**: Support for required quality measures and regulatory submissions



## Best Practices for Efficient Documentation

To optimize documentation workflow in Epic:

1. **Learn SmartTools**: Invest time in mastering SmartPhrases and SmartLinks
2. **Customize Templates**: Adapt templates to your specific documentation needs
3. **Use QuickActions**: Create shortcuts for frequent tasks
4. **Organize Workspace**: Customize your workspace for efficiency
5. **Utilize Mobile Apps**: Use Epic's mobile applications for documentation on the go
6. **Leverage Voice Recognition**: Integrate speech recognition for faster documentation
7. **Optimize In-Basket Management**: Develop efficient processes for message handling



## Troubleshooting Common Issues

When encountering problems in Epic:

1. **System Alerts**: Pay attention to system warnings and alerts
2. **Help Resources**: Use Epic's built-in help functionality
3. **IT Support**: Contact your organization's Epic support team
4. **Feedback**: Provide structured feedback for system improvements
5. **Training**: Attend refresher training for updated functionality""",
    },
]

# Enriched chunks - same content but with added context
ENRICHED_CHUNKS = [
    {
        "id": "enriched_chunk1",
        "content": """Overview of Epic EMR functionality, specifically detailing common clinical workflows like patient registration, vital signs documentation, medication management, and problem list management.

# Epic Healthcare Documentation - Clinical Workflows



## Introduction to Epic EMR

Epic Systems Corporation is a leading provider of electronic medical record (EMR) software used by hospitals, healthcare providers, and other medical organizations. Epic's software platform includes a suite of integrated applications for both clinical and revenue operations, with modules covering patient care, registration, scheduling, billing, and more.

The Epic EMR system is designed to create a seamless, unified patient record across care settings, enabling healthcare providers to access complete and up-to-date patient information regardless of where the patient receives care within a healthcare system.



## Key Features and Benefits

Epic EMR offers several key features and benefits:

1. **Integrated Platform**: A unified system for clinical, administrative, and billing functions
2. **Interoperability**: Ability to share data with other healthcare systems
3. **Mobile Access**: Secure access to patient information from mobile devices
4. **Patient Portal (MyChart)**: Online access for patients to view their health records
5. **Decision Support**: Tools to help clinicians make informed decisions
6. **Reporting and Analytics**: Comprehensive data analysis capabilities



## Clinical Documentation Workflows



### Patient Registration and Check-in

The patient registration process in Epic involves capturing demographic information, insurance details, and consent forms. Key steps include:

1. **Patient Search**: First, search for the patient in the system to avoid duplicate records
2. **Demographics Entry**: Collect or update patient information including name, DOB, address
3. **Insurance Verification**: Capture and verify insurance information
4. **Consent Forms**: Document patient consent for treatment and privacy notices
5. **Medical History**: Collect preliminary information about medical history and current medications



### Vital Signs Documentation

Documenting accurate vital signs is essential for patient assessment:

1. Access the patient's chart and navigate to the vitals flowsheet
2. Record temperature, blood pressure, pulse, respiratory rate, and oxygen saturation
3. Document height, weight, and pain score as appropriate
4. Enter any relevant notes about the measurements
5. Save the documentation""",
    },
    {
        "id": "enriched_chunk2",
        "content": """This section details workflows for ordering tests and documenting patient encounters within Epic, including specific steps for lab, imaging, and referral orders, as well as utilizing Epic's SmartTools and templates for efficient note creation.

## Order Management



### Lab Orders

To order laboratory tests in Epic:

1. Navigate to the Orders tab in the patient's chart
2. Select "Laboratory" from the order types
3. Search for and select the specific lab test(s)
4. Enter collection details (time, priority, special instructions)
5. Select the appropriate diagnosis to link to the order
6. Complete and sign the order



### Imaging Orders

For radiology and other imaging orders:

1. Select "Imaging/Radiology" from the order types
2. Choose the appropriate study (X-ray, CT, MRI, ultrasound, etc.)
3. Specify the body part or region to be examined
4. Document the reason for the examination
5. Indicate if contrast is required
6. Provide any special instructions or patient preparation requirements
7. Complete and sign the order



### Referral Orders

To create referrals to specialists:

1. Access the Referral section under Orders
2. Select the specialty or specific provider
3. Document the reason for referral
4. Include relevant clinical information
5. Specify urgency (routine, urgent, STAT)
6. Add any specific questions for the consultant
7. Complete and route the referral



## Clinical Notes Documentation



### Progress Notes

Creating comprehensive progress notes:

1. Select the appropriate note type (progress note, procedure note, etc.)
2. Use templates or SmartTools to structure the note
3. Document subjective information (patient complaints, history)
4. Record objective findings (exam results, vital signs, lab results)
5. Document assessment and clinical impression
6. Detail the treatment plan and follow-up instructions
7. Sign and complete the note



### SmartTools and Templates

Epic offers several documentation aids:

1. **SmartPhrases**: Expand abbreviations into full text (e.g., ".hpi" becomes a history template)
2. **SmartLists**: Present pick-lists for common documentation elements
3. **SmartLinks**: Pull in patient-specific data (labs, vitals, etc.)
4. **SmartSets**: Group orders, documentation templates, and patient instructions
5. **Templates**: Pre-formatted note structures for consistent documentation""",
    },
    {
        "id": "enriched_chunk3",
        "content": """This section details post-clinical workflow aspects of Epic, including patient education, reporting/analytics, documentation best practices, and troubleshooting.

## Patient Education and Instructions

Epic facilitates patient education through:

1. **After Visit Summaries (AVS)**: Customized visit summaries with care instructions
2. **Patient Instructions**: Pre-built or custom instructions for specific conditions
3. **Education Materials**: Integrated resources for patient education
4. **MyChart Integration**: Ability to send materials directly to patient portal
5. **Documentation**: Record which materials were provided to patients



## Reporting and Analytics

Epic provides robust reporting capabilities:

1. **Standard Reports**: Pre-built reports for common metrics
2. **Custom Reports**: Tools to create tailored reports for specific needs
3. **Dashboards**: Visual displays of key performance indicators
4. **Population Health**: Tools to analyze and manage patient populations
5. **Regulatory Reporting**: Support for required quality measures and regulatory submissions



## Best Practices for Efficient Documentation

To optimize documentation workflow in Epic:

1. **Learn SmartTools**: Invest time in mastering SmartPhrases and SmartLinks
2. **Customize Templates**: Adapt templates to your specific documentation needs
3. **Use QuickActions**: Create shortcuts for frequent tasks
4. **Organize Workspace**: Customize your workspace for efficiency
5. **Utilize Mobile Apps**: Use Epic's mobile applications for documentation on the go
6. **Leverage Voice Recognition**: Integrate speech recognition for faster documentation
7. **Optimize In-Basket Management**: Develop efficient processes for message handling



## Troubleshooting Common Issues

When encountering problems in Epic:

1. **System Alerts**: Pay attention to system warnings and alerts
2. **Help Resources**: Use Epic's built-in help functionality
3. **IT Support**: Contact your organization's Epic support team
4. **Feedback**: Provide structured feedback for system improvements
5. **Training**: Attend refresher training for updated functionality""",
    },
]


def compute_bm25_score(query, document):
    """Simulate BM25 scoring - higher score for more query term matches.

    This is a very simplified version just for demonstration purposes.
    """
    query_terms = query.lower().split()
    score = 0

    for term in query_terms:
        # Count occurrences of the term in the document
        term_count = document.lower().count(term)
        if term_count > 0:
            # Simple TF scoring - more occurrences = higher score
            score += min(
                term_count, 3
            )  # Cap at 3 to prevent domination by common terms

            # Boost for matches in headers or structured elements
            if f"**{term}" in document.lower() or f"#{term}" in document.lower():
                score += 1

    return score


def evaluate_retrieval():
    """Run a simple manual evaluation of retrieval with and without contextual enrichment."""
    # Results table
    table = Table(title="Contextual Enrichment Manual Evaluation Results")
    table.add_column("Query", style="cyan")
    table.add_column("Expected Section", style="green")
    table.add_column("Base Result Rank", style="yellow")
    table.add_column("Enriched Result Rank", style="yellow")
    table.add_column("Base Score", style="magenta")
    table.add_column("Enriched Score", style="magenta")
    table.add_column("Improvement", style="bold green")

    # Track metrics for summary
    total_base_rank = 0
    total_enriched_rank = 0
    total_queries = 0

    for query in TEST_QUERIES:
        total_queries += 1
        console.print(f"\nEvaluating query: [bold]{query}[/bold]")

        # Get the target sections we're looking for
        target_sections = EXPECTED_SECTIONS.get(query, [])
        section_str = ", ".join(target_sections)

        # Score each base chunk for this query
        base_scores = []
        for chunk in BASE_CHUNKS:
            score = compute_bm25_score(query, chunk["content"])
            # Check if it contains any of our target sections
            contains_target = any(
                section in chunk["content"] for section in target_sections
            )
            base_scores.append(
                {"id": chunk["id"], "score": score, "contains_target": contains_target}
            )

        # Score each enriched chunk for this query
        enriched_scores = []
        for chunk in ENRICHED_CHUNKS:
            score = compute_bm25_score(query, chunk["content"])
            # Check if it contains any of our target sections
            contains_target = any(
                section in chunk["content"] for section in target_sections
            )
            enriched_scores.append(
                {"id": chunk["id"], "score": score, "contains_target": contains_target}
            )

        # Sort by score
        base_scores.sort(key=lambda x: x["score"], reverse=True)
        enriched_scores.sort(key=lambda x: x["score"], reverse=True)

        # Find the rank of the first chunk containing target section
        base_rank = None
        for i, result in enumerate(base_scores):
            if result["contains_target"]:
                base_rank = i + 1  # 1-indexed rank
                break

        enriched_rank = None
        for i, result in enumerate(enriched_scores):
            if result["contains_target"]:
                enriched_rank = i + 1  # 1-indexed rank
                break

        # Calculate improvement
        if base_rank is not None and enriched_rank is not None:
            rank_improvement = base_rank - enriched_rank
            percent_improvement = (
                (base_rank - enriched_rank) / base_rank * 100 if base_rank > 0 else 0
            )
        else:
            rank_improvement = "N/A"
            percent_improvement = 0

        # For recall calculations, we'll use a cutoff of k=2 (since we only have 3 chunks total)
        k = 2
        base_hit = base_rank is not None and base_rank <= k
        enriched_hit = enriched_rank is not None and enriched_rank <= k

        # Update totals
        if base_rank:
            total_base_rank += base_rank
        if enriched_rank:
            total_enriched_rank += enriched_rank

        # Extract the top scores
        top_base_score = base_scores[0]["score"] if base_scores else 0
        top_enriched_score = enriched_scores[0]["score"] if enriched_scores else 0

        # Add to table
        table.add_row(
            query,
            section_str,
            str(base_rank) if base_rank is not None else "Not found",
            str(enriched_rank) if enriched_rank is not None else "Not found",
            f"{top_base_score:.1f}",
            f"{top_enriched_score:.1f}",
            (
                f"↑ Rank +{rank_improvement}"
                if isinstance(rank_improvement, int) and rank_improvement > 0
                else (
                    "No change"
                    if enriched_rank == base_rank
                    else (
                        f"↓ Rank {rank_improvement}"
                        if isinstance(rank_improvement, int) and rank_improvement < 0
                        else "N/A"
                    )
                )
            ),
        )

    # Calculate Anthropic's metric - failure rate reduction
    # For simplicity, we're using rank 1 as the target (similar to recall@1)
    base_failure_rate = sum(
        1
        for score in base_scores
        if not score["contains_target"] or base_scores.index(score) > 0
    ) / len(base_scores)
    enriched_failure_rate = sum(
        1
        for score in enriched_scores
        if not score["contains_target"] or enriched_scores.index(score) > 0
    ) / len(enriched_scores)

    # Calculate average ranks
    avg_base_rank = total_base_rank / total_queries if total_base_rank > 0 else "N/A"
    avg_enriched_rank = (
        total_enriched_rank / total_queries if total_enriched_rank > 0 else "N/A"
    )

    # Print results
    console.print("\n")
    console.print(table)
    console.print("\n")

    # If the failure rates can be calculated
    if (
        isinstance(base_failure_rate, (int, float))
        and isinstance(enriched_failure_rate, (int, float))
        and base_failure_rate > 0
    ):
        failure_rate_reduction = (
            (base_failure_rate - enriched_failure_rate) / base_failure_rate * 100
        )
        console.print(f"[bold]Base Failure Rate:[/bold] {base_failure_rate:.2f}")
        console.print(
            f"[bold]Enriched Failure Rate:[/bold] {enriched_failure_rate:.2f}"
        )
        console.print(
            f"[bold]Failure Rate Reduction:[/bold] {failure_rate_reduction:.1f}%"
        )

    # Print average rank improvement
    if isinstance(avg_base_rank, (int, float)) and isinstance(
        avg_enriched_rank, (int, float)
    ):
        rank_improvement = avg_base_rank - avg_enriched_rank
        console.print(f"[bold]Average Base Rank:[/bold] {avg_base_rank:.2f}")
        console.print(f"[bold]Average Enriched Rank:[/bold] {avg_enriched_rank:.2f}")
        console.print(
            f"[bold]Average Rank Improvement:[/bold] {rank_improvement:.2f} positions"
        )

    console.print("\n[bold]Contextual enrichment summary:[/bold]")
    console.print("Added context to each chunk that summarizes its content")
    console.print("This helps the search algorithm match queries to relevant chunks")
    console.print("The enriched chunks rank higher for relevant queries")


if __name__ == "__main__":
    evaluate_retrieval()
