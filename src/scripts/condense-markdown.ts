#!/usr/bin/env bun
import fs from 'node:fs/promises';
import path from 'node:path';

// Types
type ContentProcessor = (content: string) => string;
type CondensationStep = {
  name: string;
  description: string;
  enabled: boolean;
  processor: ContentProcessor;
};

/**
 * Configuration for the condensation process.
 * Set enabled to false to skip a particular step.
 */
const config = {
  useAbbreviations: false,
  enableContentSummarization: false, // Set to false to disable aggressive content summarization
};

/**
 * Process 1: Remove repetitive headers
 */
function removeRepetitiveHeaders(content: string): string {
  return (
    content
      // Remove "Applied Epic Help File" lines
      .replace(/^Applied Epic .*? Help File\s*$/gm, '')
      // Remove "Click here to see this page in full context"
      .replace(/^Click here to see this page in full context\s*$/gm, '')
      // Remove "To learn more..." lines
      .replace(/^To learn more about this capability.*$/gm, '')
      // Remove launch video buttons and links
      .replace(/^\[Launch video\].*$/gm, '')
      // Remove redundant navigation instructions
      .replace(/^If you are .* screen, skip steps.*$/gm, '')
  );
}

/**
 * Process 14: Remove Access Denied blocks
 */
function removeAccessDeniedBlocks(content: string): string {
  // Multiple patterns to capture different variations of access denied blocks
  return content
    // Pattern 1: Full access denied sections with heading
    .replace(/^# Access Denied[\s\S]*?The page you requested is restricted\.[\s\S]*?\n\n/gm, '')
    // Pattern 2: Access denied without heading format
    .replace(/^Access Denied[\s\S]*?The page you requested is restricted\.[\s\S]*?\n\n/gm, '')
    // Pattern 3: Lines mentioning restricted access
    .replace(/^.*[Rr]estricted [Aa]ccess.*$/gm, '')
    // Pattern 4: Unauthorized access messages
    .replace(/^.*[Uu]nauthorized [Aa]ccess.*$/gm, '');
}

/**
 * Process 15: Fix image markdown formatting issues
 */
function fixLoneExclamationMarks(content: string): string {
  // Find any lines that contain just a single "!" character and remove them
  // Proper markdown images should be in the format ![alt text](image-url)
  
  // First, let's try to rebuild broken image syntax that spans multiple lines
  // Pattern: "!" on its own line, followed by [text] on next line, followed by (url) on next line
  let lines = content.split('\n');
  let fixedLines = [];
  let i = 0;
  
  while (i < lines.length) {
    const currentLine = lines[i];
    
    // Check if this line is just an exclamation mark
    if (/^\s*!\s*$/.test(currentLine) && i < lines.length - 2) {
      const nextLine = lines[i + 1];
      const nextNextLine = lines[i + 2];
      
      // Check if next line contains an image alt text [text] and the line after contains (url)
      const altTextMatch = nextLine.match(/^\s*\[(.*?)\]\s*$/);
      const urlMatch = nextNextLine.match(/^\s*\((.*?)\)\s*$/);
      
      if (altTextMatch && urlMatch) {
        // Rebuild the proper image syntax
        fixedLines.push(`![${altTextMatch[1]}](${urlMatch[1]})`);
        i += 3; // Skip the next two lines since we've processed them
        continue;
      }
    }
    
    // If not part of a broken image pattern, keep the line as is
    fixedLines.push(currentLine);
    i++;
  }
  
  let result = fixedLines.join('\n');
  
  // Finally, remove any remaining standalone exclamation marks
  result = result.replace(/^\s*!\s*$/gm, '');
  
  return result;
}

/**
 * Process 16: Fix numbered list formatting issues
 */
function fixNumberedListFormatting(content: string): string {
  // This is a more specialized function to handle specifically the Epic docs format issues
  // with nested lists under numbered items.
  
  // First, let's identify and fix our specific pattern:
  // 1. Find a numbered list followed by "do one of the following:"
  // 2. Look for subsequence number + bullet point patterns
  
  // Split into sections (paragraphs) to process
  const sections = content.split(/\n\n+/);
  const processedSections: string[] = [];
  
  for (let i = 0; i < sections.length; i++) {
    const section = sections[i];
    
    // Check if this is a section that needs fixing (contains a numbered list item with bullet points)
    if (/^\d+\..*\n\d+\.\s*-\s/.test(section)) {
      // This appears to be a problematic section, let's fix it
      const lines = section.split('\n');
      const fixedLines: string[] = [];
      
      // Keep track of the current state
      let inNumberedItem = false;
      let currentNumber = 0;
      
      for (let j = 0; j < lines.length; j++) {
        const line = lines[j];
        
        // Check if this is a new numbered item
        const numberMatch = line.match(/^(\d+)\.\s*(.*)/);
        if (numberMatch) {
          currentNumber = parseInt(numberMatch[1], 10);
          let content = numberMatch[2].trim();
          
          // If content starts with a bullet point, it's a special case
          if (content.startsWith('-')) {
            inNumberedItem = true;
            // This is the start of a bullet list under a numbered item
            content = content.substring(1).trim();
            fixedLines.push(`${currentNumber}. ${content}`);
            // Add 4 spaces before subsequent bullet points
          } else {
            // Normal numbered list item
            inNumberedItem = false;
            fixedLines.push(`${currentNumber}. ${content}`);
          }
        }
        // Check if this is a bullet point that should be indented
        else if (line.trim().startsWith('-')) {
          // This is a bullet point that should be indented
          const bulletContent = line.trim().substring(1).trim();
          
          // If we just came from a numbered item but not already in bullet mode
          if (!inNumberedItem && currentNumber > 0) {
            inNumberedItem = true;
            // Convert previous line to make it clear this is a parent item
            if (fixedLines.length > 0 && fixedLines[fixedLines.length - 1].match(/^\d+\.\s*$/)) {
              // Previous line was just a number, content starts with this bullet
              const bullet = `    - ${bulletContent}`;
              fixedLines.push(bullet);
            } else {
              // Add the indented bullet
              fixedLines.push(`    - ${bulletContent}`);
            }
          } else {
            // Continue with indented bullets
            fixedLines.push(`    - ${bulletContent}`);
          }
        }
        // Regular line (continuation of previous item)
        else {
          if (inNumberedItem && line.trim() !== '') {
            // Indent this line to align with the bullet points
            fixedLines.push(`    ${line.trim()}`);
          } else {
            fixedLines.push(line);
          }
        }
      }
      
      processedSections.push(fixedLines.join('\n'));
    } else {
      // This section doesn't need fixing
      processedSections.push(section);
    }
  }
  
  // Apply general formatting fixes to the entire content
  let result = processedSections.join('\n\n');
  
  // Find and fix any remaining bullet points after a numbered item
  result = result.replace(/^(\d+\.\s*[^\n]*)\n(\s*-\s)/gm, '$1\n    $2');
  
  return result;
}

/**
 * Process 2: Condense paths and source lines
 */
function condensePaths(content: string): string {
  return (
    content
      // Condense path and source lines
      .replace(/^_Path: (.*?)_\n\n_Source: (.*?)_\n\n/gm, '_Path: $1_ | _Source: $2_\n\n')
  );
}

/**
 * Process 3: Remove empty lines and collapse multiple empty lines
 */
function removeEmptyLines(content: string): string {
  return (
    content
      // Collapse multiple empty lines into one
      .replace(/\n{3,}/g, '\n\n')
      // Remove blank lines consisting only of whitespace
      .replace(/^\s+$/gm, '')
  );
}

/**
 * Process 4: Condense list formatting
 */
function condenseLists(content: string): string {
  return (
    content
      // Make list items more compact
      .replace(/^(\s*[-*]\s+)(.*?)(\n\n)/gm, '$1$2\n')
  );
}

/**
 * Process 5: Remove non-essential information
 */
function removeNonEssentialInfo(content: string): string {
  return (
    content
      // Remove note sections with phrasing like "Note that" or "Please note"
      .replace(/^Note:.*$\n/gm, '')
      .replace(/^Please note.*$\n/gm, '')
      // Remove excessive separator lines
      .replace(/\n---\n\n---\n/g, '\n---\n')
  );
}

/**
 * Process 6: Condense step instructions
 */
function condenseInstructions(content: string): string {
  // Replace common instruction patterns with shorter versions
  const instructionPairs = [
    [/In the .*? area,\s+/g, ''],
    [/From the .*? screen,\s+/g, ''],
    [/On the .*? tab,\s+/g, ''],
    [/Click .*? to return to the .*?\./g, 'Return when done.'],
    [/Use the following steps to /g, ''],
    [/Use the following information to /g, ''],
    [/The system displays/g, 'Displays'],
    [/The .*? screen displays\./g, ''],
    [/When you are finished,/g, ''],
    [/When you have completed/g, 'After'],
  ];

  let result = content;
  for (const [pattern, replacement] of instructionPairs) {
    result = result.replace(pattern, replacement as string);
  }
  return result;
}

/**
 * Process 7: Abbreviate common phrases
 */
function abbreviateCommonPhrases(content: string): string {
  if (!config.useAbbreviations) return content;

  const abbreviationPairs = [
    [/Applied Epic/g, 'AE'],
    [/Applied Systems/g, 'AS'],
    [/configuration/g, 'config'],
    [/information/g, 'info'],
    [/automatically/g, 'auto'],
    [/necessary/g, 'needed'],
    [/additional/g, 'more'],
    [/documentation/g, 'docs'],
    [/functionality/g, 'features'],
    [/parameters/g, 'params'],
    [/generates/g, 'creates'],
    [/selected/g, 'chosen'],
    [/Application/g, 'App'],
    [/Administrator/g, 'Admin'],
    [/management/g, 'mgmt'],
    [/reference/g, 'ref'],
    [/transaction/g, 'trans'],
    [/organization/g, 'org'],
    [/following/g, 'these'],
    [/Processing/g, 'Proc'],
    [/Technical/g, 'Tech'],
    [/Environment/g, 'Env'],
    [/Structure/g, 'Struct'],
    [/Procedure/g, 'Proc'],
    [/integration/g, 'integ'],
    [/Interface/g, 'Intf'],
    [/Registration/g, 'Reg'],
    [/Operation/g, 'Op'],
  ];

  let result = content;
  for (const [phrase, abbrev] of abbreviationPairs) {
    result = result.replace(phrase, abbrev as string);
  }
  return result;
}

/**
 * Process 8: Optimize headings
 */
function optimizeHeadings(content: string): string {
  // Find heading level reduction point (preserve the main TOC)
  const tocEnd =
    content.indexOf('## Table of Contents') +
    content.substring(content.indexOf('## Table of Contents')).indexOf('\n\n');

  // Only apply heading reduction after the TOC
  const beforeToc = content.substring(0, tocEnd);
  let afterToc = content.substring(tocEnd);

  // Reduce heading levels (### -> ##, #### -> ###)
  afterToc = afterToc.replace(/^(#{3,6})\s+/gm, (match, hashes) => {
    return '#'.repeat(Math.max(2, hashes.length - 1)) + ' ';
  });

  return beforeToc + afterToc;
}

/**
 * Process 9: Remove unnecessary horizontal rules
 */
function removeUnnecessarySeparators(content: string): string {
  return (
    content
      // Limit horizontal rules - remove if followed by a heading
      .replace(/\n---\n\n#+\s+/g, '\n\n## ')
  );
}

/**
 * Process 10: Remove obvious steps
 */
function removeObviousSteps(content: string): string {
  const obviousSteps = [
    /Click (OK|Cancel|Finish|Close) to complete the process\.\s*/g,
    /Click (OK|Cancel|Finish|Close) when finished\.\s*/g,
    /The (window|dialog|screen) will close\.\s*/g,
    /Refer to the screenshot below\.\s*/g,
    /See the screenshot for an example\.\s*/g,
    /See the image above for reference\.\s*/g,
    /A dialog box displays\.\s*/g,
    /The system displays a message\.\s*/g,
    /The dialog box closes\.\s*/g,
  ];

  let result = content;
  for (const pattern of obviousSteps) {
    result = result.replace(pattern, '');
  }
  return result;
}

/**
 * Process 11: Remove redundant content
 */
function removeRedundantContent(content: string): string {
  return (
    content
      // Remove common navigation patterns
      .replace(/To access.*? screen:.*?\n\n/gs, '')
      .replace(/To open.*? window:.*?\n\n/gs, '')
      .replace(/To view.*? screen:.*?\n\n/gs, '')
      // Remove standard procedures that are repeated
      .replace(/Standard procedures:.*?\n\n/gs, '')
      // Remove "Learn more about" sections
      .replace(/To learn more about.*?\n\n/gs, '')
      // Remove system requirements sections
      .replace(/System requirements:.*?\n\n/gs, '')
  );
}

/**
 * Process 12: Remove low-value descriptions
 */
function removeLowValueDescriptions(content: string): string {
  return (
    content
      // Remove "This screen displays..." explanations
      .replace(/This (screen|window|dialog|area) (displays|shows|allows).*?\n\n/gs, '')
      // Remove "The purpose of..." explanations
      .replace(/The purpose of this.*?\n\n/gs, '')
      // Remove "When you..." explanations that are obvious
      .replace(/When you click.*?, the system.*?\n\n/gs, '')
  );
}

/**
 * Process 13: Summarize long sections
 */
function summarizeLongSections(content: string): string {
  if (!config.enableContentSummarization) return content;

  // Split the content by sections
  const sections = content.split(/(?=^## )/gm);

  // Process each section
  const processedSections = sections.map((section, index) => {
    // Don't process the TOC or first sections
    if (index < 2 || section.includes('Table of Contents')) {
      return section;
    }

    // For longer sections, trim examples and duplicate instructions
    if (section.length > 2000) {
      // Remove examples that are too long (>10 lines)
      section = section.replace(/Example:[\s\S]{200,2000}?(?=\n\n)/g, 'Example: [condensed]');

      // Reduce numbered steps if there are too many
      const stepsMatch = section.match(/^\d+\. .*(?:\n\d+\. .*){9,}/gm);
      if (stepsMatch) {
        for (const steps of stepsMatch) {
          // Keep only critical steps (first 4, last 2 steps)
          const allSteps = steps.split(/\n(?=\d+\. )/);
          if (allSteps.length > 7) {
            const reducedSteps = [
              ...allSteps.slice(0, 4),
              `${allSteps.length - 6} more steps...`,
              ...allSteps.slice(-2),
            ].join('\n');
            section = section.replace(steps, reducedSteps);
          }
        }
      }

      // Remove verbose explanations of parameters/options if they are extensive
      section = section.replace(
        /The following (table|list) describes.*?:[\s\S]{500,5000}?(?=\n\n## |$)/g,
        'Parameters/options: [condensed list]',
      );

      // Remove warning and caution blocks that are too verbose
      section = section.replace(
        /Warning:[\s\S]{100,500}?(?=\n\n)/g,
        'Warning: Exercise caution.\n\n',
      );
      section = section.replace(/Caution:[\s\S]{100,500}?(?=\n\n)/g, 'Caution: Take care.\n\n');
    }

    return section;
  });

  // Reassemble the content
  return processedSections.join('');
}

/**
 * Define all the condensation steps
 */
const condensationSteps: CondensationStep[] = [
  {
    name: 'Remove access denied blocks',
    description: 'Removes "Access Denied" sections from restricted pages',
    enabled: true,
    processor: removeAccessDeniedBlocks,
  },
  {
    name: 'Fix lone exclamation marks',
    description: 'Removes lone exclamation marks that might be from broken image tags',
    enabled: true,
    processor: fixLoneExclamationMarks,
  },
  {
    name: 'Fix numbered list formatting',
    description: 'Fixes problems with nested bullet points in numbered lists',
    enabled: true,
    processor: fixNumberedListFormatting,
  },
  {
    name: 'Remove repetitive headers',
    description:
      'Removes repetitive header text like "Applied Epic Help File" and navigation instructions',
    enabled: true,
    processor: removeRepetitiveHeaders,
  },
  {
    name: 'Condense paths',
    description: 'Condenses path and source lines to reduce space',
    enabled: true,
    processor: condensePaths,
  },
  {
    name: 'Remove empty lines',
    description: 'Removes empty lines and collapses multiple empty lines',
    enabled: true,
    processor: removeEmptyLines,
  },
  {
    name: 'Condense lists',
    description: 'Makes list items more compact',
    enabled: true,
    processor: condenseLists,
  },
  {
    name: 'Remove non-essential info',
    description: 'Removes note sections and excessive separator lines',
    enabled: false,
    processor: removeNonEssentialInfo,
  },
  {
    name: 'Condense instructions',
    description: 'Condenses step instructions by removing redundant phrases',
    enabled: false,
    processor: condenseInstructions,
  },
  {
    name: 'Abbreviate common phrases',
    description: 'Replaces common phrases with abbreviations to reduce token count',
    enabled: config.useAbbreviations,
    processor: abbreviateCommonPhrases,
  },
  {
    name: 'Optimize headings',
    description: 'Optimizes heading levels by reducing depth',
    enabled: true,
    processor: optimizeHeadings,
  },
  {
    name: 'Remove unnecessary separators',
    description: 'Removes unnecessary horizontal rules',
    enabled: true,
    processor: removeUnnecessarySeparators,
  },
  {
    name: 'Remove obvious steps',
    description: 'Removes steps that are too obvious or reference screenshots',
    enabled: true,
    processor: removeObviousSteps,
  },
  {
    name: 'Remove redundant content',
    description: 'More aggressively removes redundant navigation patterns and standard procedures',
    enabled: true,
    processor: removeRedundantContent,
  },
  {
    name: 'Remove low-value descriptions',
    description: 'Removes low-value descriptions like "This screen displays..."',
    enabled: true,
    processor: removeLowValueDescriptions,
  },
  {
    name: 'Summarize long sections',
    description: 'Summarizes long sections by condensing examples and reducing step count',
    enabled: config.enableContentSummarization,
    processor: summarizeLongSections,
  },
];

/**
 * Apply all enabled condensation steps to the content
 */
function condenseContent(content: string): {
  condensed: string;
  metrics: Record<string, { before: number; after: number }>;
} {
  const metrics: Record<string, { before: number; after: number }> = {};
  let condensed = content;

  // Apply each enabled processor
  for (const step of condensationSteps) {
    if (step.enabled) {
      const before = condensed.length;
      condensed = step.processor(condensed);
      metrics[step.name] = {
        before,
        after: condensed.length,
      };
    }
  }

  return { condensed, metrics };
}

/**
 * Script to condense the Epic documentation to fit within a 2M token context window
 */
async function main() {
  console.log('Condensing Epic Documentation');
  console.log('============================');

  // Parse command line args
  const args = process.argv.slice(2);
  if (args.length < 2) {
    console.log('Usage: bun run condense-markdown.ts <input-file> <output-file>');
    console.log(
      'Example: bun run condense-markdown.ts output/epic-docs.md output/epic-docs-condensed.md',
    );
    process.exit(1);
  }

  const inputFile = args[0];
  const outputFile = args[1];

  try {
    // Read the input file
    console.log(`Reading input file: ${inputFile}`);
    const content = await fs.readFile(inputFile, 'utf8');
    const originalSize = content.length;

    // Perform condensing operations
    console.log('Applying condensing strategies...');
    console.log('Enabled steps:');
    condensationSteps.forEach((step) => {
      console.log(`- ${step.name}: ${step.enabled ? 'Enabled' : 'Disabled'}`);
    });

    const { condensed, metrics } = condenseContent(content);

    // Write the condensed content to the output file
    await fs.mkdir(path.dirname(outputFile), { recursive: true });
    await fs.writeFile(outputFile, condensed, 'utf8');

    // Calculate the reduction
    const finalSize = condensed.length;
    const reductionPercent = (((originalSize - finalSize) / originalSize) * 100).toFixed(2);

    console.log(`\nCondensation Results:`);
    console.log(`Original size: ${(originalSize / 1024 / 1024).toFixed(2)} MB`);
    console.log(`Condensed size: ${(finalSize / 1024 / 1024).toFixed(2)} MB`);
    console.log(`Reduction: ${reductionPercent}%`);

    // Print metrics for each strategy
    console.log('\nReduction by strategy:');
    for (const [strategy, { before, after }] of Object.entries(metrics)) {
      const diff = before - after;
      const strategyPercent = ((diff / originalSize) * 100).toFixed(2);
      console.log(`- ${strategy}: ${(diff / 1024).toFixed(2)} KB (${strategyPercent}%)`);
    }

    // Estimate token count
    const tokenEstimate = Math.round(finalSize / 3.2); // Gemini estimate
    console.log(`\nEstimated Gemini token count: ~${tokenEstimate.toLocaleString()}`);
    if (tokenEstimate <= 2000000) {
      console.log(`✅ Content should fit within Gemini Pro 2.0's 2M token context window`);
    } else {
      console.log(
        `❌ Content still exceeds Gemini Pro 2.0's 2M token context window by ~${(tokenEstimate - 2000000).toLocaleString()} tokens`,
      );
      console.log(
        `💡 Try enabling abbreviations by setting 'useAbbreviations = true' for further reduction`,
      );
    }

    console.log(`\nCondensed content written to: ${outputFile}`);
  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

// Run the script
main().catch(console.error);
