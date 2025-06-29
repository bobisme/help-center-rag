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
  return (
    content
      // Pattern 1: Full access denied sections with heading
      .replace(/^# Access Denied[\s\S]*?The page you requested is restricted\.[\s\S]*?\n\n/gm, '')
      // Pattern 2: Access denied without heading format
      .replace(/^Access Denied[\s\S]*?The page you requested is restricted\.[\s\S]*?\n\n/gm, '')
      // Pattern 3: Lines mentioning restricted access
      .replace(/^.*[Rr]estricted [Aa]ccess.*$/gm, '')
      // Pattern 4: Unauthorized access messages
      .replace(/^.*[Uu]nauthorized [Aa]ccess.*$/gm, '')
  );
}

/**
 * Process 15: Fix image markdown formatting issues
 */
function fixLoneExclamationMarks(content: string): string {
  // Find any lines that contain just a single "!" character and remove them
  // Proper markdown images should be in the format ![alt text](image-url)

  // First, let's try to rebuild broken image syntax that spans multiple lines
  // Pattern: "!" on its own line, followed by [text] on next line, followed by (url) on next line
  const lines = content.split('\n');
  const fixedLines = [];
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
  // Process line by line to handle complex nesting cases
  const lines = content.split('\n');
  const result: string[] = [];

  // Track state
  let inList = false; // Are we inside a numbered list?
  let currentNumber = 0; // Current list item number
  let continuationText = false; // Is this line a continuation of previous list item?
  let prevLineWasBullet = false; // Was the previous line a bullet point?

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Check if this is a numbered list item
    const numMatch = line.match(/^(\d+)\.(\s+)(.*)/);
    if (numMatch) {
      const [, num, , lineContent] = numMatch;
      const numValue = parseInt(num, 10);

      // Special case: Check if this is the start of a nested sequence
      if (numValue === 1 && inList && currentNumber === 1) {
        // This is likely "1. ... \n 1. - ..." pattern - handle specially
        // Continue with the current list and don't create a new numbered item

        // Check if the content starts with a bullet point
        if (lineContent.trim().startsWith('-')) {
          const bulletContent = lineContent.trim().substring(1).trim();
          // Instead of a new list item, format as a bullet point
          result.push(`    - ${bulletContent}`);
          prevLineWasBullet = true;
          continuationText = true;
          continue;
        }
      }

      // Normal numbered list item processing
      inList = true;
      currentNumber = numValue;
      continuationText = false;

      // Add blank line before list items if needed
      if (i > 0 && !lines[i - 1].trim().endsWith(':') && result[result.length - 1] !== '') {
        result.push('');
      }

      // Add the list item
      result.push(`${num}. ${lineContent}`);
      prevLineWasBullet = false;
    }
    // Check for bullet points
    else if (line.match(/^\s*[-*]\s/)) {
      const bulletMatch = line.match(/^(\s*)[-*](\s+)(.*)/);
      if (bulletMatch) {
        const [, , , bulletContent] = bulletMatch;

        // If in a numbered list, standardize the indentation
        if (inList) {
          result.push(`    - ${bulletContent}`);
          prevLineWasBullet = true;
          continuationText = true;
        } else {
          // Regular bullet (not in a list)
          result.push(line);
          prevLineWasBullet = true;
          continuationText = false;
        }
      } else {
        // No match, preserve the line
        result.push(line);
        prevLineWasBullet = false;
      }
    }
    // Check for empty lines
    else if (line.trim() === '') {
      result.push('');
      // Don't change list state on empty lines
      continuationText = false;
      prevLineWasBullet = false;
    }
    // Regular line - could be list item continuation
    else {
      // Special case: look for lines that should be continuations of list items
      if (inList) {
        if (
          prevLineWasBullet ||
          continuationText ||
          line.trim().startsWith('From any other area')
        ) {
          // This is a continuation that should be indented
          result.push(`    ${line.trim()}`);
          continuationText = true;
        } else {
          // Regular line, not a list continuation
          result.push(line);

          // End list mode if this is clearly not part of the list
          if (!line.startsWith(' ') && !line.trim().endsWith(':')) {
            inList = false;
            continuationText = false;
          }
        }
      } else {
        // Not in list mode
        result.push(line);
      }
      prevLineWasBullet = false;
    }
  }

  // Final processing to fix specific patterns
  let processed = result.join('\n');

  // Look for the pattern where we have multiple numbered list items with incorrect format
  processed = processed.replace(
    /^1\.\s+(.*?)(?:\n\n(\d+)\.\s+-\s+)/gms,
    (match, firstContent, nextNum) => {
      // If nextNum is 2, this is likely a single list with nested items
      if (nextNum === '2') {
        return `1. ${firstContent}\n\n    - `;
      }
      return match;
    },
  );

  // Look for the specific pattern in the Commission/Premium Calculations section
  processed = processed.replace(
    /^(1\. From the Home screen, do one of the following:)\n\n1\. -/gm,
    '$1\n\n    -',
  );

  // Look for "From any other area of the program" text that should be indented
  processed = processed.replace(
    /(\n\s+- .*?menubar\.\n)From any other area of the program/g,
    '$1    From any other area of the program',
  );

  // Ultra-specific fix for the Commission/Premium Calculations section
  processed = processed.replace(
    '- Click Areas > Configure on the menubar.\nFrom any other area of the program',
    '- Click Areas > Configure on the menubar.\n    From any other area of the program',
  );

  // Fix any remaining spacing issues
  processed = processed
    // Ensure consistent bullet point spacing
    .replace(/^(\s*)[-*]\s{2,}/gm, '$1- ')
    // Fix multiple consecutive empty lines
    .replace(/\n{3,}/g, '\n\n');

  return processed;
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
export async function main(args: string[] = []): Promise<void> {
  console.log('Condensing Epic Documentation');
  console.log('============================');

  // If args are empty, use process.argv (command line args)
  if (args.length === 0) {
    args = process.argv.slice(2);
  }

  // Check for help flag
  if (args.includes('--help') || args.includes('-h')) {
    console.log('Usage: help-center-rag condense <input-file> <output-file> [options]');
    console.log('Options:');
    console.log('  --abbreviate         Enable abbreviations for greater reduction');
    console.log('  --summarize          Enable aggressive content summarization');
    console.log('');
    console.log('Example: help-center-rag condense output/scraped-docs.md output/scraped-docs-condensed.md --abbreviate');
    return;
  }

  // Check for required arguments
  if (args.length < 2) {
    console.log('Error: Missing required arguments');
    console.log('Usage: help-center-rag condense <input-file> <output-file>');
    console.log('Run "help-center-rag condense --help" for more information');
    throw new Error('Missing required arguments');
  }

  const inputFile = args[0];
  const outputFile = args[1];

  // Parse optional flags
  if (args.includes('--abbreviate')) {
    config.useAbbreviations = true;
  }
  
  if (args.includes('--summarize')) {
    config.enableContentSummarization = true;
  }

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
    throw error;
  }
}

// If run directly, execute the main function
if (require.main === module) {
  main().catch((error) => {
    console.error('Error:', error);
    process.exit(1);
  });
}
