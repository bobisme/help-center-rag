#!/usr/bin/env bun
import fs from 'node:fs';
import path from 'node:path';
import { loadConfig, defaultConfig } from './config';

// Import command handlers
import * as jsonToMarkdown from './scripts/json-to-markdown';
import * as condenseMarkdown from './scripts/condense-markdown';
import * as countTokens from './scripts/count-tokens';
import * as parallelJsonCrawler from './scripts/parallel-json-crawler';

interface Command {
  name: string;
  description: string;
  usage: string;
  examples: string[];
  handler: (args: string[]) => Promise<void>;
}

/**
 * Command registry
 */
const commands: Record<string, Command> = {
  'crawl': {
    name: 'crawl',
    description: 'Crawl the help center website and output to JSON with images',
    usage: 'help-center-rag crawl [options]',
    examples: [
      'help-center-rag crawl',
      'help-center-rag crawl --depth 3 --concurrency 4',
      'help-center-rag crawl --timeout 5000',
      'help-center-rag crawl --no-images',
      'help-center-rag crawl --all-images',
    ],
    handler: parallelJsonCrawler.main,
  },
  'convert': {
    name: 'convert',
    description: 'Convert scraped JSON to markdown format',
    usage: 'help-center-rag convert [options]',
    examples: [
      'help-center-rag convert',
      'help-center-rag convert --input custom-input.json --output custom-output.md',
    ],
    handler: jsonToMarkdown.main,
  },
  'condense': {
    name: 'condense',
    description: 'Reduce markdown content to fit within context windows',
    usage: 'help-center-rag condense <input-file> <output-file>',
    examples: [
      'help-center-rag condense output/scraped-docs.md output/scraped-docs-condensed.md',
    ],
    handler: condenseMarkdown.main,
  },
  'count': {
    name: 'count',
    description: 'Estimate token counts for LLM context windows',
    usage: 'help-center-rag count <file-path>',
    examples: [
      'help-center-rag count output/scraped-docs.md',
      'help-center-rag count output/scraped-docs-condensed.md',
    ],
    handler: countTokens.main,
  },
};

/**
 * Print help information
 */
function printHelp(commandName?: string) {
  if (commandName && commands[commandName]) {
    const cmd = commands[commandName];
    console.log(`Help Center Documentation Processing Toolkit - ${cmd.name}`);
    console.log('='.repeat(40));
    console.log(`\nDescription: ${cmd.description}`);
    console.log(`\nUsage: ${cmd.usage}`);
    console.log('\nExamples:');
    cmd.examples.forEach((example) => {
      console.log(`  ${example}`);
    });
    return;
  }

  console.log('Help Center Documentation Processing Toolkit');
  console.log('==========================================');
  console.log('\nA CLI tool to crawl and process help center documentation');
  console.log('\nUsage: help-center-rag <command> [options]');
  console.log('\nAvailable commands:');
  
  Object.values(commands).forEach((cmd) => {
    console.log(`  ${cmd.name.padEnd(12)} ${cmd.description}`);
  });
  
  console.log('\nFor help with a specific command:');
  console.log('  help-center-rag help <command>');
  console.log('  help-center-rag <command> --help');
}

/**
 * Main entry point for the Epic Documentation toolkit
 */
async function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  // Handle help
  if (!command || command === 'help') {
    const helpTopic = args[1];
    printHelp(helpTopic);
    return;
  }

  // Check if command exists
  if (!commands[command]) {
    console.error(`Unknown command: ${command}`);
    console.log('Run "epic-help help" to see available commands');
    process.exit(1);
  }

  // Check for help flag
  if (args.includes('--help') || args.includes('-h')) {
    printHelp(command);
    return;
  }

  // Execute the command, passing along any additional arguments
  try {
    await commands[command].handler(args.slice(1));
  } catch (error) {
    console.error(`Error executing command ${command}:`, error);
    process.exit(1);
  }
}

// Run the script
main().catch((error) => {
  console.error('Unhandled error:', error);
  process.exit(1);
});
