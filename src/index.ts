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
    description: 'Crawl the Epic docs website and output to JSON',
    usage: 'epic-help crawl [options]',
    examples: [
      'epic-help crawl',
      'epic-help crawl --depth 3 --concurrency 4',
      'epic-help crawl --timeout 5000',
    ],
    handler: parallelJsonCrawler.main,
  },
  'convert': {
    name: 'convert',
    description: 'Convert scraped JSON to markdown format',
    usage: 'epic-help convert [options]',
    examples: [
      'epic-help convert',
      'epic-help convert --input custom-input.json --output custom-output.md',
    ],
    handler: jsonToMarkdown.main,
  },
  'condense': {
    name: 'condense',
    description: 'Reduce markdown content to fit within context windows',
    usage: 'epic-help condense <input-file> <output-file>',
    examples: [
      'epic-help condense output/epic-docs.md output/epic-docs-condensed.md',
    ],
    handler: condenseMarkdown.main,
  },
  'count': {
    name: 'count',
    description: 'Estimate token counts for LLM context windows',
    usage: 'epic-help count <file-path>',
    examples: [
      'epic-help count output/epic-docs.md',
      'epic-help count output/epic-docs-condensed.md',
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
    console.log(`Epic Documentation Processing Toolkit - ${cmd.name}`);
    console.log('='.repeat(40));
    console.log(`\nDescription: ${cmd.description}`);
    console.log(`\nUsage: ${cmd.usage}`);
    console.log('\nExamples:');
    cmd.examples.forEach((example) => {
      console.log(`  ${example}`);
    });
    return;
  }

  console.log('Epic Documentation Processing Toolkit');
  console.log('===================================');
  console.log('\nA CLI tool to crawl and process Applied Systems Epic documentation');
  console.log('\nUsage: epic-help <command> [options]');
  console.log('\nAvailable commands:');
  
  Object.values(commands).forEach((cmd) => {
    console.log(`  ${cmd.name.padEnd(12)} ${cmd.description}`);
  });
  
  console.log('\nFor help with a specific command:');
  console.log('  epic-help help <command>');
  console.log('  epic-help <command> --help');
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
