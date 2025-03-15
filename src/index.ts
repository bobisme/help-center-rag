#!/usr/bin/env bun
import fs from 'node:fs';
import path from 'node:path';

/**
 * Main entry point for the Epic Documentation toolkit
 */
function main() {
  console.log('Epic Documentation Processing Toolkit');
  console.log('===================================');
  console.log('');

  // List available scripts in the scripts directory
  const scriptsDir = path.join(__dirname, 'scripts');
  const scripts = fs
    .readdirSync(scriptsDir)
    .filter((file) => file.endsWith('.ts'))
    .map((file) => file.replace('.ts', ''));

  console.log('Available tools:');
  scripts.forEach((script) => {
    console.log(`- ${script}`);
  });

  console.log('');
  console.log('Run a tool with: bun run src/scripts/<script-name>.ts');
  console.log('Example: bun run src/scripts/condense-markdown.ts input.md output.md');
}

// Run the script
main();
