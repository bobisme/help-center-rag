#!/usr/bin/env bun
import fs from 'node:fs/promises';

/**
 * A more accurate token counter for different LLM models
 */
export async function main(args: string[] = []): Promise<void> {
  console.log('Token Counter for LLM Models');
  console.log('===========================');

  // If args are empty, use process.argv (command line args)
  if (args.length === 0) {
    args = process.argv.slice(2);
  }

  // Check for help flag
  if (args.includes('--help') || args.includes('-h')) {
    console.log('Usage: epic-help count <file-path>');
    console.log('Example: epic-help count output/epic-docs.md');
    return;
  }

  // Check for required arguments
  if (args.length === 0) {
    console.log('Error: Missing file path');
    console.log('Usage: epic-help count <file-path>');
    throw new Error('Missing file path');
  }

  const filePath = args[0];

  try {
    // Read the file
    console.log(`Reading file: ${filePath}`);
    const content = await fs.readFile(filePath, 'utf8');

    // Basic stats
    const chars = content.length;
    const words = content.split(/\s+/).length;
    const lines = content.split('\n').length;

    console.log(`File size: ${(chars / 1024 / 1024).toFixed(2)} MB`);
    console.log(`Characters: ${chars.toLocaleString()}`);
    console.log(`Words: ${words.toLocaleString()}`);
    console.log(`Lines: ${lines.toLocaleString()}`);

    // Token estimates for different models
    console.log('\nToken estimates by model:');

    // GPT-3.5/4 estimate (roughly 100 tokens per 75 characters)
    const gptTokens = Math.round((chars * (4 / 3)) / 4);
    console.log(`OpenAI GPT models: ~${gptTokens.toLocaleString()} tokens`);

    // Claude estimate (slightly different tokenization)
    const claudeTokens = Math.round(chars / 3.5);
    console.log(`Anthropic Claude models: ~${claudeTokens.toLocaleString()} tokens`);

    // Gemini estimate
    // Gemini tends to have smaller tokens on average compared to GPT
    const geminiTokens = Math.round(chars / 3.2);
    console.log(`Google Gemini models: ~${geminiTokens.toLocaleString()} tokens`);

    // Llama estimate
    const llamaTokens = Math.round(chars / 3.6);
    console.log(`Meta Llama models: ~${llamaTokens.toLocaleString()} tokens`);

    // More accurate token counting method
    console.log('\nDetailed token calculation:');

    // Function to count tokens more accurately
    function countTokensAccurately(text: string): number {
      // This is a simplified approach - actual tokenizers are more complex

      // Break on common token boundaries
      const patterns = [
        /\b\w+\b/g, // Words
        /[.,;:!?()[\]{}'"]/g, // Punctuation
        /\s+/g, // Whitespace
        /[^a-zA-Z0-9\s.,;:!?()[\]{}'"]+/g, // Other characters
      ];

      let tokenCount = 0;

      // Count matches for each pattern
      for (const pattern of patterns) {
        const matches = text.match(pattern);
        if (matches) {
          tokenCount += matches.length;
        }
      }

      // Account for common compression in tokenizers
      // Common words and subwords are often single tokens
      const commonPrefixes = [
        'un',
        're',
        'in',
        'dis',
        'en',
        'non',
        'de',
        'over',
        'mis',
        'sub',
        'pre',
        'inter',
        'fore',
        'anti',
        'auto',
        'bi',
        'co',
        'ex',
        'mid',
        'semi',
      ];
      const commonSuffixes = [
        'ing',
        'ed',
        'ly',
        'tion',
        'ment',
        'ness',
        'ity',
        'ize',
        'ise',
        'ful',
        'able',
        'ible',
        'al',
        'ial',
        'er',
        'est',
        'ism',
        'ist',
        'ious',
        'ous',
        's',
      ];

      let prefixSuffixMatches = 0;

      // Count prefix/suffix matches
      for (const prefix of commonPrefixes) {
        const prefixRegex = new RegExp(`\\b${prefix}\\w+`, 'g');
        const matches = text.match(prefixRegex);
        if (matches) {
          prefixSuffixMatches += matches.length;
        }
      }

      for (const suffix of commonSuffixes) {
        const suffixRegex = new RegExp(`\\w+${suffix}\\b`, 'g');
        const matches = text.match(suffixRegex);
        if (matches) {
          prefixSuffixMatches += matches.length;
        }
      }

      // Adjust the token count based on common prefixes/suffixes
      const adjustedTokenCount = tokenCount - Math.round(prefixSuffixMatches * 0.5);

      return adjustedTokenCount;
    }

    const accurateTokens = countTokensAccurately(content);
    console.log(`Estimate using pattern recognition: ~${accurateTokens.toLocaleString()} tokens`);

    // Context window information
    console.log('\nLLM Context Window Utilization:');
    console.log(`- GPT-3.5 (4K): ${((gptTokens / 4000) * 100).toFixed(1)}%`);
    console.log(`- GPT-4 (8K): ${((gptTokens / 8000) * 100).toFixed(1)}%`);
    console.log(`- GPT-4 Turbo (128K): ${((gptTokens / 128000) * 100).toFixed(1)}%`);
    console.log(`- Claude 3 Sonnet (180K): ${((claudeTokens / 180000) * 100).toFixed(1)}%`);
    console.log(`- Gemini Pro (32K): ${((geminiTokens / 32000) * 100).toFixed(1)}%`);
    console.log(`- Gemini Ultra (1M): ${((geminiTokens / 1000000) * 100).toFixed(1)}%`);

    // Chunking advice
    if (geminiTokens > 500000) {
      console.log('\nRecommendation: Split this content for most models.');
      console.log('For RAG applications, consider chunks of 1000-2000 tokens each.');
    } else if (geminiTokens > 32000) {
      console.log(
        '\nRecommendation: Suitable for Gemini Ultra and Claude 3 Opus without chunking.',
      );
      console.log("For other models, you'll need RAG or chunking approaches.");
    } else {
      console.log('\nRecommendation: This content should fit in most modern LLMs directly.');
    }
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
