#!/usr/bin/env bun
import * as fs from 'node:fs/promises';

/**
 * Modern token counter for 2024-2025 LLM models with updated context windows
 */
export async function main(args: string[] = []): Promise<void> {
  console.log('Token Counter for Modern LLM Models (2024-2025)');
  console.log('==============================================');

  // If args are empty, use process.argv (command line args)
  if (args.length === 0) {
    args = process.argv.slice(2);
  }

  // Check for help flag
  if (args.includes('--help') || args.includes('-h')) {
    console.log('Usage: bun run src/scripts/count-tokens.ts <file-path> [--model <model-name>]');
    console.log('Models: gpt-4o, gpt-4o-mini, claude-3.5-sonnet, claude-4, gemini-1.5-pro, gemini-2.5-pro');
    console.log('Example: bun run src/scripts/count-tokens.ts output/epic-docs.md');
    return;
  }

  // Check for required arguments
  if (args.length === 0) {
    console.log('Error: Missing file path');
    console.log('Usage: bun run src/scripts/count-tokens.ts <file-path>');
    throw new Error('Missing file path');
  }

  const filePath = args[0];
  const modelFlag = args.indexOf('--model');
  const targetModel = modelFlag !== -1 && args[modelFlag + 1] ? args[modelFlag + 1] : null;

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

    // Updated token estimates for 2024-2025 models
    console.log('\nToken estimates by model (2024-2025):');

    // OpenAI GPT-4o family - uses o200k_base encoding
    const gpt4oTokens = Math.round(chars / 3.7); // More accurate for o200k_base
    console.log(`OpenAI GPT-4o/4o-mini: ~${gpt4oTokens.toLocaleString()} tokens`);

    // Anthropic Claude - produces ~16% more tokens than GPT for English
    const claudeTokens = Math.round(chars / 3.2); // Updated based on 2024 research
    console.log(`Anthropic Claude 3.5/4 Sonnet: ~${claudeTokens.toLocaleString()} tokens`);

    // Google Gemini - more efficient tokenization
    const geminiTokens = Math.round(chars / 3.8);
    console.log(`Google Gemini 1.5/2.5 Pro: ~${geminiTokens.toLocaleString()} tokens`);

    // Meta Llama 3/4 family
    const llamaTokens = Math.round(chars / 3.6);
    console.log(`Meta Llama 3/4 models: ~${llamaTokens.toLocaleString()} tokens`);

    // More accurate token counting method
    console.log('\nDetailed token calculation:');

    // Function to count tokens more accurately
    const countTokensAccurately = (text: string): number => {
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
    };

    const accurateTokens = countTokensAccurately(content);
    console.log(`Estimate using pattern recognition: ~${accurateTokens.toLocaleString()} tokens`);

    // Modern Context Window Utilization (2024-2025)
    console.log('\nContext Window Utilization (2024-2025):');
    
    // Tier 1: Massive context windows (1M+ tokens)
    console.log('\n🚀 Ultra-Long Context Models:');
    console.log(`- Google Gemini 2.5 Pro (2M): ${((geminiTokens / 2000000) * 100).toFixed(2)}%`);
    console.log(`- Google Gemini 1.5 Pro (1M): ${((geminiTokens / 1000000) * 100).toFixed(2)}%`);
    console.log(`- GPT-4.1 (1M): ${((gpt4oTokens / 1000000) * 100).toFixed(2)}%`);
    console.log(`- Meta Llama 4 Maverick (1M): ${((llamaTokens / 1000000) * 100).toFixed(2)}%`);
    
    // Tier 2: Large context windows (200K+ tokens)
    console.log('\n📚 Large Context Models:');
    console.log(`- Claude 4 Sonnet/Opus (200K): ${((claudeTokens / 200000) * 100).toFixed(2)}%`);
    console.log(`- Claude 3.5 Sonnet (200K): ${((claudeTokens / 200000) * 100).toFixed(2)}%`);
    console.log(`- Claude Enterprise (500K): ${((claudeTokens / 500000) * 100).toFixed(2)}%`);
    console.log(`- OpenAI o3/o4 (200K): ${((gpt4oTokens / 200000) * 100).toFixed(2)}%`);
    
    // Tier 3: Standard context windows (128K tokens)
    console.log('\n💻 Standard Context Models:');
    console.log(`- GPT-4o/4o-mini (128K): ${((gpt4oTokens / 128000) * 100).toFixed(2)}%`);
    console.log(`- OpenAI o1 family (128K): ${((gpt4oTokens / 128000) * 100).toFixed(2)}%`);

    // Model-specific recommendations
    if (targetModel) {
      console.log(`\n🎯 Specific analysis for ${targetModel}:`);
      const modelSpecs = getModelSpecs(targetModel);
      if (modelSpecs) {
        const relevantTokens = getTokensForModel(targetModel, { gpt4oTokens, claudeTokens, geminiTokens, llamaTokens });
        console.log(`- Context window: ${modelSpecs.contextWindow.toLocaleString()} tokens`);
        console.log(`- Utilization: ${((relevantTokens / modelSpecs.contextWindow) * 100).toFixed(2)}%`);
        console.log(`- Cost tier: ${modelSpecs.costTier}`);
        console.log(`- Best for: ${modelSpecs.bestFor}`);
      }
    }

    // Updated chunking advice
    const maxTokens = Math.max(gpt4oTokens, claudeTokens, geminiTokens, llamaTokens);
    console.log('\n📋 Recommendations:');
    
    if (maxTokens > 1000000) {
      console.log('✨ Suitable for ultra-long context models (Gemini 2.5 Pro, GPT-4.1) without chunking');
      console.log('⚠️  Consider chunking for cost optimization and faster processing');
    } else if (maxTokens > 200000) {
      console.log('✅ Fits in large context models (Claude 4, Claude 3.5) without chunking');
      console.log('🔄 May need chunking for GPT-4o family (128K limit)');
    } else if (maxTokens > 128000) {
      console.log('⚠️  Exceeds GPT-4o context window - chunking required for OpenAI models');
      console.log('✅ Fits in Claude and ultra-long context models');
    } else {
      console.log('✅ Fits in all modern LLM context windows without chunking');
    }
    
    console.log('\n🔧 RAG Chunking Guidelines:');
    console.log('- Recommended chunk size: 800-1200 tokens');
    console.log('- Overlap: 100-200 tokens');
    console.log('- Max chunks per query: 8-12 for optimal performance');

  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}

function getModelSpecs(model: string) {
  const specs: Record<string, any> = {
    'gpt-4o': { contextWindow: 128000, costTier: 'Premium', bestFor: 'General reasoning, coding' },
    'gpt-4o-mini': { contextWindow: 128000, costTier: 'Budget', bestFor: 'Fast tasks, API calls' },
    'claude-3.5-sonnet': { contextWindow: 200000, costTier: 'Premium', bestFor: 'Analysis, writing' },
    'claude-4': { contextWindow: 200000, costTier: 'Premium', bestFor: 'Complex reasoning' },
    'gemini-1.5-pro': { contextWindow: 1000000, costTier: 'Ultra', bestFor: 'Long documents, multimodal' },
    'gemini-2.5-pro': { contextWindow: 2000000, costTier: 'Ultra', bestFor: 'Massive context analysis' },
  };
  return specs[model];
}

function getTokensForModel(model: string, tokens: any) {
  if (model.includes('gpt')) return tokens.gpt4oTokens;
  if (model.includes('claude')) return tokens.claudeTokens;
  if (model.includes('gemini')) return tokens.geminiTokens;
  if (model.includes('llama')) return tokens.llamaTokens;
  return tokens.gpt4oTokens;
}

// If run directly, execute the main function
if (require.main === module) {
  main().catch((error) => {
    console.error('Error:', error);
    process.exit(1);
  });
}
