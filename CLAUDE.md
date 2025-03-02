# Development Guidelines for scrape-epic-docs

## Commands
- Install: `bun install`
- Run: `bun run index.ts`
- Run HTML to Markdown converter: `bun run src/scripts/json-to-markdown.ts`
- Run Markdown condenser: `bun run src/scripts/condense-markdown.ts input.md output.md`
- Run token counter: `bun run src/scripts/count-tokens.ts input.md`
- Run parallel crawler: `bun run src/scripts/parallel-json-crawler.ts`
- Typecheck: `bun x tsc --noEmit`
- Format: `bun x prettier --write "**/*.{ts,js,json}"`

## Project Structure
- **src/index.ts**: Main entry point that shows available scripts
- **src/scripts/**: Contains individual tools for the documentation processing pipeline
  - **parallel-json-crawler.ts**: Scrapes Epic docs website and outputs to JSON
  - **json-to-markdown.ts**: Converts scraped JSON to markdown format
  - **condense-markdown.ts**: Reduces markdown content to fit within context windows
  - **count-tokens.ts**: Estimates token counts for LLM context windows

## Processing Pipeline
1. **Crawl**: Use parallel-json-crawler to scrape the Epic docs (outputs epic-docs.json)
2. **Convert**: Use json-to-markdown to convert HTML to markdown (outputs epic-docs.md)
3. **Condense**: Use condense-markdown to reduce content size (outputs epic-docs-condensed.md)
4. **Count**: Use count-tokens to verify it fits in target context window

## Code Style
- **TypeScript**: Use strict mode with ESNext features
- **Imports**: Use ESM format (import/export)
- **Formatting**: 2-space indentation, trailing commas
- **Naming**: camelCase for variables/functions, PascalCase for classes/types
- **Error Handling**: Use try/catch blocks for async operations
- **Types**: Prefer explicit return types on functions
- **Async**: Use async/await pattern over raw promises
- **Comments**: JSDoc for public APIs, inline for complex logic
- **File Structure**: One component/concept per file
- **Bun APIs**: Utilize Bun-specific APIs for performance when appropriate

## Notes on HTML to Markdown Conversion
- The project uses TurndownService with custom rules for handling complex HTML
- Special handling has been added for nested lists (bullet points within numbered lists)
- Additional post-processing is applied to ensure proper markdown formatting