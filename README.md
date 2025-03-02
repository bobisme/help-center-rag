# Applied Systems Epic Documentation Scraper

A CLI tool to crawl the Applied Systems Epic documentation and compile it into a single markdown file.

## Features

- Crawls the Epic documentation website
- Extracts all documentation content
- Converts HTML to clean Markdown
- Preserves document structure and hierarchy
- Generates a comprehensive table of contents
- Outputs everything to a single searchable markdown file

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/scrape-epic-docs.git
cd scrape-epic-docs

# Install dependencies
bun install
```

## Usage

```bash
# Run the standard crawler (original implementation)
bun run index.ts

# Run the simplified and more robust crawler
bun run final-crawler.ts
```

The output will be saved to `output/epic-docs.md`.

### Command Line Options

The standard crawler supports the following options:

```bash
# Change the maximum crawl depth
bun run index.ts --depth 3

# Set the number of concurrent requests
bun run index.ts --concurrency 4

# Specify a different output file
bun run index.ts --output custom-output.md

# Set the page load timeout in milliseconds
bun run index.ts --timeout 5000

# Set the dynamic content wait time in milliseconds
bun run index.ts --wait 500
```

## Requirements

- [Bun](https://bun.sh/) runtime
- Playwright browsers (installed automatically)

## Development

- Install dependencies: `bun install`
- Run: `bun run index.ts`
- Typecheck: `bun x tsc --noEmit`
- Format: `bun x prettier --write "**/*.{ts,js,json}"`

## License

MIT