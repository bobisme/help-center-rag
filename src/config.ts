// Configuration options for the help center documentation scraper

export interface ScraperConfig {
  // Base URL of the documentation site to scrape
  baseUrl: string;

  // Maximum crawl depth from the homepage
  maxDepth: number;

  // Number of concurrent page crawls
  concurrency: number;

  // Output file path (relative to the project root)
  outputPath: string;

  // Whether to run the browser in headless mode
  headless: boolean;

  // Timeout for page loads (in milliseconds)
  pageTimeout: number;

  // Default wait time for content to load (in milliseconds)
  waitTime: number;
}

// Default configuration
export const defaultConfig: ScraperConfig = {
  baseUrl: process.env.HELP_CENTER_URL || '',
  maxDepth: 5,
  concurrency: 2,
  outputPath: 'output/scraped-docs.md',
  headless: true,
  pageTimeout: 5000, // 5 seconds timeout for page loads
  waitTime: 500, // 500ms waiting time for dynamic content
};

// Load custom configuration from command line arguments or file
export function loadConfig(args: string[]): ScraperConfig {
  const config = { ...defaultConfig };

  // Parse command line arguments
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    if (arg === '--url' || arg === '-u') {
      config.baseUrl = args[++i];
    } else if (arg === '--depth' || arg === '-d') {
      config.maxDepth = parseInt(args[++i], 10);
    } else if (arg === '--concurrency' || arg === '-c') {
      config.concurrency = parseInt(args[++i], 10);
    } else if (arg === '--output' || arg === '-o') {
      config.outputPath = args[++i];
    } else if (arg === '--headless') {
      config.headless = args[++i] === 'true';
    } else if (arg === '--timeout') {
      config.pageTimeout = parseInt(args[++i], 10);
    } else if (arg === '--wait') {
      config.waitTime = parseInt(args[++i], 10);
    }
  }

  return config;
}
