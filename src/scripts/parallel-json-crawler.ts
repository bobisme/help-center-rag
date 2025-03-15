#!/usr/bin/env bun
import { chromium, type Browser } from 'playwright';
import fs from 'node:fs/promises';
import path from 'node:path';

// Configuration
const CONFIG = {
  baseUrl: 'https://help.appliedsystems.com/Help/Epic/2023.2en-US',
  outputJsonFile: path.join(process.cwd(), 'output', 'epic-docs.json'),
  concurrency: 8, // Number of parallel workers
  maxDepth: 3, // Maximum crawl depth
  maxPages: 1_000, // Maximum pages to process
  pageTimeout: 5000, // Page load timeout (ms)
  waitTime: 500, // Wait time for dynamic content (ms)
  requestInterval: 200, // Delay between requests (ms)
};

// Types for our data structures
interface PageContent {
  url: string;
  title: string;
  rawHtml: string; // Store HTML for later processing
  metadata: {
    depth: number;
    path: string[];
    crawlDate: string;
    parentUrl?: string;
    parentTitle?: string;
  };
  links: {
    text: string;
    url: string;
    isInternal: boolean;
  }[];
}

interface QueueItem {
  url: string;
  title: string;
  depth: number;
  parentUrl?: string;
  parentTitle?: string;
  path?: string[];
}

// Local types to avoid circular references
interface CrawlerConfig {
  baseUrl: string;
  outputJsonFile: string;
  concurrency: number;
  maxDepth: number;
  maxPages: number;
  pageTimeout: number;
  waitTime: number;
  requestInterval: number;
}

/**
 * Parallel crawler with JSON output
 */
export async function main(args: string[] = []): Promise<void> {
  console.log('Epic Documentation Parallel JSON Crawler');
  console.log('=======================================');

  // Parse command line args for custom configuration
  let crawlerConfig: CrawlerConfig = { ...CONFIG };
  
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--url' || arg === '-u') {
      crawlerConfig.baseUrl = args[++i];
    } else if (arg === '--output' || arg === '-o') {
      crawlerConfig.outputJsonFile = path.resolve(process.cwd(), args[++i]);
    } else if (arg === '--concurrency' || arg === '-c') {
      crawlerConfig.concurrency = parseInt(args[++i], 10);
    } else if (arg === '--depth' || arg === '-d') {
      crawlerConfig.maxDepth = parseInt(args[++i], 10);
    } else if (arg === '--max' || arg === '-m') {
      crawlerConfig.maxPages = parseInt(args[++i], 10);
    } else if (arg === '--timeout' || arg === '-t') {
      crawlerConfig.pageTimeout = parseInt(args[++i], 10);
    } else if (arg === '--wait' || arg === '-w') {
      crawlerConfig.waitTime = parseInt(args[++i], 10);
    } else if (arg === '--interval' || arg === '-i') {
      crawlerConfig.requestInterval = parseInt(args[++i], 10);
    } else if (arg === '--help' || arg === '-h') {
      console.log('Usage: epic-help crawl [options]');
      console.log('Options:');
      console.log('  --url, -u <url>           Base URL to crawl');
      console.log('  --output, -o <file>       Output JSON file path');
      console.log('  --concurrency, -c <num>   Number of parallel workers');
      console.log('  --depth, -d <num>         Maximum crawl depth');
      console.log('  --max, -m <num>           Maximum pages to process');
      console.log('  --timeout, -t <ms>        Page load timeout in milliseconds');
      console.log('  --wait, -w <ms>           Wait time for dynamic content in milliseconds');
      console.log('  --interval, -i <ms>       Delay between requests in milliseconds');
      return;
    }
  }

  // Create shared data structures
  const visitedUrls = new Set<string>();
  const seenUrls = new Set<string>();
  const pageContents: PageContent[] = [];
  const urlQueue: QueueItem[] = [];

  // Status tracking
  let processed = 0;
  let totalFound = 0;
  let isRunning = true;

  // List of browsers to close on shutdown
  const browsers: Browser[] = [];

  // Handle process termination gracefully
  process.on('SIGINT', async () => {
    console.log('\nCaught interrupt signal. Cleaning up...');
    isRunning = false;

    // Close all browsers
    for (const browser of browsers) {
      await browser.close().catch(() => {});
    }

    // Save partial results if we have any
    if (pageContents.length > 0) {
      await saveResults(pageContents, crawlerConfig.outputJsonFile, crawlerConfig);
    }

    process.exit(0);
  });

  try {
    // Initial URLs to process
    const initialTopics = [
      { name: 'Introduction', path: '/Introduction/default.htm' },
      { name: 'Accounts', path: '/Accounts/Account_Detail.htm' },
      { name: 'Activities', path: '/Accounts/activities/Activities.htm' },
      { name: 'Policies', path: '/Accounts/Policies/Policies.htm' },
      { name: 'Claims', path: '/Accounts/Claims/Claims.htm' },
      { name: 'Email', path: '/Email/Email.htm' },
      { name: 'General Ledger', path: '/General_Ledger/GL.htm' },
      { name: 'Procedures', path: '/Procedures/Procedures.htm' },
      { name: 'Utilities', path: '/Utilities/Utilities.htm' },
      { name: 'Configure', path: '/Configure/Configure.htm' },
    ];

    // Add initial URLs to queue
    for (const topic of initialTopics) {
      const url = crawlerConfig.baseUrl + topic.path;
      urlQueue.push({
        url,
        title: topic.name,
        depth: 0,
        path: [topic.name],
      });
      seenUrls.add(url);
    }

    totalFound = urlQueue.length;
    console.log(
      `Starting with ${urlQueue.length} initial pages, using ${crawlerConfig.concurrency} workers`,
    );

    // Create workers
    const workers: Promise<void>[] = [];
    for (let i = 0; i < crawlerConfig.concurrency; i++) {
      workers.push(createWorker(i, urlQueue, visitedUrls, seenUrls, pageContents, crawlerConfig));
    }

    // Monitor and report progress while workers are running
    const progressInterval = setInterval(() => {
      const queueSize = urlQueue.length;
      console.log(`Progress: ${processed}/${totalFound} pages processed, ${queueSize} in queue`);

      // Check if we're done or reached limits
      if (queueSize === 0 || processed >= crawlerConfig.maxPages || !isRunning) {
        clearInterval(progressInterval);
        isRunning = false;
      }
    }, 5000);

    // Wait for all workers to complete
    await Promise.all(workers);

    // Final progress report
    console.log(
      `\nCrawling complete: Processed ${processed} pages out of ${totalFound} discovered`,
    );

    // Save results
    await saveResults(pageContents, crawlerConfig.outputJsonFile, crawlerConfig);

    console.log('Crawling completed successfully!');
  } catch (error) {
    console.error('Fatal error during crawl:', error);

    // Try to save partial results
    if (pageContents.length > 0) {
      await saveResults(pageContents, crawlerConfig.outputJsonFile, crawlerConfig);
    }
    
    throw error;
  }

  // Helper function to create a worker
  async function createWorker(
    id: number,
    queue: QueueItem[],
    visited: Set<string>,
    seen: Set<string>,
    results: PageContent[],
    config: CrawlerConfig,
  ): Promise<void> {
    console.log(`Starting worker ${id}`);

    // Launch a browser for this worker
    const browser = await chromium.launch({ headless: true });
    browsers.push(browser);

    const context = await browser.newContext({
      userAgent:
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36',
    });

    const page = await context.newPage();
    page.setDefaultTimeout(config.pageTimeout);

    try {
      // Process URLs from the queue while running
      while (isRunning) {
        // Get a URL to process from the queue
        let currentItem: QueueItem | undefined;

        // Thread-safe queue access with locking
        if (queue.length > 0) {
          currentItem = queue.shift();
        }

        // If queue is empty or we've reached max pages, exit
        if (!currentItem || processed >= config.maxPages) {
          break;
        }

        const { url, title, depth, parentUrl, parentTitle, path = [] } = currentItem;

        // Skip if already visited
        if (visited.has(url)) {
          continue;
        }

        // Mark as visited to avoid duplicates across workers
        visited.add(url);
        processed++;

        try {
          console.log(`[Worker ${id}][${processed}/${totalFound}] Processing: ${title} (${url})`);

          // Load the page
          await page.goto(url, { timeout: config.pageTimeout }).catch((e) => {
            throw new Error(`Navigation failed: ${e.message}`);
          });

          // Wait for dynamic content
          await page.waitForTimeout(config.waitTime);

          // Extract page information
          const pageInfo = await page.evaluate(() => {
            // Better title detection
            const titleEl = document.querySelector('h1') || document.querySelector('title');
            const pageTitle = titleEl ? titleEl.textContent?.trim() : 'Untitled Page';

            // Get the page content
            const content = document.body?.innerHTML || '';

            // Extract links
            const links = Array.from(document.querySelectorAll('a[href]'))
              .map((a) => ({
                text: a.textContent?.trim() || '',
                href: a.getAttribute('href') || '',
              }))
              .filter((link) => link.href && !link.href.startsWith('javascript:'));

            return { pageTitle, content, links };
          });

          // Process links
          const processedLinks = pageInfo.links.map((link) => {
            let fullUrl = link.href;

            // Determine if it's an internal or external link
            const isInternal =
              !fullUrl.includes('://') ||
              (fullUrl.includes('://') && fullUrl.includes(config.baseUrl));

            // Make relative URLs absolute
            if (!fullUrl.includes('://')) {
              const urlObj = new URL(url);
              const baseDir = urlObj.href.substring(0, urlObj.href.lastIndexOf('/') + 1);
              fullUrl = new URL(fullUrl, baseDir).toString();
            }

            return {
              text: link.text,
              url: fullUrl,
              isInternal,
            };
          });

          // Save content with metadata
          results.push({
            url,
            title: pageInfo.pageTitle || title,
            rawHtml: pageInfo.content,
            metadata: {
              depth,
              path,
              crawlDate: new Date().toISOString(),
              parentUrl,
              parentTitle,
            },
            links: processedLinks,
          });

          // Add new links to the queue if not at max depth
          if (depth < config.maxDepth) {
            for (const link of processedLinks) {
              // Only add internal links that we haven't seen
              if (link.isInternal && link.url.includes('.htm') && !seen.has(link.url)) {
                // Create path for the new item
                const newPath = [...path];
                if (link.text) newPath.push(link.text);

                // Add to queue and mark as seen
                queue.push({
                  url: link.url,
                  title: link.text,
                  depth: depth + 1,
                  parentUrl: url,
                  parentTitle: pageInfo.pageTitle || title,
                  path: newPath,
                });

                seen.add(link.url);
                totalFound++;
              }
            }
          }
        } catch (error) {
          console.error(
            `  [Worker ${id}] Error processing ${url}: ${error instanceof Error ? error.message : String(error)}`,
          );
        }

        // Delay between requests
        await new Promise((resolve) => setTimeout(resolve, config.requestInterval));
      }
    } finally {
      // Clean up
      await context.close().catch(() => {});
      await browser.close().catch(() => {});
      console.log(`Worker ${id} finished`);
    }
  }

  // Save results to a JSON file
  async function saveResults(
    results: PageContent[], 
    outputFile: string, 
    config: CrawlerConfig
  ): Promise<void> {
    // Create output directory if needed
    await fs.mkdir(path.dirname(outputFile), { recursive: true });

    // Add crawl metadata
    const outputData = {
      metadata: {
        crawlDate: new Date().toISOString(),
        baseUrl: config.baseUrl,
        totalPages: results.length,
        maxDepth: config.maxDepth,
      },
      pages: results,
    };

    // Write the final JSON to file
    console.log(`Writing ${results.length} pages to ${outputFile}`);
    await fs.writeFile(outputFile, JSON.stringify(outputData, null, 2), 'utf8');

    const stats = await fs.stat(outputFile);
    const fileSizeMB = (stats.size / (1024 * 1024)).toFixed(2);
    console.log(`Output file size: ${fileSizeMB} MB`);
  }
}

// If run directly, execute the main function
if (require.main === module) {
  main().catch((error) => {
    console.error('Error:', error);
    process.exit(1);
  });
}
