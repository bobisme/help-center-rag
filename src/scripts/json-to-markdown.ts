#!/usr/bin/env bun
import fs from 'node:fs/promises';
import path from 'node:path';
import TurndownService from 'turndown';
import { JSDOM } from 'jsdom'; // Will need to add this with: bun add jsdom @types/jsdom

// Configuration
const CONFIG = {
  inputJsonFile: path.join(process.cwd(), 'output', 'epic-docs.json'),
  outputMarkdownFile: path.join(process.cwd(), 'output', 'epic-docs.md'),
  outputMetadataFile: path.join(process.cwd(), 'output', 'epic-metadata.json'),
};

// Types for our data structures
interface PageContent {
  url: string;
  title: string;
  rawHtml: string;
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

interface CrawlData {
  metadata: {
    crawlDate: string;
    baseUrl: string;
    totalPages: number;
    maxDepth: number;
  };
  pages: PageContent[];
}

// Process JSON data into Markdown
async function main() {
  console.log('Converting JSON to Markdown');
  console.log('===========================');

  try {
    // Read the JSON file
    console.log(`Reading from ${CONFIG.inputJsonFile}`);
    const jsonData = await fs.readFile(CONFIG.inputJsonFile, 'utf8');
    const crawlData: CrawlData = JSON.parse(jsonData);

    console.log(`Processing ${crawlData.pages.length} pages`);

    // Create a turndown service for HTML to Markdown conversion
    const turndown = new TurndownService({
      headingStyle: 'atx',
      codeBlockStyle: 'fenced',
      emDelimiter: '*',
      bulletListMarker: '-',
    });

    // Customize turndown for tables and other special content
    turndown.addRule('tables', {
      filter: ['table'],
      replacement: function (content) {
        return '\n\n' + content + '\n\n';
      },
    });
    
    // Let's use a simpler approach to fix nested lists
    // Instead of completely overriding the list processors, we'll post-process the markdown

    // Pre-process the HTML to remove junk before conversion
    function cleanHtml(html: string): string {
      // Remove script tags and their contents
      let cleaned = html.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');

      // Remove style tags and their contents
      cleaned = cleaned.replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '');

      // Remove CDATA sections
      cleaned = cleaned.replace(/\/\/<!\[CDATA\[[\s\S]*?\/\/\]\]>/g, '');

      // Remove JavaScript event handlers
      cleaned = cleaned.replace(/ on\w+="[^"]*"/g, '');

      // Remove common junk elements by their classes or IDs
      const junkSelectors = [
        '.PopupOver',
        '.PopupNotOver',
        '.PopupMenu',
        '.TextPopup',
        '#PopupMenu',
        '#TextPopup',
        '.BreadCrumb',
        '.NavigationBar',
        '.SearchBox',
        '.Banner',
        '.Header',
        '.Footer',
        '.NavButtons',
        '.Copyright',
        '.Buttons',
      ];

      // Use JSDOM for DOM manipulation
      try {
        const dom = new JSDOM(cleaned);
        const { document } = dom.window;

        // Remove junk elements
        junkSelectors.forEach((selector) => {
          document.querySelectorAll(selector).forEach((el) => {
            el?.parentNode?.removeChild(el);
          });
        });

        // Remove empty paragraphs and divs
        document.querySelectorAll('p, div').forEach((el) => {
          if (el.innerHTML.trim() === '') {
            el?.parentNode?.removeChild(el);
          }
        });

        // Remove JavaScript event attributes from all elements
        document.querySelectorAll('*').forEach((el) => {
          Array.from(el.attributes).forEach((attr) => {
            if (attr.name.startsWith('on') || attr.value.includes('javascript:')) {
              el.removeAttribute(attr.name);
            }
          });
        });

        // Remove specific junk elements that often appear in HTML help files
        const junkElements = [
          'script',
          'style',
          'iframe',
          'frame',
          'meta',
          'link',
          'noscript',
          'object',
          'embed',
          'applet',
        ];

        junkElements.forEach((tagName) => {
          document.querySelectorAll(tagName).forEach((el) => {
            el?.parentNode?.removeChild(el);
          });
        });

        // Return the cleaned HTML
        return dom.serialize();
      } catch (e) {
        console.warn('Error using JSDOM for cleaning:', e);
        // Fallback to regex-based cleaning

        // Remove inline JavaScript
        cleaned = cleaned.replace(/javascript:[^"')]+/g, 'javascript:void(0)');

        // Remove CSS-like content that leaked into the text
        cleaned = cleaned.replace(/\.\w+\s*{[^}]*}/g, '');

        // Remove multiple consecutive newlines
        cleaned = cleaned.replace(/\n{3,}/g, '\n\n');

        return cleaned;
      }
    }

    // Sort pages by path/hierarchy for better organization
    crawlData.pages.sort((a, b) => {
      // First by depth
      if (a.metadata.depth !== b.metadata.depth) {
        return a.metadata.depth - b.metadata.depth;
      }

      // Then by path string comparison
      const aPath = a.metadata.path.join('/');
      const bPath = b.metadata.path.join('/');
      return aPath.localeCompare(bPath);
    });

    // Prepare the markdown content
    let markdown = '# Applied Systems Epic Documentation\n\n';
    markdown += `_Generated on ${new Date().toLocaleString()}_\n\n`;

    // Add metadata
    markdown += '## About This Document\n\n';
    markdown += `- Source: ${crawlData.metadata.baseUrl}\n`;
    markdown += `- Pages: ${crawlData.metadata.totalPages}\n`;
    markdown += `- Generated: ${new Date(crawlData.metadata.crawlDate).toLocaleString()}\n`;
    markdown += `- Max Depth: ${crawlData.metadata.maxDepth}\n\n`;

    // Table of contents
    markdown += '## Table of Contents\n\n';

    // Build table of contents with depth-based indentation
    for (const page of crawlData.pages) {
      const indent = '  '.repeat(page.metadata.depth);
      markdown += `${indent}- [${page.title}](#${slugify(page.title)})\n`;
    }

    markdown += '\n\n';

    // Extract metadata for potential use in RAG systems
    const metadata = crawlData.pages.map((page) => ({
      url: page.url,
      title: page.title,
      path: page.metadata.path,
      depth: page.metadata.depth,
      parentUrl: page.metadata.parentUrl,
      parentTitle: page.metadata.parentTitle,
      crawlDate: page.metadata.crawlDate,
      linkCount: page.links.length,
      internalLinks: page.links.filter((l) => l.isInternal).length,
      externalLinks: page.links.filter((l) => !l.isInternal).length,
    }));

    // Convert each page and add to markdown
    for (const page of crawlData.pages) {
      // const headingLevel = Math.min(page.metadata.depth + 2, 6);
      // markdown += `${'#'.repeat(headingLevel)} ${page.title}\n\n`;
      // // Add path/breadcrumb
      // if (page.metadata.path.length > 0) {
      //   markdown += `_Path: ${page.metadata.path.join(' > ')}_\n\n`;
      // }
      // markdown += `_Source: <${page.url}>_\n\n`;

      // Clean the HTML to remove junk before conversion
      const cleanedHtml = cleanHtml(page.rawHtml);

      // Convert cleaned HTML to markdown
      const contentMarkdown = turndown.turndown(cleanedHtml);

      // Additional text cleanup for Markdown
      let cleanedMarkdown = contentMarkdown
        // Remove lines with just JavaScript references
        .replace(/^.*javascript:void\(0\).*$/gm, '')
        // Remove lines with CDATA or CSS-like content
        .replace(/^.*\/\/<!\[CDATA\[.*$/gm, '')
        .replace(/^.*\/\/\]\]>.*$/gm, '')
        .replace(/^\.[\w\s]+\{.*\}$/gm, '')
        // Remove lines that are just image buttons or icons
        // .replace(/^!\[.*\]\(.*\)$/gm, '')
        .replace(/^!\[.*btn.*\].*$/gm, '')
        .replace(/^!\[.*icon.*\].*$/gm, '')
        .replace(/^!\[See Also\].*$/gm, '')
        // Remove empty links and brackets
        .replace(/\[\]\(.*\)/g, '')
        .replace(/\[\]\[\]/g, '')
        // Remove text that looks like JavaScript function calls
        .replace(/^.*\w+\([\w\s'",.]*\).*$/gm, (match) => {
          // Only remove if it looks like a JS function call, not normal text in parentheses
          return match.includes('Init') || match.includes('POPUP') || match.includes('javascript')
            ? ''
            : match;
        })
        // Clean up TextPopup text
        .replace(/TextPopup.*POPUP\d+.*$/gm, '')
        // Fix double spaces
        .replace(/[ ]{2,}/g, ' ')
        // Fix multiple blank lines (more than 2)
        .replace(/\n{3,}/g, '\n\n');
        
      // Apply a line-by-line approach to fix formatting issues
      const lines = cleanedMarkdown.split('\n');
      const result = [];
      let inNumberedList = false;
      let prevIndent = 0;
      
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        
        // Check if this is a numbered list item
        const numMatch = line.match(/^(\s*)(\d+)\.(\s+)(.*)/);
        if (numMatch) {
          // This is a numbered list item
          const [_, indent, num, space, content] = numMatch;
          inNumberedList = true;
          prevIndent = indent.length;
          
          // Add a blank line before starting a new list (if not at start of document)
          if (i > 0 && !lines[i-1].trim().endsWith(':') && !/^\s*\d+\./.test(lines[i-1]) && lines[i-1].trim() !== '') {
            if (result[result.length - 1] !== '') {
              result.push('');
            }
          }
          
          // Add the numbered list item with consistent spacing
          result.push(`${indent}${num}. ${content}`);
        }
        // Check if this is a bullet point
        else if (line.match(/^\s*[-*]\s/)) {
          const bulletMatch = line.match(/^(\s*)[-*](\s+)(.*)/);
          if (bulletMatch) {
            const [_, indent, space, content] = bulletMatch;
            
            // If we're in a numbered list, indent the bullet points properly
            if (inNumberedList) {
              // Ensure bullet points are indented 4 spaces from the number
              result.push(`${' '.repeat(prevIndent + 4)}- ${content}`);
            } else {
              // Not in a numbered list, keep original indentation
              result.push(`${indent}- ${content}`);
            }
          } else {
            // No match, keep the line as is
            result.push(line);
          }
        }
        // Empty line
        else if (line.trim() === '') {
          result.push('');
          // End numbered list if we encounter an empty line
          inNumberedList = false;
        }
        // Regular line
        else {
          result.push(line);
        }
      }
      
      cleanedMarkdown = result.join('\n');
      
      // Final pass to fix any remaining issues
      cleanedMarkdown = cleanedMarkdown
        // Remove triple or more newlines
        .replace(/\n{3,}/g, '\n\n');

      markdown += cleanedMarkdown;
      markdown += '\n\n---\n\n';
    }

    // Write the files
    console.log(`Writing markdown to ${CONFIG.outputMarkdownFile}`);
    await fs.mkdir(path.dirname(CONFIG.outputMarkdownFile), { recursive: true });
    await fs.writeFile(CONFIG.outputMarkdownFile, markdown, 'utf8');

    console.log(`Writing metadata to ${CONFIG.outputMetadataFile}`);
    await fs.writeFile(CONFIG.outputMetadataFile, JSON.stringify(metadata, null, 2), 'utf8');

    // Calculate stats
    const markdownStats = await fs.stat(CONFIG.outputMarkdownFile);
    const markdownSizeMB = (markdownStats.size / (1024 * 1024)).toFixed(2);

    console.log(`Markdown file size: ${markdownSizeMB} MB`);

    // Estimate tokens
    const charCount = markdown.length;
    const estimatedTokens = Math.round(charCount / 4);
    console.log(`Estimated token count: ~${estimatedTokens.toLocaleString()} tokens`);

    console.log('Conversion completed successfully!');
  } catch (error) {
    console.error('Error during conversion:', error);
  }
}

// Helper function to convert text to URL-friendly slug
function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s_-]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

// Start the conversion
main().catch(console.error);
