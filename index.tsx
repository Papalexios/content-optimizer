import { GoogleGenAI, GenerateContentResponse, Type } from "@google/genai";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import React, { useState, useMemo, useEffect, useCallback, useReducer, useRef, memo } from 'react';
import ReactDOM from 'react-dom/client';

const AI_MODELS = {
    GEMINI_FLASH: 'gemini-2.5-flash',
    GEMINI_IMAGEN: 'imagen-4.0-generate-001',
    OPENAI_GPT4_TURBO: 'gpt-4o',
    OPENAI_DALLE3: 'dall-e-3',
    ANTHROPIC_OPUS: 'claude-3-7-sonnet-20250219',
    ANTHROPIC_HAIKU: 'claude-3-5-haiku-20241022',
    OPENROUTER_DEFAULT: [
        'google/gemini-2.5-flash',
        'anthropic/claude-3-haiku',
        'microsoft/wizardlm-2-8x22b',
        'openrouter/auto'
    ],
    GROQ_MODELS: [
        'llama-3.3-70b-versatile',
        'llama-3.1-8b-instant',
        'gemma2-9b-it',
        'llama3-70b-8192',
        'llama3-8b-8192',
        'mixtral-8x7b-32768',
        'gemma-7b-it',
        'meta-llama/llama-4-scout-17b-16e-instruct',
    ]
};


// ════════════════════════════════════════════════════════════════════════════════
// SCHEMA MARKUP HANDLER - PREVENTS SCHEMA IN VISIBLE CONTENT
// ════════════════════════════════════════════════════════════════════════════════

function extractSchemaFromContent(contentData) {
    const schemas = [];
    if (!contentData.schema) return schemas;

    if (contentData.schema.localBusiness) schemas.push(contentData.schema.localBusiness);
    if (contentData.schema.article) schemas.push(contentData.schema.article);
    if (contentData.schema.faqPage) schemas.push(contentData.schema.faqPage);
    if (contentData.schema.videoObjects) schemas.push(...contentData.schema.videoObjects);
    if (contentData.schema.breadcrumb) schemas.push(contentData.schema.breadcrumb);
    if (contentData.schema.organization) schemas.push(contentData.schema.organization);

    console.log(`✅ Extracted ${schemas.length} schema types`);
    return schemas;
}

function generateSchemaMarkup(schemas) {
    if (!schemas || schemas.length === 0) return '';
    const schemaObject = {"@context": "https://schema.org", "@graph": schemas};
    return `<script type="application/ld+json">\n${JSON.stringify(schemaObject, null, 2)}\n</script>`;
}

function renderCleanArticleContent(contentData) {
    let html = '';

    if (contentData.title) html += `<h1>${contentData.title}</h1>\n\n`;

    if (contentData.author) {
        html += `<div class="author-info">\n<p><strong>By ${contentData.author.name}</strong></p>\n`;
        if (contentData.author.credentials) html += `<p>${contentData.author.credentials}</p>\n`;
        if (contentData.author.bio) html += `<p><em>${contentData.author.bio}</em></p>\n`;
        html += `</div>\n\n`;
    }

    if (contentData.content?.introduction) html += contentData.content.introduction + '\n\n';

    if (contentData.content?.sections) {
        contentData.content.sections.forEach(section => {
            html += `<h2>${section.heading}</h2>\n${section.content}\n\n`;
            if (section.subsections) {
                section.subsections.forEach(sub => {
                    html += `<h3>${sub.subheading}</h3>\n${sub.content}\n\n`;
                });
            }
        });
    }

    if (contentData.youtubeVideos) {
        contentData.youtubeVideos.forEach(video => {
            if (video.embedContext) html += `<p>${video.embedContext}</p>\n`;
            html += `<div class="video-container">\n`;
            html += `<iframe width="560" height="315" src="https://www.youtube.com/embed/${video.videoId}" `;
            html += `frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>\n`;
            html += `</div>\n\n`;
        });
    }

    if (contentData.content?.faqSection) {
        html += `<h2>Frequently Asked Questions</h2>\n\n`;
        contentData.content.faqSection.forEach(faq => {
            html += `<h3>${faq.question}</h3>\n<p>${faq.answer}</p>\n\n`;
        });
    }

    if (contentData.content?.conclusion) html += contentData.content.conclusion + '\n\n';

    if (contentData.externalReferences?.length > 0) {
        html += `<h2>References</h2>\n<ol>\n`;
        contentData.externalReferences.forEach(ref => {
            html += `<li><a href="${ref.url}" target="_blank" rel="nofollow noopener">${ref.source}</a></li>\n`;
        });
        html += `</ol>\n`;
    }

    // ⚠️ CRITICAL: NO SCHEMA MARKUP IN CONTENT!
    return html;
}

// ════════════════════════════════════════════════════════════════════════════════
// WORD COUNT ENFORCEMENT (2,500-3,000 WORDS MANDATORY)
// ════════════════════════════════════════════════════════════════════════════════

function enforceWordCount(content, minWords = 2500, maxWords = 3000) {
    const textOnly = content.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
    const words = textOnly.split(/\s+/).filter(w => w.length > 0);
    const wordCount = words.length;

    console.log(`📊 Word Count: ${wordCount} (target: ${minWords}-${maxWords})`);

    if (wordCount < minWords) {
        throw new ContentTooShortError(`CONTENT TOO SHORT: ${wordCount} words (minimum ${minWords} required)`, content, wordCount);
    }

    if (wordCount > maxWords) {
        console.warn(`⚠️  Content is ${wordCount - maxWords} words over target`);
    }

    return wordCount;
}

function checkHumanWritingScore(content) {
    const aiPhrases = [
        'delve into', 'in today\'s digital landscape', 'revolutionize', 'game-changer',
        'unlock', 'leverage', 'robust', 'seamless', 'cutting-edge', 'elevate', 'empower',
        'it\'s important to note', 'it\'s worth mentioning', 'needless to say',
        'in conclusion', 'to summarize', 'in summary', 'holistic', 'paradigm shift',
        'utilize', 'commence', 'endeavor', 'facilitate', 'implement', 'demonstrate',
        'ascertain', 'procure', 'terminate', 'disseminate', 'expedite',
        'in order to', 'due to the fact that', 'for the purpose of', 'with regard to',
        'in the event that', 'at this point in time', 'for all intents and purposes',
        'furthermore', 'moreover', 'additionally', 'consequently', 'nevertheless',
        'notwithstanding', 'aforementioned', 'heretofore', 'whereby', 'wherein',
        'landscape', 'realm', 'sphere', 'domain', 'ecosystem', 'framework',
        'navigate', 'embark', 'journey', 'transform', 'transition',
        'plethora', 'myriad', 'multitude', 'abundance', 'copious',
        'crucial', 'vital', 'essential', 'imperative', 'paramount',
        'optimize', 'maximize', 'enhance', 'augment', 'amplify',
        'intricate', 'nuanced', 'sophisticated', 'elaborate', 'comprehensive',
        'comprehensive guide', 'ultimate guide', 'complete guide',
        'dive deep', 'take a deep dive', 'let\'s explore', 'let\'s dive in'
    ];

    let aiScore = 0;
    const lowerContent = content.toLowerCase();

    aiPhrases.forEach(phrase => {
        const count = (lowerContent.match(new RegExp(phrase, 'g')) || []).length;
        if (count > 0) {
            aiScore += (count * 10);
            console.warn(`⚠️  AI phrase detected ${count}x: "${phrase}"`);
        }
    });

    const sentences = content.match(/[^.!?]+[.!?]+/g) || [];
    if (sentences.length > 0) {
        const avgLength = sentences.reduce((sum, s) => sum + s.split(/\s+/).length, 0) / sentences.length;
        if (avgLength > 25) {
            aiScore += 15;
            console.warn(`⚠️  Average sentence too long (${avgLength.toFixed(1)} words)`);
        }
    }

    const humanScore = Math.max(0, 100 - aiScore);
    console.log(`🤖 Human Writing Score: ${humanScore}% (target: 100%)`);

    return humanScore;
}

console.log('✅ Schema handler & word count enforcer loaded');


// ═══════════════════════════════════════════════════════════════════════════════
// 🎥 YOUTUBE VIDEO DEDUPLICATION - CRITICAL FIX FOR DUPLICATE VIDEOS
// ═══════════════════════════════════════════════════════════════════════════════

function getUniqueYoutubeVideos(videos, count = 2) {
    if (!videos || videos.length === 0) {
        console.warn('⚠️  No YouTube videos provided');
        return null;
    }

    const uniqueVideos = [];
    const usedVideoIds = new Set();

    for (const video of videos) {
        if (uniqueVideos.length >= count) break;

        const videoId = video.videoId || 
                       video.embedUrl?.match(/embed\/([^?&]+)/)?.[1] ||
                       video.url?.match(/[?&]v=([^&]+)/)?.[1] ||
                       video.url?.match(/youtu\.be\/([^?&]+)/)?.[1];

        if (videoId && !usedVideoIds.has(videoId)) {
            usedVideoIds.add(videoId);
            uniqueVideos.push({
                ...video,
                videoId: videoId,
                embedUrl: `https://www.youtube.com/embed/${videoId}`
            });
            console.log(`✅ Video ${uniqueVideos.length} selected: ${videoId} - "${(video.title || '').substring(0, 50)}..."`);
        } else if (videoId) {
            console.warn(`⚠️  Duplicate video skipped: ${videoId}`);
        }
    }

    if (uniqueVideos.length < 2) {
        console.error(`❌ Only ${uniqueVideos.length} unique video(s) found. Need 2 for quality content.`);
    } else {
        console.log(`✅ Video deduplication complete: ${uniqueVideos.length} unique videos ready`);
    }

    return uniqueVideos.length > 0 ? uniqueVideos : null;
}

console.log('✅ YouTube video deduplication function loaded');



// ==========================================
// CONTENT & SEO REQUIREMENTS
// ==========================================
const TARGET_MIN_WORDS = 2200; // Increased for higher quality
const TARGET_MAX_WORDS = 2800;
const TARGET_MIN_WORDS_PILLAR = 3500; // Increased for depth
const TARGET_MAX_WORDS_PILLAR = 4500;
const YOUTUBE_EMBED_COUNT = 2;
const MIN_INTERNAL_LINKS = 8; // User wants 8-12, this is the floor
const MAX_INTERNAL_LINKS = 15;
const MIN_TABLES = 3;
const FAQ_COUNT = 8;
const KEY_TAKEAWAYS = 8;

// SEO Power Words
const POWER_WORDS = ['Ultimate', 'Complete', 'Essential', 'Proven', 'Secret', 'Powerful', 'Effective', 'Simple', 'Fast', 'Easy', 'Best', 'Top', 'Expert', 'Advanced', 'Master', 'Definitive', 'Comprehensive', 'Strategic', 'Revolutionary', 'Game-Changing'];

// Track videos to prevent duplicates
const usedVideoUrls = new Set();


// --- START: Performance & Caching Enhancements ---

/**
 * A sophisticated caching layer for API responses to reduce redundant calls
 * and improve performance within a session.
 */
class ContentCache {
  private cache = new Map<string, {data: any, timestamp: number}>();
  private TTL = 3600000; // 1 hour
  
  set(key: string, data: any) {
    this.cache.set(key, {data, timestamp: Date.now()});
  }
  
  get(key: string): any | null {
    const item = this.cache.get(key);
    if (item && Date.now() - item.timestamp < this.TTL) {
      console.log(`[Cache] HIT for key: ${key}`);
      return item.data;
    }
    console.log(`[Cache] MISS for key: ${key}`);
    return null;
  }
}
const apiCache = new ContentCache();

// --- END: Performance & Caching Enhancements ---


// --- START: Core Utility Functions ---

// Debounce function to limit how often a function gets called
const debounce = (func: (...args: any[]) => void, delay: number) => {
    let timeoutId: ReturnType<typeof setTimeout>;
    return (...args: any[]) => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
            func.apply(null, args);
        }, delay);
    };
};


/**
 * A highly resilient function to extract a JSON object from a string.
 * It surgically finds the JSON boundaries by balancing brackets, strips conversational text and markdown,
 * and automatically repairs common syntax errors like trailing commas.
 * @param text The raw string response from the AI, which may contain conversational text.
 * @returns The clean, valid JSON object.
 * @throws {Error} if a valid JSON object cannot be found or parsed.
 */
const extractJson = (text: string): string => {
    if (!text || typeof text !== 'string') {
        throw new Error("Input text is invalid or empty.");
    }
    
    // First, try a simple parse. If it's valid, we're done.
    try {
        JSON.parse(text);
        return text;
    } catch (e) { /* Not valid, proceed with cleaning */ }

    // Aggressively clean up common conversational text and markdown fences.
    let cleanedText = text
        .replace(/^```(?:json)?\s*/, '') // Remove opening ```json or ```
        .replace(/\s*```$/, '')           // Remove closing ```
        .trim();

    // Remove any remaining markdown blocks
    cleanedText = cleanedText.replace(/```json\s*/gi, '').replace(/```\s*/g, '');

    // Remove trailing commas before closing brackets  
    cleanedText = cleanedText.replace(/,(\s*[}\]])/g, '$1');

    // Find the first real start of a JSON object or array.
    const firstBracket = cleanedText.indexOf('{');
    const firstSquare = cleanedText.indexOf('[');
    
    if (firstBracket === -1 && firstSquare === -1) {
        console.error(`[extractJson] No JSON start characters ('{' or '[') found after cleanup.`, { originalText: text });
        throw new Error("No JSON object/array found. Ensure your prompt requests JSON output only without markdown.");
    }

    let startIndex = -1;
    if (firstBracket === -1) startIndex = firstSquare;
    else if (firstSquare === -1) startIndex = firstBracket;
    else startIndex = Math.min(firstBracket, firstSquare);

    let potentialJson = cleanedText.substring(startIndex);
    
    // Find the balanced end bracket for the structure.
    const startChar = potentialJson[0];
    const endChar = startChar === '{' ? '}' : ']';
    
    let balance = 1;
    let inString = false;
    let escapeNext = false;
    let endIndex = -1;

    for (let i = 1; i < potentialJson.length; i++) {
        const char = potentialJson[i];
        
        if (escapeNext) {
            escapeNext = false;
            continue;
        }
        
        if (char === '\\') {
            escapeNext = true;
            continue;
        }
        
        if (char === '"' && !escapeNext) {
            inString = !inString;
        }
        
        if (inString) continue;

        if (char === startChar) balance++;
        else if (char === endChar) balance--;

        if (balance === 0) {
            endIndex = i;
            break;
        }
    }

    let jsonCandidate;
    if (endIndex !== -1) {
        jsonCandidate = potentialJson.substring(0, endIndex + 1);
    } else {
        jsonCandidate = potentialJson;
        if (balance > 0) {
            console.warn(`[extractJson] Could not find a balanced closing bracket (unclosed structures: ${balance}). The response may be truncated. Attempting to auto-close.`);
            jsonCandidate += endChar.repeat(balance);
        } else {
             console.warn("[extractJson] Could not find a balanced closing bracket. The AI response may have been truncated.");
        }
    }

    // Attempt to parse the candidate string.
    try {
        JSON.parse(jsonCandidate);
        return jsonCandidate;
    } catch (e) {
        // If parsing fails, try to repair common issues like trailing commas.
        console.warn("[extractJson] Initial parse failed. Attempting to repair trailing commas.");
        try {
            const repaired = jsonCandidate.replace(/,(?=\s*[}\]])/g, '');
            JSON.parse(repaired);
            return repaired;
        } catch (repairError: any) {
            console.error(`[extractJson] CRITICAL FAILURE: Parsing failed even after repair.`, { 
                errorMessage: repairError.message,
                attemptedToParse: jsonCandidate
            });
            throw new Error(`Unable to parse JSON from AI response after multiple repair attempts.`);
        }
    }
};


/**
 * Extracts a YouTube video ID from various URL formats.
 * @param url The YouTube URL.
 * @returns The 11-character video ID or null if not found.
 */
const extractYouTubeID = (url: string): string | null => {
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*/;
    const match = url.match(regExp);
    if (match && match[2].length === 11) {
        return match[2];
    }
    return null;
};


/**
 * Extracts the final, clean slug from a URL, intelligently removing parent paths and file extensions.
 * This ensures a perfect match with the WordPress database slug.
 * @param urlString The full URL to parse.
 * @returns The extracted slug.
 */
const extractSlugFromUrl = (urlString: string): string => {
    try {
        const url = new URL(urlString);
        let pathname = url.pathname;

        // 1. Remove trailing slash to handle URLs like /my-post/
        if (pathname.endsWith('/') && pathname.length > 1) {
            pathname = pathname.slice(0, -1);
        }

        // 2. Get the last segment after the final '/'
        const lastSegment = pathname.substring(pathname.lastIndexOf('/') + 1);

        // 3. Remove common web file extensions like .html, .php, etc.
        const cleanedSlug = lastSegment.replace(/\.[a-zA-Z0-9]{2,5}$/, '');

        return cleanedSlug;
    } catch (error) {
        console.error("Could not parse URL to extract slug:", urlString, error);
        // Fallback for non-URL strings, though unlikely
        return urlString.split('/').pop() || '';
    }
};


/**
 * A more professional and resilient fetch function for AI APIs that includes
 * exponential backoff for retries and intelligently fails fast on non-retriable errors.
 * This is crucial for handling rate limits (429) and transient server issues (5xx)
 * while avoiding wasted time on client-side errors (4xx).
 * @param apiCall A function that returns the promise from the AI SDK call.
 * @param maxRetries The maximum number of times to retry the call.
 * @param initialDelay The baseline delay in milliseconds for the first retry.
 * @returns The result of the successful API call.
 * @throws {Error} if the call fails after all retries or on a non-retriable error.
 */
// FIX: Changed catch parameter from `e: unknown` to `error: any` to allow direct property access and fix type errors.
const callAiWithRetry = async (apiCall: () => Promise<any>, maxRetries = 5, initialDelay = 5000) => {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            return await apiCall();
        } catch (error: any) {
            console.error(`AI call failed on attempt ${attempt + 1}. Error:`, error);

            const errorMessage = (error.message || '').toLowerCase();
            // Try to get status from error object, or parse it from the message as a fallback.
            const statusMatch = errorMessage.match(/\[(\d{3})[^\]]*\]/); 
            const statusCode = error.status || (statusMatch ? parseInt(statusMatch[1], 10) : null);

            const isNonRetriableClientError = statusCode && statusCode >= 400 && statusCode < 500 && statusCode !== 429;
            const isContextLengthError = errorMessage.includes('context length') || errorMessage.includes('token limit');
            const isInvalidApiKeyError = errorMessage.includes('api key not valid');

            if (isNonRetriableClientError || isContextLengthError || isInvalidApiKeyError) {
                 console.error(`Encountered a non-retriable error (Status: ${statusCode}, Message: ${error.message}). Failing immediately.`);
                 throw error; // Fail fast.
            }

            // If it's the last attempt for any retriable error, give up.
            if (attempt === maxRetries - 1) {
                console.error(`AI call failed on final attempt (${maxRetries}).`);
                throw error;
            }
            
            let delay: number;
            // --- Intelligent Rate Limit Handling ---
            if (error.status === 429 || statusCode === 429) {
                // Respect the 'Retry-After' header if the provider sends it. This is the gold standard.
                const retryAfterHeader = error.headers?.['retry-after'] || error.response?.headers?.get('retry-after');
                if (retryAfterHeader) {
                    const retryAfterSeconds = parseInt(retryAfterHeader, 10);
                    if (!isNaN(retryAfterSeconds)) {
                        // The value is in seconds.
                        delay = retryAfterSeconds * 1000 + 500; // Add a 500ms buffer.
                        console.log(`Rate limit hit. Provider requested a delay of ${retryAfterSeconds}s. Waiting...`);
                    } else {
                        // The value might be an HTTP-date.
                        const retryDate = new Date(retryAfterHeader);
                        if (!isNaN(retryDate.getTime())) {
                            delay = retryDate.getTime() - new Date().getTime() + 500; // Add buffer.
                             console.log(`Rate limit hit. Provider requested waiting until ${retryDate.toISOString()}. Waiting...`);
                        } else {
                             // Fallback if the date format is unexpected.
                             delay = initialDelay * Math.pow(2, attempt) + (Math.random() * 1000);
                             console.log(`Rate limit hit. Could not parse 'Retry-After' header ('${retryAfterHeader}'). Using exponential backoff.`);
                        }
                    }
                } else {
                    // If no 'Retry-After' header, use our more patient exponential backoff.
                    delay = initialDelay * Math.pow(2, attempt) + (Math.random() * 1000);
                    console.log(`Rate limit hit. No 'Retry-After' header found. Using exponential backoff.`);
                }
            } else {
                 // --- Standard Exponential Backoff for Server-Side Errors (5xx) ---
                 const backoff = Math.pow(2, attempt);
                 const jitter = Math.random() * 1000;
                 delay = initialDelay * backoff + jitter;
            }

            console.log(`Retrying in ${Math.round(delay)}ms...`);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
    throw new Error("AI call failed after all retries.");
};

/**
 * Fetches a URL by first attempting a direct connection, then falling back to a
 * series of public CORS proxies. This strategy makes the sitemap crawling feature
 * significantly more resilient to CORS issues and unreliable proxies.
 * @param url The target URL to fetch.
 * @param options The options for the fetch call (method, headers, body).
 * @returns The successful Response object.
 * @throws {Error} if the direct connection and all proxies fail.
 */
const fetchWithProxies = async (url: string, options: RequestInit = {}): Promise<Response> => {
    let lastError: Error | null = null;
    const REQUEST_TIMEOUT = 20000; // 20 seconds

    // Standard headers to mimic a browser request, reducing the chance of being blocked.
    const browserHeaders = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
    };

    // --- NEW: Attempt a direct fetch first ---
    // This will work if the server has CORS enabled, and is the fastest option.
    try {
        console.log("Attempting direct fetch (no proxy)...");
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);
        const directResponse = await fetch(url, {
            ...options,
            headers: {
                ...browserHeaders,
                ...(options.headers || {}),
            },
            signal: controller.signal,
        });
        clearTimeout(timeoutId);
        if (directResponse.ok) {
            console.log("Successfully fetched directly (no proxy)!");
            return directResponse;
        }
    } catch (error: any) {
        // A TypeError here is the classic sign of a CORS error.
        if (error.name !== 'AbortError') { // Don't log timeout as a CORS error
            console.warn("Direct fetch failed (likely due to CORS). Proceeding with proxies.", error.name);
        }
        lastError = error;
    }

    // --- END: Direct fetch attempt ---

    const encodedUrl = encodeURIComponent(url);
    // An expanded and diversified list of public CORS proxies.
    const proxies = [
        `https://corsproxy.io/?${url}`,
        `https://api.codetabs.com/v1/proxy?quest=${encodedUrl}`,
        `https://api.allorigins.win/raw?url=${encodedUrl}`,
        `https://thingproxy.freeboard.io/fetch/${url}`,
    ];


    for (let i = 0; i < proxies.length; i++) {
        const proxyUrl = proxies[i];
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

        try {
            const shortProxyUrl = new URL(proxyUrl).hostname;
            console.log(`Attempting fetch via proxy #${i + 1} (${shortProxyUrl})...`);
            
            const response = await fetch(proxyUrl, {
                ...options,
                 headers: {
                    ...browserHeaders,
                    ...(options.headers || {}),
                },
                signal: controller.signal,
            });

            if (response.ok) {
                console.log(`Successfully fetched via proxy #${i + 1} (${shortProxyUrl})`);
                return response; // Success!
            }
            const responseText = await response.text().catch(() => `(could not read response body)`);
            lastError = new Error(`Proxy request failed with status ${response.status} for ${shortProxyUrl}. Response: ${responseText.substring(0, 100)}`);

        } catch (error: any) {
            if (error.name === 'AbortError') {
                const shortProxyUrl = new URL(proxyUrl).hostname;
                console.error(`Fetch via proxy #${i + 1} (${shortProxyUrl}) timed out after ${REQUEST_TIMEOUT / 1000}s.`);
                lastError = new Error(`Request timed out for proxy: ${shortProxyUrl}`);
            } else {
                console.error(`Fetch via proxy #${i + 1} failed:`, error);
                lastError = error as Error;
            }
        } finally {
            clearTimeout(timeoutId);
        }
    }

    // If we're here, all proxies failed.
    const baseErrorMessage = "Failed to crawl your sitemap. This is often due to a network or CORS issue where our proxy servers are blocked by your website's security (like Cloudflare or a firewall), or the target server is too slow to respond.\n\n" +
        "Please check that:\n" +
        "1. Your sitemap URL is correct and publicly accessible.\n" +
        "2. Your website's security settings aren't blocking anonymous proxy access.\n"
        "3. Your internet connection is stable.";

    throw new Error(lastError ? `${baseErrorMessage}\n\nLast Error: ${lastError.message}` : baseErrorMessage);
};


/**
 * Smartly fetches a WordPress API endpoint. If the request is authenticated, it forces a direct
 * connection, as proxies will strip authentication headers. Unauthenticated requests will use
 * the original proxy fallback logic.
 * @param targetUrl The full URL to the WordPress API endpoint.
 * @param options The options for the fetch call (method, headers, body).
 * @returns The successful Response object.
 * @throws {Error} if the connection fails.
 */
const fetchWordPressWithRetry = async (targetUrl: string, options: RequestInit): Promise<Response> => {
    const REQUEST_TIMEOUT = 30000; // 30 seconds for potentially large uploads
    const hasAuthHeader = options.headers && (options.headers as Record<string, string>)['Authorization'];

    // If the request has an Authorization header, it MUST be a direct request.
    // Proxies will strip authentication headers and cause a guaranteed failure.
    if (hasAuthHeader) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);
            const directResponse = await fetch(targetUrl, { ...options, signal: controller.signal });
            clearTimeout(timeoutId);
            return directResponse; // Return the response directly, regardless of status, to be handled by the caller.
        } catch (error: any) {
            if (error.name === 'AbortError') {
                throw new Error("WordPress API request timed out.");
            }
            // A TypeError is the classic sign of a CORS error on a failed fetch.
            // This will be caught and diagnosed by the calling function (e.g., verifyWpConnection)
            throw error;
        }
    }

    // --- Fallback to original proxy logic for NON-AUTHENTICATED requests ---
    let lastError: Error | null = null;
    
    // 1. Attempt Direct Connection
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);
        const directResponse = await fetch(targetUrl, { ...options, signal: controller.signal });
        clearTimeout(timeoutId);

        if (directResponse.ok || (directResponse.status >= 400 && directResponse.status < 500)) {
            return directResponse;
        }
        lastError = new Error(`Direct connection failed with status ${directResponse.status}`);
    } catch (error: any) {
        if (error.name !== 'AbortError') {
            console.warn("Direct WP API call failed (likely CORS or network issue). Trying proxies.", error.name);
        }
        lastError = error;
    }
    
    // 2. Attempt with Proxies if Direct Fails
    const encodedUrl = encodeURIComponent(targetUrl);
    const proxies = [
        `https://corsproxy.io/?${encodedUrl}`,
        `https://api.allorigins.win/raw?url=${encodedUrl}`,
    ];

    for (const proxyUrl of proxies) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);
        try {
            const shortProxyUrl = new URL(proxyUrl).hostname;
            console.log(`Attempting WP API call via proxy: ${shortProxyUrl}`);
            const response = await fetch(proxyUrl, { ...options, signal: controller.signal });
            if (response.ok || (response.status >= 400 && response.status < 500)) {
                console.log(`Successfully fetched via proxy: ${shortProxyUrl}`);
                return response;
            }
            const responseText = await response.text().catch(() => '(could not read response body)');
            lastError = new Error(`Proxy request failed with status ${response.status} for ${shortProxyUrl}. Response: ${responseText.substring(0, 100)}`);
        } catch (error: any) {
             if (error.name === 'AbortError') {
                const shortProxyUrl = new URL(proxyUrl).hostname;
                console.error(`Fetch via proxy ${shortProxyUrl} timed out.`);
                lastError = new Error(`Request timed out for proxy: ${shortProxyUrl}`);
            } else {
                lastError = error;
            }
        } finally {
            clearTimeout(timeoutId);
        }
    }

    throw lastError || new Error("All attempts to connect to the WordPress API failed.");
};


/**
 * Processes an array of items concurrently using async workers, with a cancellable mechanism.
 * @param items The array of items to process.
 * @param processor An async function that processes a single item.
 * @param concurrency The number of parallel workers.
 * @param onProgress An optional callback to track progress.
 * @param shouldStop An optional function that returns true to stop processing.
 */
async function processConcurrently<T>(
    items: T[],
    processor: (item: T) => Promise<void>,
    concurrency = 5,
    onProgress?: (completed: number, total: number) => void,
    shouldStop?: () => boolean
): Promise<void> {
    const queue = [...items];
    let completed = 0;
    const total = items.length;

    const run = async () => {
        while (queue.length > 0) {
            if (shouldStop?.()) {
                // Emptying the queue is a robust way to signal all workers to stop
                // after they finish their current task.
                queue.length = 0;
                break;
            }
            const item = queue.shift();
            if (item) {
                await processor(item);
                completed++;
                onProgress?.(completed, total);
            }
        }
    };

    const workers = Array(concurrency).fill(null).map(run);
    await Promise.all(workers);
};

/**
 * Validates and repairs internal link placeholders from AI content. If an AI invents a slug,
 * this "Smart Link Forger" finds the best matching real page based on anchor text and repairs the link.
 * @param content The HTML content string with AI-generated placeholders.
 * @param availablePages An array of page objects from the sitemap, each with 'id', 'title', and 'slug'.
 * @returns The HTML content with invalid link placeholders repaired or removed.
 */
const validateAndRepairInternalLinks = (content: string, availablePages: any[]): string => {
    if (!content || !availablePages || availablePages.length === 0) {
        return content;
    }

    const pagesBySlug = new Map(availablePages.map(p => [p.slug, p]));
    const placeholderRegex = /\[INTERNAL_LINK\s+slug="([^"]+)"\s+text="([^"]+)"\]/g;

    return content.replace(placeholderRegex, (match, slug, text) => {
        // If the slug is valid and exists, we're good.
        if (pagesBySlug.has(slug)) {
            return match; // Return the original placeholder unchanged.
        }

        // --- Slug is INVALID. AI invented it. Time to repair. ---
        console.warn(`[Link Repair] AI invented slug "${slug}". Attempting to repair based on anchor text: "${text}".`);

        const anchorTextLower = text.toLowerCase();
        const anchorWords = anchorTextLower.split(/\s+/).filter(w => w.length > 2); // Meaningful words
        const anchorWordSet = new Set(anchorWords);
        let bestMatch: any = null;
        let highestScore = -1;

        for (const page of availablePages) {
            if (!page.slug || !page.title) continue;

            let currentScore = 0;
            const titleLower = page.title.toLowerCase();

            // Scoring Algorithm
            // 1. Exact title match (very high confidence)
            if (titleLower === anchorTextLower) {
                currentScore += 100;
            }
            
            // 2. Partial inclusion (high confidence)
            // - Anchor text is fully inside the title (e.g., anchor "SEO tips" in title "Advanced SEO Tips for 2025")
            if (titleLower.includes(anchorTextLower)) {
                currentScore += 60;
            }
            // - Title is fully inside the anchor text (rarer, but possible)
            if (anchorTextLower.includes(titleLower)) {
                currentScore += 50;
            }

            // 3. Keyword Overlap Score (the core of the enhancement)
            const titleWords = titleLower.split(/\s+/).filter(w => w.length > 2);
            if (titleWords.length === 0) continue; // Avoid division by zero
            
            const titleWordSet = new Set(titleWords);
            const intersection = new Set([...anchorWordSet].filter(word => titleWordSet.has(word)));
            
            if (intersection.size > 0) {
                // Calculate a relevance score based on how many words match
                const anchorMatchPercentage = (intersection.size / anchorWordSet.size) * 100;
                const titleMatchPercentage = (intersection.size / titleWordSet.size) * 100;
                // Average the two percentages. This rewards matches that are significant to both the anchor and the title.
                const overlapScore = (anchorMatchPercentage + titleMatchPercentage) / 2;
                currentScore += overlapScore;
            }

            if (currentScore > highestScore) {
                highestScore = currentScore;
                bestMatch = page;
            }
        }
        
        // Use a threshold to avoid bad matches
        if (bestMatch && highestScore > 50) {
            console.log(`[Link Repair] Found best match: "${bestMatch.slug}" with a score of ${highestScore.toFixed(2)}. Forging corrected link.`);
            const sanitizedText = text.replace(/"/g, '&quot;');
            return `[INTERNAL_LINK slug="${bestMatch.slug}" text="${sanitizedText}"]`;
        } else {
            console.warn(`[Link Repair] Could not find any suitable match for slug "${slug}" (best score: ${highestScore.toFixed(2)}). Removing link, keeping text.`);
            return text; // Fallback: If no good match, just return the anchor text.
        }
    });
};

/**
 * The "Link Quota Guardian": Programmatically ensures the final content meets a minimum internal link count.
 * If the AI-generated content is deficient, this function finds relevant keywords in the text and injects
 * new, 100% correct internal link placeholders.
 * @param content The HTML content, post-repair.
 * @param availablePages The sitemap page data.
 * @param primaryKeyword The primary keyword of the article being generated.
 * @param minLinks The minimum number of internal links required.
 * @returns The HTML content with the link quota enforced.
 */
const enforceInternalLinkQuota = (content: string, availablePages: any[], primaryKeyword: string, minLinks: number): string => {
    if (!availablePages || availablePages.length === 0) return content;

    const placeholderRegex = /\[INTERNAL_LINK\s+slug="[^"]+"\s+text="[^"]+"\]/g;
    const existingLinks = [...content.matchAll(placeholderRegex)];
    const linkedSlugs = new Set(existingLinks.map(match => match[1]));

    let deficit = minLinks - existingLinks.length;
    if (deficit <= 0) {
        return content; // Quota already met.
    }

    console.log(`[Link Guardian] Link deficit detected. Need to add ${deficit} more links.`);

    let newContent = content;

    // 1. Create a pool of high-quality candidate pages that are not already linked.
    const candidatePages = availablePages
        .filter(p => p.slug && p.title && !linkedSlugs.has(p.slug) && p.title.split(' ').length > 2) // Filter out pages with very short/generic titles
        .map(page => {
            const title = page.title;
            // Create a prioritized list of search terms from the page title.
            const searchTerms = [
                title, // 1. Full title (highest priority)
                // 2. Sub-phrases (e.g., from "The Ultimate Guide to SEO" -> "Ultimate Guide to SEO", "Guide to SEO")
                ...title.split(' ').length > 4 ? [title.split(' ').slice(0, -1).join(' ')] : [], // all but last word
                ...title.split(' ').length > 3 ? [title.split(' ').slice(1).join(' ')] : [],    // all but first word
            ]
            .filter((v, i, a) => a.indexOf(v) === i && v.length > 10) // Keep unique terms of reasonable length
            .sort((a, b) => b.length - a.length); // Sort by length, longest first

            return { ...page, searchTerms };
        })
        .filter(p => p.searchTerms.length > 0);

    // This tracks which pages we've successfully added a link for in this run to avoid duplicate links.
    const newlyLinkedSlugs = new Set<string>();

    for (const page of candidatePages) {
        if (deficit <= 0) break;
        if (newlyLinkedSlugs.has(page.slug)) continue;

        let linkPlaced = false;
        for (const term of page.searchTerms) {
            if (linkPlaced) break;

            // This advanced regex finds the search term as plain text, avoiding matches inside existing HTML tags or attributes.
            // It looks for the term preceded by a tag closing `>` or whitespace, and followed by punctuation, whitespace, or a tag opening `<`.
            const searchRegex = new RegExp(`(?<=[>\\s\n\t(])(${escapeRegExp(term)})(?=[<\\s\n\t.,!?)])`, 'gi');
            
            let firstMatchReplaced = false;
            const tempContent = newContent.replace(searchRegex, (match) => {
                // Only replace the very first valid occurrence we find for this page.
                if (firstMatchReplaced) {
                    return match; 
                }
                
                const newPlaceholder = `[INTERNAL_LINK slug="${page.slug}" text="${match}"]`;
                console.log(`[Link Guardian] Injecting link for "${page.slug}" using anchor: "${match}"`);
                
                firstMatchReplaced = true;
                linkPlaced = true;
                return newPlaceholder;
            });
            
            if (linkPlaced) {
                newContent = tempContent;
                newlyLinkedSlugs.add(page.slug);
                deficit--;
            }
        }
    }
    
    if (deficit > 0) {
        console.warn(`[Link Guardian] Could not meet the full link quota. ${deficit} links still missing.`);
    }

    return newContent;
};


/**
 * Processes custom internal link placeholders in generated content and replaces them
 * with valid, full URL links based on a list of available pages.
 * @param content The HTML content string containing placeholders.
 * @param availablePages An array of page objects, each with 'id' (full URL) and 'slug'.
 * @returns The HTML content with placeholders replaced by valid <a> tags.
 */
const processInternalLinks = (content: string, availablePages: any[]): string => {
    if (!content || !availablePages || availablePages.length === 0) {
        return content;
    }

    // Create a map for efficient slug-to-page lookups.
    const pagesBySlug = new Map(availablePages.filter(p => p.slug).map(p => [p.slug, p]));

    // Regex to find placeholders like [INTERNAL_LINK slug="some-slug" text="some anchor text"]
    const placeholderRegex = /\[INTERNAL_LINK\s+slug="([^"]+)"\s+text="([^"]+)"\]/g;

    return content.replace(placeholderRegex, (match, slug, text) => {
        const page = pagesBySlug.get(slug);
        if (page && page.id) {
            // Found a valid page, create the link with the full URL.
            console.log(`[Link Processor] Found match for slug "${slug}". Replacing with link to ${page.id}`);
            // Escape quotes in text just in case AI includes them
            const sanitizedText = text.replace(/"/g, '&quot;');
            return `<a href="${page.id}">${sanitizedText}</a>`;
        } else {
            // This should rarely happen now with the new validation/repair/enforcement steps.
            console.warn(`[Link Processor] Could not find a matching page for slug "${slug}". This is unexpected. Replacing with plain text.`);
            return text; // Fallback: just return the anchor text.
        }
    });
};

// --- END: Core Utility Functions ---


// --- TYPE DEFINITIONS ---
type SitemapPage = {
    id: string;
    title: string;
    slug: string;
    lastMod: string | null;
    wordCount: number | null;
    crawledContent: string | null;
    healthScore: number | null;
    updatePriority: string | null;
    justification: string | null;
    daysOld: number | null;
    isStale: boolean;
    publishedState: string;
};

type GeneratedContent = {
    title: string;
    slug: string;
    metaDescription: string;
    primaryKeyword: string;
    semanticKeywords: string[];
    content: string;
    imageDetails: {
        prompt: string;
        altText: string;
        title: string;
        placeholder: string;
        generatedImageSrc?: string;
    }[];
    strategy: {
        targetAudience: string;
        searchIntent: string;
        competitorAnalysis: string;
        contentAngle: string;
    };
    jsonLdSchema: object;
    socialMediaCopy: {
        twitter: string;
        linkedIn: string;
    };
};

/**
 * Custom error for when generated content fails a quality gate,
 * but we still want to preserve the content for manual review.
 */
class ContentTooShortError extends Error {
  public content: string;
  public wordCount: number;

  constructor(message: string, content: string, wordCount: number) {
    super(message);
    this.name = 'ContentTooShortError';
    this.content = content;
    this.wordCount = wordCount;
  }
}

/**
 * "Zero-Tolerance Video Guardian": Scans generated content for duplicate YouTube embeds
 * and programmatically replaces the second instance with the correct, unique video.
 * This provides a crucial fallback for when the AI fails to follow instructions.
 * @param content The HTML content string with potential duplicate video iframes.
 * @param youtubeVideos The array of unique video objects that *should* have been used.
 * @returns The HTML content with duplicate videos corrected.
 */
const enforceUniqueVideoEmbeds = (content: string, youtubeVideos: any[]): string => {
    if (!youtubeVideos || youtubeVideos.length < 2) {
        return content; // Not enough videos to have a duplicate issue.
    }

    const iframeRegex = /<iframe[^>]+src="https:\/\/www\.youtube\.com\/embed\/([^"?&]+)[^>]*><\/iframe>/g;
    const matches = [...content.matchAll(iframeRegex)];
    
    if (matches.length < 2) {
        return content; // Not enough embeds to have duplicates.
    }

    const videoIdsInContent = matches.map(m => m[1]);
    const firstVideoId = videoIdsInContent[0];
    const isDuplicate = videoIdsInContent.every(id => id === firstVideoId);


    if (isDuplicate) {
        const duplicateId = videoIdsInContent[0];
        console.warn(`[Video Guardian] Duplicate video ID "${duplicateId}" detected. Attempting to replace second instance.`);

        const secondVideo = youtubeVideos[1];
        if (secondVideo && secondVideo.videoId && secondVideo.videoId !== duplicateId) {
            const secondMatch = matches[1]; // The second iframe tag found
            // Find the start index of the second match to ensure we don't replace the first one
            const secondMatchIndex = content.indexOf(secondMatch[0], secondMatch.index);

            if (secondMatchIndex !== -1) {
                // Construct the replacement iframe tag by replacing just the ID
                const correctedIframe = secondMatch[0].replace(duplicateId, secondVideo.videoId);
                content = content.substring(0, secondMatchIndex) + correctedIframe + content.substring(secondMatchIndex + secondMatch[0].length);
                console.log(`[Video Guardian] Successfully replaced second duplicate with unique video: "${secondVideo.videoId}".`);
            }
        }
    }
    return content;
};


/**
 * Validates and normalizes the JSON object returned by the AI to ensure it
 * has all the required fields, preventing crashes from schema deviations.
 * @param parsedJson The raw parsed JSON from the AI.
 * @param itemTitle The original title of the content item, used for fallbacks.
 * @returns A new object with all required fields guaranteed to exist.
 */
const normalizeGeneratedContent = (parsedJson: any, itemTitle: string): GeneratedContent => {
    const normalized = { ...parsedJson };

    // --- Critical Fields ---
    if (!normalized.title) normalized.title = itemTitle;
    if (!normalized.slug) normalized.slug = itemTitle.toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]+/g, '');
    if (!normalized.content) {
        console.warn(`[Normalization] 'content' field was missing for "${itemTitle}". Defaulting to empty string.`);
        normalized.content = '';
    }

    // --- Image Details: The main source of errors ---
    if (!normalized.imageDetails || !Array.isArray(normalized.imageDetails) || normalized.imageDetails.length === 0) {
        console.warn(`[Normalization] 'imageDetails' was missing or invalid for "${itemTitle}". Generating default image prompts.`);
        const slugBase = normalized.slug || itemTitle.toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]+/g, '');
        normalized.imageDetails = [
            {
                prompt: `A high-quality, photorealistic image representing the concept of: "${normalized.title}". Cinematic, professional blog post header image, 16:9 aspect ratio.`,
                altText: `A conceptual image for "${normalized.title}"`,
                title: `${slugBase}-feature-image`,
                placeholder: '[IMAGE_1_PLACEHOLDER]'
            },
            {
                prompt: `An infographic or diagram illustrating a key point from the article: "${normalized.title}". Clean, modern design with clear labels. 16:9 aspect ratio.`,
                altText: `Infographic explaining a key concept from "${normalized.title}"`,
                title: `${slugBase}-infographic`,
                placeholder: '[IMAGE_2_PLACEHOLDER]'
            }
        ];
        
        // Ensure placeholders are injected if missing from content
        if (normalized.content && !normalized.content.includes('[IMAGE_1_PLACEHOLDER]')) {
            const paragraphs = normalized.content.split('</p>');
            if (paragraphs.length > 2) {
                paragraphs.splice(2, 0, '<p>[IMAGE_1_PLACEHOLDER]</p>');
                normalized.content = paragraphs.join('</p>');
            } else {
                normalized.content += '<p>[IMAGE_1_PLACEHOLDER]</p>';
            }
        }
        if (normalized.content && !normalized.content.includes('[IMAGE_2_PLACEHOLDER]')) {
            const paragraphs = normalized.content.split('</p>');
            if (paragraphs.length > 5) {
                paragraphs.splice(5, 0, '<p>[IMAGE_2_PLACEHOLDER]</p>');
                 normalized.content = paragraphs.join('</p>');
            } else {
                 normalized.content += '<p>[IMAGE_2_PLACEHOLDER]</p>';
            }
        }
    }

    // --- Other required fields for UI stability ---
    if (!normalized.metaDescription) normalized.metaDescription = `Read this comprehensive guide on ${normalized.title}.`;
    if (!normalized.primaryKeyword) normalized.primaryKeyword = itemTitle;
    if (!normalized.semanticKeywords || !Array.isArray(normalized.semanticKeywords)) normalized.semanticKeywords = [];
    if (!normalized.strategy) normalized.strategy = { targetAudience: '', searchIntent: '', competitorAnalysis: '', contentAngle: '' };
    if (!normalized.jsonLdSchema) normalized.jsonLdSchema = {};
    if (!normalized.socialMediaCopy) normalized.socialMediaCopy = { twitter: '', linkedIn: '' };

    return normalized as GeneratedContent;
};

const PROMPT_TEMPLATES = {
    cluster_planner: {
        systemInstruction: `You are a master SEO strategist specializing in building topical authority through pillar-and-cluster content models. Your task is to analyze a user's broad topic and generate a complete, SEO-optimized content plan that addresses user intent at every stage.

**RULES:**
1.  **Output Format:** Your entire response MUST be a single, valid JSON object. Do not include any text before or after the JSON.
2.  **Pillar Content:** The 'pillarTitle' must be a broad, comprehensive title for a definitive guide. It must be engaging, keyword-rich, and promise immense value to the reader. Think "The Ultimate Guide to..." or "Everything You Need to Know About...".
3.  **Cluster Content:** The 'clusterTitles' must be an array of 5 to 7 unique strings. Each title should be a compelling question or a long-tail keyword phrase that a real person would search for. These should be distinct sub-topics that logically support and link back to the main pillar page.
    - Good Example: "How Much Does Professional Landscaping Cost in 2025?"
    - Bad Example: "Landscaping Costs"
4.  **Keyword Focus:** All titles must be optimized for search engines without sounding robotic.
{{GEO_TARGET_INSTRUCTIONS}}
5.  **JSON Structure:** The JSON object must conform to this exact structure:
    {
      "pillarTitle": "A comprehensive, SEO-optimized title for the main pillar article.",
      "clusterTitles": [
        "A specific, long-tail keyword-focused title for the first cluster article.",
        "A specific, long-tail keyword-focused title for the second cluster article.",
        "..."
      ]
    }

**FINAL INSTRUCTION:** Your ENTIRE response MUST be ONLY the JSON object, starting with { and ending with }. Do not add any introductory text, closing remarks, or markdown code fences. Your output will be parsed directly by a machine.`,
        userPrompt: (topic: string) => `Generate a pillar-and-cluster content plan for the topic: "${topic}".`
    },
    content_meta_and_outline: {
        systemInstruction: `You are an ELITE content strategist and SEO expert. Your task is to generate ALL metadata and a comprehensive structural plan for a world-class article. You will create the title, meta description, image prompts, key takeaways, an outline of H2 sections, FAQ questions, an introduction, and a conclusion.

**RULES:**
1.  **JSON OUTPUT ONLY:** Your ENTIRE response MUST be a single, valid JSON object. No text before or after.
2.  **DO NOT WRITE THE ARTICLE BODY:** Your role is to plan, not write the main content. The 'outline' should be a list of H2 headings ONLY. The 'introduction' and 'conclusion' sections should be fully written paragraphs.
3.  **WRITING STYLE (For Intro/Conclusion):** Follow the "ANTI-AI" protocol: Short, direct sentences (avg. 10 words). Tiny paragraphs (2-3 sentences max). Active voice. No forbidden phrases (e.g., 'delve into', 'in today's digital landscape').
4.  **STRUCTURAL REQUIREMENTS:**
    - **keyTakeaways**: Exactly 8 high-impact bullet points (as an array of strings).
    - **outline**: 10-15 H2 headings (as an array of strings). Use the provided semantic keywords.
    - **faqSection**: Exactly 8 questions (as an array of objects: \`[{ "question": "..." }]\`).
    - **imageDetails**: Exactly 2 image prompts. Placeholders MUST be '[IMAGE_1_PLACEHOLDER]' and '[IMAGE_2_PLACEHOLDER]'.
5.  **JSON STRUCTURE:** Adhere strictly to the provided JSON schema. Ensure all fields are present.
`,
        userPrompt: (primaryKeyword: string, semanticKeywords: string[] | null, serpData: any[] | null, existingPages: any[] | null, originalContent: string | null = null) => {
            const MAX_CONTENT_CHARS = 8000;
            const MAX_LINKING_PAGES = 50;
            const MAX_SERP_SNIPPET_LENGTH = 200;

            let contentForPrompt = originalContent 
                ? `***CRITICAL REWRITE MANDATE:*** You are to deconstruct the following outdated article and rebuild its plan.
<original_content_to_rewrite>
${originalContent.substring(0, MAX_CONTENT_CHARS)}
</original_content_to_rewrite>`
                : '';

            return `
**PRIMARY KEYWORD:** "${primaryKeyword}"
${contentForPrompt}
${semanticKeywords ? `**MANDATORY SEMANTIC KEYWORDS:** You MUST integrate these into the outline headings: <semantic_keywords>${JSON.stringify(semanticKeywords)}</semantic_keywords>` : ''}
${serpData ? `**SERP COMPETITOR DATA:** Analyze for gaps. <serp_data>${JSON.stringify(serpData.map(d => ({title: d.title, link: d.link, snippet: d.snippet?.substring(0, MAX_SERP_SNIPPET_LENGTH)})))}</serp_data>` : ''}
${existingPages && existingPages.length > 0 ? `**INTERNAL LINKING TARGETS (for context):** <existing_articles_for_linking>${JSON.stringify(existingPages.slice(0, MAX_LINKING_PAGES).map(p => ({slug: p.slug, title: p.title})).filter(p => p.slug && p.title))}</existing_articles_for_linking>` : ''}

Generate the complete JSON plan.
`;
        }
    },
    write_article_section: {
        systemInstruction: `You are an ELITE content writer, writing in the style of a world-class thought leader like Alex Hormozi. Your SOLE task is to write the content for a single section of a larger article, based on the provided heading.

**RULES:**
1.  **RAW HTML OUTPUT:** Your response must be ONLY the raw HTML content for the section. NO JSON, NO MARKDOWN, NO EXPLANATIONS. Start directly with a \`<p>\` tag. Do not include the \`<h2>\` tag for the main heading; it will be added automatically.
2.  **WORD COUNT:** The section MUST be between 250 and 300 words. This is mandatory.
3.  **ELITE WRITING STYLE (THE "ANTI-AI" PROTOCOL):**
    - Short, direct sentences. Average 10 words. Max 15.
    - Tiny paragraphs. 2-3 sentences. MAXIMUM.
    - Use contractions: "it's," "you'll," "can't."
    - Active voice. Simple language. No filler words.
    - Ask the reader direct questions. Use analogies.
4.  **FORBIDDEN PHRASES (ZERO TOLERANCE):**
    - ❌ 'delve into', 'in today's digital landscape', 'revolutionize', 'game-changer', 'unlock', 'leverage', 'in conclusion', 'to summarize', 'utilize', 'furthermore', 'moreover', 'landscape', 'realm', 'dive deep', etc.
5.  **STRUCTURE:**
    - You MAY use \`<h3>\` tags for sub-headings.
    - You MUST include at least one HTML table (\`<table>\`), list (\`<ul>\`/\`<ol>\`), or blockquote (\`<blockquote>\`) if relevant to the topic.
    - You MUST naturally integrate 1-2 internal link placeholders where contextually appropriate: \`[INTERNAL_LINK slug="example-slug" text="anchor text"]\`.
`,
        userPrompt: (primaryKeyword: string, articleTitle: string, sectionHeading: string, existingPages: any[] | null) => `
**Primary Keyword:** "${primaryKeyword}"
**Main Article Title:** "${articleTitle}"
**Section to Write:** "${sectionHeading}"

${existingPages && existingPages.length > 0 ? `**Available Internal Links:** You can link to these pages.
<pages>${JSON.stringify(existingPages.slice(0, 50).map(p => ({slug: p.slug, title: p.title})))}</pages>` : ''}

Write the HTML content for this section now.
`
    },
    write_faq_answer: {
        systemInstruction: `You are an expert content writer. Your task is to provide a clear, concise, and helpful answer to a single FAQ question.

**RULES:**
1.  **RAW HTML PARAGRAPH:** Respond with ONLY the answer wrapped in a single \`<p>\` tag. Do not repeat the question.
2.  **STYLE:** The answer should be direct, easy to understand, and typically 2-4 sentences long, following the "ANTI-AI" writing style (simple words, active voice).
`,
        userPrompt: (question: string) => `Question: "${question}"`
    },
    semantic_keyword_generator: {
        systemInstruction: `You are a world-class SEO analyst. Your task is to generate a comprehensive list of semantic and LSI (Latent Semantic Indexing) keywords related to a primary topic. These keywords should cover sub-topics, user intent variations, and related entities.

**RULES:**
1.  **Output Format:** Your entire response MUST be a single, valid JSON object. Do not include any text, markdown, or justification before or after the JSON.
2.  **Quantity:** Generate between 15 and 25 keywords.
3.  **JSON Structure:** The JSON object must conform to this exact structure:
    {
      "semanticKeywords": [
        "A highly relevant LSI keyword.",
        "A long-tail question-based keyword.",
        "Another related keyword or phrase.",
        "..."
      ]
    }

**FINAL INSTRUCTION:** Your ENTIRE response MUST be ONLY the JSON object, starting with { and ending with }. Do not add any introductory text, closing remarks, or markdown code fences. Your output will be parsed directly by a machine.`,
        userPrompt: (primaryKeyword: string) => `Generate semantic keywords for the primary topic: "${primaryKeyword}".`
    },
    content_health_analyzer: {
        systemInstruction: `You are an expert SEO content auditor. Your task is to analyze the provided text from a blog post and assign it a "Health Score". A low score indicates the content is thin, outdated, poorly structured, or not helpful, signaling an urgent need for an update.

**Evaluation Criteria:**
*   **Content Depth & Helpfulness (40%):** How thorough is the content? Does it seem to satisfy user intent? Is it just surface-level, or does it provide real value?
*   **Readability & Structure (30%):** Is it well-structured with clear headings? Are paragraphs short and scannable? Is the language complex or easy to read?
*   **Engagement Potential (15%):** Does it use lists, bullet points, or other elements that keep a reader engaged?
*   **Freshness Signals (15%):** Does the content feel current, or does it reference outdated concepts, statistics, or years?

**RULES:**
1.  **Output Format:** Your entire response MUST be a single, valid JSON object. Do not include any text, markdown, or justification before or after the JSON.
2.  **Health Score:** The 'healthScore' must be an integer between 0 and 100.
3.  **Update Priority:** The 'updatePriority' must be one of: "Critical" (score 0-25), "High" (score 26-50), "Medium" (score 51-75), or "Healthy" (score 76-100).
4.  **Justification:** Provide a concise, one-sentence explanation for your scoring in the 'justification' field.
5.  **JSON Structure:**
    {
      "healthScore": 42,
      "updatePriority": "High",
      "justification": "The content covers the topic superficially and lacks clear structure, making it difficult to read."
    }

**FINAL INSTRUCTION:** Your ENTIRE response MUST be ONLY the JSON object, starting with { and ending with }. Do not add any introductory text, closing remarks, or markdown code fences. Your output will be parsed directly by a machine.`,
        userPrompt: (content: string) => `Analyze the following blog post content and provide its SEO health score.\n\n&lt;content&gt;\n${content}\n&lt;/content&gt;`
    }
};

type ContentItem = {
    id: string;
    title: string;
    type: 'pillar' | 'cluster' | 'standard';
    status: 'idle' | 'generating' | 'done' | 'error';
    statusText: string;
    generatedContent: GeneratedContent | null;
    crawledContent: string | null;
    originalUrl?: string;
};

type SeoCheck = {
    valid: boolean;
    text: string;
    count?: number;
};


// --- REDUCER for items state ---
type ItemsAction =
    | { type: 'SET_ITEMS'; payload: Partial<ContentItem>[] }
    | { type: 'UPDATE_STATUS'; payload: { id: string; status: ContentItem['status']; statusText: string } }
    | { type: 'SET_CONTENT'; payload: { id: string; content: GeneratedContent } }
    | { type: 'SET_CRAWLED_CONTENT'; payload: { id: string; content: string } };

const itemsReducer = (state: ContentItem[], action: ItemsAction): ContentItem[] => {
    switch (action.type) {
        case 'SET_ITEMS':
            return action.payload.map((item: any) => ({ ...item, status: 'idle', statusText: 'Not Started', generatedContent: null, crawledContent: item.crawledContent || null }));
        case 'UPDATE_STATUS':
            return state.map(item =>
                item.id === action.payload.id
                    ? { ...item, status: action.payload.status, statusText: action.payload.statusText }
                    : item
            );
        case 'SET_CONTENT':
            return state.map(item =>
                item.id === action.payload.id
                    ? { ...item, status: 'done', statusText: 'Completed', generatedContent: action.payload.content }
                    : item
            );
        case 'SET_CRAWLED_CONTENT':
             return state.map(item =>
                item.id === action.payload.id
                    ? { ...item, crawledContent: action.payload.content }
                    : item
            );
        default:
            return state;
    }
};

// --- Child Components ---

const ProgressBar = memo(({ currentStep, onStepClick }: { currentStep: number; onStepClick: (step: number) => void; }) => {
    const steps = ["Setup", "Content Strategy", "Review & Export"];
    return (
        <nav aria-label="Main navigation">
            <ol className="progress-bar">
                {steps.map((name, index) => {
                    const stepIndex = index + 1;
                    const isClickable = true; // All steps are clickable
                    const status = stepIndex < currentStep ? 'completed' : stepIndex === currentStep ? 'active' : 'upcoming';
                    return (
                        <li 
                            key={index} 
                            className={`progress-step ${status} ${isClickable ? 'clickable' : ''}`} 
                            aria-current={status === 'active'}
                            onClick={() => isClickable && onStepClick(stepIndex)}
                            role="button"
                            tabIndex={0}
                        >
                            <div className="step-circle">{status === 'completed' ? '✓' : stepIndex}</div>
                            <span className="step-name">{name}</span>
                        </li>
                    );
                })}
            </ol>
        </nav>
    );
});


interface ApiKeyInputProps {
    provider: string;
    value: string;
    onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => void;
    status: 'idle' | 'validating' | 'valid' | 'invalid';
    name?: string;
    placeholder?: string;
    isTextArea?: boolean;
    isEditing: boolean;
    onEdit: () => void;
    type?: 'text' | 'password';
}
const ApiKeyInput = memo(({ provider, value, onChange, status, name, placeholder, isTextArea, isEditing, onEdit, type = 'password' }: ApiKeyInputProps) => {
    const InputComponent = isTextArea ? 'textarea' : 'input';

    if (status === 'valid' && !isEditing) {
        return (
            <div className="api-key-group">
                <input type="text" readOnly value={`**** **** **** ${value.slice(-4)}`} />
                <button onClick={onEdit} className="btn-edit-key" aria-label={`Edit ${provider} API Key`}>Edit</button>
            </div>
        );
    }

    const commonProps = {
        name: name || `${provider}ApiKey`,
        value: value,
        onChange: onChange,
        placeholder: placeholder || `Enter your ${provider.charAt(0).toUpperCase() + provider.slice(1)} API Key`,
        'aria-invalid': status === 'invalid',
        'aria-describedby': `${provider}-status`,
        ...(isTextArea ? { rows: 4 } : { type: type })
    };

    return (
        <div className="api-key-group">
            <InputComponent {...commonProps} />
            <div className="key-status-icon" id={`${provider}-status`} role="status">
                {status === 'validating' && <div className="key-status-spinner" aria-label="Validating key"></div>}
                {status === 'valid' && <svg className="success" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-label="Key is valid"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" /></svg>}
                {status === 'invalid' && <svg className="error" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-label="Key is invalid"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" /></svg>}
            </div>
        </div>
    );
});

const SeoChecklist = memo(({ checks }: { checks: Record<string, SeoCheck> }) => (
    <ul className="guardian-checklist">
        {Object.entries(checks).map(([key, check]) => (
            <li key={key} className={check.valid ? 'valid' : 'invalid'}>
                {check.valid ? (
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-label="Success"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" /></svg>
                ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-label="Error"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" /></svg>
                )}
                <span>{check.text}</span>
            </li>
        ))}
    </ul>
));

// --- START: Advanced Content Quality Analysis ---
const countSyllables = (word: string): number => {
    if (!word) return 0;
    word = word.toLowerCase().trim();
    if (word.length <= 3) { return 1; }
    word = word.replace(/(?:[^laeiouy]es|ed|[^laeiouy]e)$/, '');
    word = word.replace(/^y/, '');
    const matches = word.match(/[aeiouy]{1,2}/g);
    return matches ? matches.length : 0;
};

const calculateFleschReadability = (text: string): number => {
    const sentences = (text.match(/[.!?]+/g) || []).length || 1;
    const words = text.split(/\s+/).filter(Boolean).length;
    if (words < 100) return 0; // Not enough content for an accurate score

    let syllableCount = 0;
    text.split(/\s+/).forEach(word => {
        syllableCount += countSyllables(word);
    });

    const fleschScore = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllableCount / words);
    return Math.round(Math.min(100, Math.max(0, fleschScore)));
};
// --- END: Advanced Content Quality Analysis ---

interface RankGuardianProps {
    seoData: {
        title: string;
        metaDescription: string;
        slug: string;
        primaryKeyword: string;
        content: string;
    };
}
const RankGuardian = memo(({ seoData }: RankGuardianProps) => {
    const checks = useMemo(() => {
        const { title, metaDescription, primaryKeyword, content } = seoData;
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = content || '';
        const textContent = tempDiv.textContent || '';
        const wordCount = textContent.split(/\s+/).filter(Boolean).length;
        const h1s = tempDiv.getElementsByTagName('h1').length;
        const h2s = tempDiv.getElementsByTagName('h2').length;

        const titleChecks: Record<string, SeoCheck> = {
            length: { valid: title.length > 0 && title.length <= 60, text: `Title is ${title.length} chars (1-60 ideal)` },
            keyword: { valid: title.toLowerCase().includes(primaryKeyword.toLowerCase()), text: 'Title contains keyword' }
        };

        const metaChecks: Record<string, SeoCheck> = {
            length: { valid: metaDescription.length > 0 && metaDescription.length <= 155, text: `Meta is ${metaDescription.length} chars (1-155 ideal)` },
            keyword: { valid: metaDescription.toLowerCase().includes(primaryKeyword.toLowerCase()), text: 'Meta contains keyword' }
        };
        
        const contentChecks: Record<string, SeoCheck> = {
            wordCount: { valid: wordCount >= 300, text: `${wordCount} words (300+ recommended)` },
            keywordDensity: {
                count: (textContent.toLowerCase().match(new RegExp(primaryKeyword.toLowerCase(), 'g')) || []).length,
                get valid() { return this.count! > 0; },
                get text() { return `Keyword used ${this.count} time(s)`; }
            },
            h1: { valid: h1s === 0, text: `${h1s} H1 tags (0 is required in content)` },
            h2: { valid: h2s >= 2, text: `${h2s} H2 tags (2+ recommended)` },
            lists: {
                count: tempDiv.getElementsByTagName('ul').length + tempDiv.getElementsByTagName('ol').length,
                get valid() { return this.count! >= 2; },
                get text() { return `${this.count} lists (<ul>/<ol>) found (2+ recommended)`; }
            },
            questions: {
                count: (textContent.match(/\?/g) || []).length,
                get valid() { return this.count! >= 3; },
                get text() { return `${this.count} questions (?) found (3+ recommended)`; }
            }
        };
        
        const readabilityScore = calculateFleschReadability(textContent);
        const allChecks = { ...titleChecks, ...metaChecks, ...contentChecks };
        const seoScore = Object.values(allChecks).filter(c => c.valid).length;
        const totalChecks = Object.keys(allChecks).length;
        const seoScorePercent = Math.round((seoScore / totalChecks) * 100);

        return { titleChecks, metaChecks, contentChecks, readabilityScore, seoScorePercent };
    }, [seoData]);

    const ScoreCircle = ({ score, label }: { score: number; label: string }) => {
        const radius = 40;
        const circumference = 2 * Math.PI * radius;
        const offset = circumference - (score / 100) * circumference;
        let strokeColor = 'var(--success-color)';
        if (score < 75) strokeColor = 'var(--warning-text-color)';
        if (score < 50) strokeColor = 'var(--error-color)';

        return (
            <div className="score-circle-container" role="meter" aria-valuenow={score} aria-valuemin={0} aria-valuemax={100} aria-label={`${label} score is ${score} out of 100`}>
                <svg className="score-circle" viewBox="0 0 100 100">
                    <circle className="circle-bg" cx="50" cy="50" r={radius}></circle>
                    <circle
                        className="circle"
                        cx="50"
                        cy="50"
                        r={radius}
                        stroke={strokeColor}
                        strokeDasharray={circumference}
                        strokeDashoffset={offset}
                    ></circle>
                </svg>
                <span className="score-text" style={{ color: strokeColor }}>{score}</span>
            </div>
        );
    };

    return (
        <div className="rank-guardian-pane">
            <div className="score-display">
                <div className="score-card">
                    <h4>SEO Score</h4>
                    <ScoreCircle score={checks.seoScorePercent} label="SEO"/>
                </div>
                <div className="score-card">
                    <h4>Readability</h4>
                    <ScoreCircle score={checks.readabilityScore} label="Readability" />
                </div>
            </div>
            <div className="checklists-container">
                <div className="checklist-column">
                    <h5>Title & Meta</h5>
                    <SeoChecklist checks={{ ...checks.titleChecks, ...checks.metaChecks }} />
                </div>
                <div className="checklist-column">
                    <h5>Content</h5>
                    <SeoChecklist checks={checks.contentChecks} />
                </div>
            </div>
        </div>
    );
});

// Helper function to escape characters for use in a regular expression
const escapeRegExp = (string: string) => {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

interface ReviewModalProps {
    item: ContentItem;
    onClose: () => void;
    onSaveChanges: (itemId: string, updatedSeo: { title: string; metaDescription: string; slug: string }, updatedContent: string) => void;
    wpConfig: { url: string, username: string };
    wpPassword: string;
    onPublishSuccess: (originalUrl: string) => void;
    publishItem: (itemToPublish: ContentItem, currentWpPassword: string) => Promise<{ success: boolean; message: React.ReactNode; link?: string }>;
}

const ReviewModal = ({ item, onClose, onSaveChanges, wpConfig, wpPassword, onPublishSuccess, publishItem }: ReviewModalProps) => {
    if (!item || !item.generatedContent) return null;

    const [activeTab, setActiveTab] = useState('Live Preview');
    const [activeSeoTab, setActiveSeoTab] = useState('serp');
    const [editedSeo, setEditedSeo] = useState({ title: '', metaDescription: '', slug: '' });
    const [editedContent, setEditedContent] = useState('');
    const [copyStatus, setCopyStatus] = useState('Copy HTML');
    const [wpPublishStatus, setWpPublishStatus] = useState('idle'); // idle, publishing, success, error
    const [wpPublishMessage, setWpPublishMessage] = useState<React.ReactNode>('');

    // SOTA Editor State
    const editorRef = useRef<HTMLTextAreaElement>(null);
    const lineNumbersRef = useRef<HTMLPreElement>(null);
    const [lineCount, setLineCount] = useState(1);

    useEffect(() => {
        if (item && item.generatedContent) {
            const isUpdate = !!item.originalUrl;
            // CRITICAL FIX: If this is an update, extract and enforce the original slug.
            const originalSlug = isUpdate ? extractSlugFromUrl(item.originalUrl!) : item.generatedContent.slug;

            setEditedSeo({
                title: item.generatedContent.title,
                metaDescription: item.generatedContent.metaDescription,
                slug: originalSlug,
            });
            setEditedContent(item.generatedContent.content);
            setActiveTab('Live Preview'); // Reset tab on new item
            setWpPublishStatus('idle'); // Reset publish status
            setWpPublishMessage('');
        }
    }, [item]);

    // SOTA Editor Logic
    useEffect(() => {
        const lines = editedContent.split('\n').length;
        setLineCount(lines || 1);
    }, [editedContent]);

    const handleEditorScroll = useCallback(() => {
        if (lineNumbersRef.current && editorRef.current) {
            lineNumbersRef.current.scrollTop = editorRef.current.scrollTop;
        }
    }, []);


    const previewContent = useMemo(() => {
        // The editedContent now contains the base64 images directly, so no replacement is needed for preview.
        return editedContent;
    }, [editedContent]);

    const handleSeoChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        const { name, value } = e.target;
        setEditedSeo(prev => ({ ...prev, [name]: value }));
    };

    const handleSlugChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = e.target.value
            .toLowerCase()
            .replace(/\s+/g, '-') // Replace spaces with -
            .replace(/[^\w-]+/g, ''); // Remove all non-word chars
        setEditedSeo(prev => ({ ...prev, slug: value }));
    };

    const handleCopyHtml = () => {
        if (!item?.generatedContent) return;
        navigator.clipboard.writeText(editedContent)
            .then(() => {
                setCopyStatus('Copied!');
                setTimeout(() => setCopyStatus('Copy HTML'), 2000);
            })
            .catch(err => console.error('Failed to copy HTML: ', err));
    };

    const handleDownloadImage = (base64Data: string, fileName: string) => {
        const link = document.createElement('a');
        link.href = base64Data;
        const safeName = fileName.replace(/[^a-z0-9]/gi, '_').toLowerCase();
        link.download = `${safeName}.jpg`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const handlePublishToWordPress = async () => {
        if (!wpConfig.url || !wpConfig.username || !wpPassword) {
            setWpPublishStatus('error');
            setWpPublishMessage('Please fill in WordPress URL, Username, and Application Password in Step 2.');
            return;
        }

        setWpPublishStatus('publishing');
        
        // Create a temporary item with the latest edits for the publish function
        const itemWithEdits: ContentItem = {
            ...item,
            generatedContent: {
                ...item.generatedContent!,
                ...editedSeo,
                content: editedContent,
            }
        };

        const result = await publishItem(itemWithEdits, wpPassword);

        if (result.success) {
            setWpPublishStatus('success');
            if (item.originalUrl) {
                onPublishSuccess(item.originalUrl);
            }
        } else {
            setWpPublishStatus('error');
        }
        setWpPublishMessage(result.message);
    };

    const TABS = ['Live Preview', 'Editor', 'Assets', 'SEO & Meta', 'Raw JSON'];
    const { title, metaDescription, slug } = editedSeo;
    const { primaryKeyword } = item.generatedContent;

    const titleLength = title.length;
    const titleStatus = titleLength > 60 ? 'bad' : titleLength > 50 ? 'warn' : 'good';
    const metaLength = metaDescription.length;
    const metaStatus = metaLength > 155 ? 'bad' : metaLength > 120 ? 'warn' : 'good';

    const isUpdate = !!item.originalUrl;
    const publishButtonText = isUpdate ? 'Update Live Post' : 'Publish to WordPress';
    const publishingButtonText = isUpdate ? 'Updating...' : 'Publishing...';

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()} role="dialog" aria-modal="true" aria-labelledby="review-modal-title">
                <h2 id="review-modal-title" className="sr-only">Review and Edit Content</h2>
                <button className="modal-close-btn" onClick={onClose} aria-label="Close modal">&times;</button>
                <div className="review-tabs" role="tablist">
                    {TABS.map(tab => (
                        <button key={tab} className={`tab-btn ${activeTab === tab ? 'active' : ''}`} onClick={() => setActiveTab(tab)} role="tab" aria-selected={activeTab === tab} aria-controls={`tab-panel-${tab.replace(/\s/g, '-')}`}>
                            {tab}
                        </button>
                    ))}
                </div>

                <div className="tab-content">
                    {activeTab === 'Live Preview' && (
                        <div id="tab-panel-Live-Preview" role="tabpanel" className="live-preview" dangerouslySetInnerHTML={{ __html: previewContent }}></div>
                    )}
                    
                    {activeTab === 'Editor' && (
                        <div id="tab-panel-Editor" role="tabpanel" className="editor-tab-container">
                            <div className="sota-editor">
                                <pre className="line-numbers" ref={lineNumbersRef} aria-hidden="true">
                                    {Array.from({ length: lineCount }, (_, i) => i + 1).join('\n')}
                                </pre>
                                <textarea
                                    ref={editorRef}
                                    className="html-editor"
                                    value={editedContent}
                                    onChange={(e) => setEditedContent(e.target.value)}
                                    onScroll={handleEditorScroll}
                                    aria-label="HTML Content Editor"
                                    spellCheck="false"
                                />
                            </div>
                        </div>
                    )}

                    {activeTab === 'Assets' && (
                        <div id="tab-panel-Assets" role="tabpanel" className="assets-tab-container">
                            <h3>Generated Images</h3>
                            <p className="help-text" style={{fontSize: '1rem', maxWidth: '800px', margin: '0 0 2rem 0'}}>These images are embedded in your article. They will be automatically uploaded to your WordPress media library when you publish. You can also download them for manual use.</p>
                            <div className="image-assets-grid">
                                {item.generatedContent.imageDetails.map((image, index) => (
                                    image.generatedImageSrc ? (
                                        <div key={index} className="image-asset-card">
                                            <img src={image.generatedImageSrc} alt={image.altText} />
                                            <div className="image-asset-details">
                                                <p><strong>Alt Text:</strong> {image.altText}</p>
                                                <button className="btn btn-small" onClick={() => handleDownloadImage(image.generatedImageSrc!, image.title)}>Download Image</button>
                                            </div>
                                        </div>
                                    ) : null
                                ))}
                            </div>
                        </div>
                    )}

                    {activeTab === 'SEO & Meta' && (
                         <div id="tab-panel-SEO-&-Meta" role="tabpanel" className="seo-meta-container">
                             <div className="seo-meta-tabs" role="tablist" aria-label="SEO & Meta sections">
                                <button className={`seo-meta-tab-btn ${activeSeoTab === 'serp' ? 'active' : ''}`} onClick={() => setActiveSeoTab('serp')} role="tab" aria-selected={activeSeoTab === 'serp'}>
                                    SERP Preview
                                </button>
                                <button className={`seo-meta-tab-btn ${activeSeoTab === 'guardian' ? 'active' : ''}`} onClick={() => setActiveSeoTab('guardian')} role="tab" aria-selected={activeSeoTab === 'guardian'}>
                                    Rank Guardian
                                </button>
                            </div>

                            <div className="seo-meta-grid">
                                <div className="seo-inputs">
                                    <div className="form-group">
                                        <div className="label-wrapper">
                                            <label htmlFor="title">SEO Title</label>
                                            <span className={`char-counter ${titleStatus}`}>{titleLength} / 60</span>
                                        </div>
                                        <input type="text" id="title" name="title" value={title} onChange={handleSeoChange} />
                                        <div className="progress-bar-container">
                                          <div className={`progress-bar-fill ${titleStatus}`} style={{ width: `${Math.min(100, (titleLength / 60) * 100)}%` }}></div>
                                        </div>
                                    </div>
                                    <div className="form-group">
                                         <div className="label-wrapper">
                                            <label htmlFor="metaDescription">Meta Description</label>
                                            <span className={`char-counter ${metaStatus}`}>{metaLength} / 155</span>
                                        </div>
                                        <textarea id="metaDescription" name="metaDescription" className="meta-description-input" value={metaDescription} onChange={handleSeoChange}></textarea>
                                         <div className="progress-bar-container">
                                          <div className={`progress-bar-fill ${metaStatus}`} style={{ width: `${Math.min(100, (metaLength / 155) * 100)}%` }}></div>
                                        </div>
                                    </div>
                                    <div className="form-group">
                                        <label htmlFor="slug">URL Slug</label>
                                        <div className="slug-group">
                                            <span className="slug-base-url">/</span>
                                            <input
                                                type="text"
                                                id="slug"
                                                name="slug"
                                                value={slug}
                                                onChange={handleSlugChange}
                                                disabled={isUpdate}
                                                aria-describedby={isUpdate ? "slug-help" : undefined}
                                            />
                                        </div>
                                        {isUpdate && (
                                            <p id="slug-help" className="help-text" style={{color: 'var(--success-color)'}}>
                                                Original slug is preserved to prevent breaking existing URLs.
                                            </p>
                                        )}
                                    </div>
                                    <div className="form-group">
                                        <label>Primary Keyword</label>
                                        <input type="text" value={primaryKeyword} disabled />
                                    </div>
                                </div>
                                <div className="serp-preview-container">
                                    <h4>Google Preview</h4>
                                    <div className="serp-preview">
                                        <div className="serp-url">{wpConfig.url.replace(/^(https?:\/\/)?/, '').replace(/\/+$/, '')}/{slug}</div>
                                        <a href="#" className="serp-title" onClick={(e) => e.preventDefault()} tabIndex={-1}>{title}</a>
                                        <div className="serp-description">{metaDescription}</div>
                                    </div>
                                </div>
                                {activeSeoTab === 'guardian' && (
                                    <div className="rank-guardian-container">
                                        <RankGuardian seoData={{ ...editedSeo, primaryKeyword, content: editedContent }} />
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {activeTab === 'Raw JSON' && (
                        <pre id="tab-panel-Raw-JSON" role="tabpanel" className="json-viewer">
                            {JSON.stringify(item.generatedContent, null, 2)}
                        </pre>
                    )}
                </div>

                <div className="modal-footer">
                    <div className="wp-publish-container">
                        {wpPublishMessage && <div className={`publish-status ${wpPublishStatus}`} role="alert">{wpPublishMessage}</div>}
                    </div>

                    <div className="modal-actions">
                        <button className="btn btn-secondary" onClick={() => onSaveChanges(item.id, editedSeo, editedContent)}>Save Changes</button>
                        <button className="btn btn-secondary" onClick={handleCopyHtml}>{copyStatus}</button>
                        <button 
                            className="btn btn-success"
                            onClick={handlePublishToWordPress}
                            disabled={wpPublishStatus === 'publishing'}
                        >
                            {wpPublishStatus === 'publishing' ? publishingButtonText : publishButtonText}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

interface BulkPublishModalProps {
    items: ContentItem[];
    onClose: () => void;
    publishItem: (item: ContentItem, password: string) => Promise<{ success: boolean; message: React.ReactNode; link?: string; }>;
    wpPassword: string;
    onPublishSuccess: (originalUrl: string) => void;
}

const BulkPublishModal = ({ items, onClose, publishItem, wpPassword, onPublishSuccess }: BulkPublishModalProps) => {
    const [publishState, setPublishState] = useState<Record<string, { status: 'queued' | 'publishing' | 'success' | 'error', message: React.ReactNode }>>(() => {
        const initialState: Record<string, any> = {};
        items.forEach(item => {
            initialState[item.id] = { status: 'queued', message: 'In queue' };
        });
        return initialState;
    });
    const [isPublishing, setIsPublishing] = useState(false);
    const [isComplete, setIsComplete] = useState(false);

    const handleStartPublishing = async () => {
        setIsPublishing(true);
        setIsComplete(false);
        
        await processConcurrently(
            items,
            async (item) => {
                setPublishState(prev => ({ ...prev, [item.id]: { status: 'publishing', message: 'Publishing...' } }));
                const result = await publishItem(item, wpPassword);
                setPublishState(prev => ({ ...prev, [item.id]: { status: result.success ? 'success' : 'error', message: result.message } }));
                if (result.success && item.originalUrl) {
                    onPublishSuccess(item.originalUrl);
                }
            },
            3 // Concurrently publish 3 at a time to avoid overwhelming the server
        );

        setIsPublishing(false);
        setIsComplete(true);
    };

    return (
        <div className="modal-overlay" onClick={isPublishing ? undefined : onClose}>
            <div className="modal-content small-modal" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h2>Bulk Publish to WordPress</h2>
                    {!isPublishing && <button className="modal-close-btn" onClick={onClose} aria-label="Close modal">&times;</button>}
                </div>
                <div className="modal-body">
                    <p>The following {items.length} articles will be published sequentially to your WordPress site. Please do not close this window until the process is complete.</p>
                    <ul className="bulk-publish-list">
                        {items.map(item => (
                            <li key={item.id} className="bulk-publish-item">
                                <span className="bulk-publish-item-title" title={item.title}>{item.title}</span>
                                <div className="bulk-publish-item-status">
                                    {publishState[item.id].status === 'queued' && <span style={{ color: 'var(--text-light-color)' }}>Queued</span>}
                                    {publishState[item.id].status === 'publishing' && <><div className="spinner"></div><span>Publishing...</span></>}
                                    {publishState[item.id].status === 'success' && <span className="success">✓ Success</span>}
                                    {publishState[item.id].status === 'error' && <span className="error">✗ Error</span>}
                                </div>
                            </li>
                        ))}
                    </ul>
                     {Object.values(publishState).some(s => s.status === 'error') &&
                        <div className="result error" style={{marginTop: '1.5rem'}}>
                            Some articles failed to publish. Check your WordPress credentials, ensure the REST API is enabled, and try again.
                        </div>
                    }
                </div>
                <div className="modal-footer">
                    {isComplete ? (
                        <button className="btn" onClick={onClose}>Close</button>
                    ) : (
                        <button className="btn" onClick={handleStartPublishing} disabled={isPublishing}>
                            {isPublishing ? `Publishing... (${Object.values(publishState).filter(s => s.status === 'success' || s.status === 'error').length}/${items.length})` : `Publish ${items.length} Articles`}
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
};


// --- Main App Component ---
const App = () => {
    const [currentStep, setCurrentStep] = useState(1);
    
    // Step 1: API Keys & Config
    const [apiKeys, setApiKeys] = useState(() => {
        const saved = localStorage.getItem('apiKeys');
        return saved ? JSON.parse(saved) : { geminiApiKey: '', openaiApiKey: '', anthropicApiKey: '', openrouterApiKey: '', serperApiKey: '', groqApiKey: '' };
    });
    const [apiKeyStatus, setApiKeyStatus] = useState({ gemini: 'idle', openai: 'idle', anthropic: 'idle', openrouter: 'idle', serper: 'idle', groq: 'idle' } as Record<string, 'idle' | 'validating' | 'valid' | 'invalid'>);
    const [editingApiKey, setEditingApiKey] = useState<string | null>(null);
    const [apiClients, setApiClients] = useState<{ gemini: GoogleGenAI | null, openai: OpenAI | null, anthropic: Anthropic | null, openrouter: OpenAI | null, groq: OpenAI | null }>({ gemini: null, openai: null, anthropic: null, openrouter: null, groq: null });
    const [selectedModel, setSelectedModel] = useState(() => localStorage.getItem('selectedModel') || 'gemini');
    const [selectedGroqModel, setSelectedGroqModel] = useState(() => localStorage.getItem('selectedGroqModel') || AI_MODELS.GROQ_MODELS[0]);
    const [openrouterModels, setOpenrouterModels] = useState<string[]>(AI_MODELS.OPENROUTER_DEFAULT);
    const [geoTargeting, setGeoTargeting] = useState(() => {
        const saved = localStorage.getItem('geoTargeting');
        return saved ? JSON.parse(saved) : { enabled: false, location: '' };
    });
    const [useGoogleSearch, setUseGoogleSearch] = useState(false);


    // Step 2: Content Strategy
    const [contentMode, setContentMode] = useState('bulk'); // 'bulk', 'single', or 'imageGenerator'
    const [topic, setTopic] = useState('');
    const [primaryKeyword, setPrimaryKeyword] = useState('');
    const [sitemapUrl, setSitemapUrl] = useState('');
    const [isCrawling, setIsCrawling] = useState(false);
    const [crawlMessage, setCrawlMessage] = useState('');
    const [crawlProgress, setCrawlProgress] = useState({ current: 0, total: 0 });
    const [existingPages, setExistingPages] = useState<SitemapPage[]>([]);
    const [wpConfig, setWpConfig] = useState(() => {
        const saved = localStorage.getItem('wpConfig');
        return saved ? JSON.parse(saved) : { url: '', username: '' };
    });
    const [wpPassword, setWpPassword] = useState(() => localStorage.getItem('wpPassword') || '');
    const [wpConnectionStatus, setWpConnectionStatus] = useState<'idle' | 'verifying' | 'valid' | 'invalid'>('idle');
    const [wpConnectionMessage, setWpConnectionMessage] = useState<React.ReactNode>('');


    // Image Generator State
    const [imagePrompt, setImagePrompt] = useState('');
    const [numImages, setNumImages] = useState(1);
    const [aspectRatio, setAspectRatio] = useState('1:1');
    const [isGeneratingImages, setIsGeneratingImages] = useState(false);
    const [generatedImages, setGeneratedImages] = useState<{ src: string, prompt: string }[]>([]); // Array of { src: string, prompt: string }
    const [imageGenerationError, setImageGenerationError] = useState('');

    // Step 3: Generation & Review
    const [items, dispatch] = useReducer(itemsReducer, []);
    const [isGenerating, setIsGenerating] = useState(false);
    const [generationProgress, setGenerationProgress] = useState({ current: 0, total: 0 });
    const [selectedItems, setSelectedItems] = useState(new Set<string>());
    const [filter, setFilter] = useState('');
    const [sortConfig, setSortConfig] = useState({ key: 'title', direction: 'asc' });
    const [selectedItemForReview, setSelectedItemForReview] = useState<ContentItem | null>(null);
    const [isBulkPublishModalOpen, setIsBulkPublishModalOpen] = useState(false);
    const stopGenerationRef = useRef(new Set<string>());
    const isMobile = useMemo(() => window.innerWidth <= 767, []);
    
    // Content Hub State
    const [hubSearchFilter, setHubSearchFilter] = useState('');
    const [hubStatusFilter, setHubStatusFilter] = useState('All');
    const [hubSortConfig, setHubSortConfig] = useState<{key: string, direction: 'asc' | 'desc'}>({ key: 'default', direction: 'desc' });
    const [isAnalyzingHealth, setIsAnalyzingHealth] = useState(false);
    const [healthAnalysisProgress, setHealthAnalysisProgress] = useState({ current: 0, total: 0 });
    const [selectedHubPages, setSelectedHubPages] = useState(new Set<string>());
    
    // Web Worker
    const workerRef = useRef<Worker | null>(null);

    // --- Effects ---
    
    // Persist settings to localStorage
    useEffect(() => { localStorage.setItem('apiKeys', JSON.stringify(apiKeys)); }, [apiKeys]);
    useEffect(() => { localStorage.setItem('selectedModel', selectedModel); }, [selectedModel]);
    useEffect(() => { localStorage.setItem('selectedGroqModel', selectedGroqModel); }, [selectedGroqModel]);
    useEffect(() => { localStorage.setItem('wpConfig', JSON.stringify(wpConfig)); }, [wpConfig]);
    useEffect(() => { localStorage.setItem('wpPassword', wpPassword); }, [wpPassword]);
    useEffect(() => { localStorage.setItem('geoTargeting', JSON.stringify(geoTargeting)); }, [geoTargeting]);


    // Initialize Web Worker
    useEffect(() => {
        const workerCode = `
            self.addEventListener('message', async (e) => {
                const { type, payload } = e.data;

                const fetchWithProxies = ${fetchWithProxies.toString()};
                const extractSlugFromUrl = ${extractSlugFromUrl.toString()};

                if (type === 'CRAWL_SITEMAP') {
                    const { sitemapUrl } = payload;
                    const pageDataMap = new Map();
                    const crawledSitemapUrls = new Set();
                    const sitemapsToCrawl = [sitemapUrl];
                    
                    try {
                        self.postMessage({ type: 'CRAWL_UPDATE', payload: { message: 'Discovering all pages from sitemap(s)...' } });
                        while (sitemapsToCrawl.length > 0) {
                            const currentSitemapUrl = sitemapsToCrawl.shift();
                            if (!currentSitemapUrl || crawledSitemapUrls.has(currentSitemapUrl)) continue;

                            crawledSitemapUrls.add(currentSitemapUrl);
                            self.postMessage({ type: 'CRAWL_UPDATE', payload: { message: \`Parsing sitemap: \${currentSitemapUrl.substring(0, 100)}...\` } });

                            const response = await fetchWithProxies(currentSitemapUrl);
                            const text = await response.text();
                            
                            const initialUrlCount = pageDataMap.size;
                            const sitemapRegex = /<sitemap>\\s*<loc>(.*?)<\\/loc>\\s*<\\/sitemap>/g;
                            const urlBlockRegex = /<url>([\\s\\S]*?)<\\/url>/g;
                            let match;
                            let isSitemapIndex = false;

                            while((match = sitemapRegex.exec(text)) !== null) {
                                sitemapsToCrawl.push(match[1]);
                                isSitemapIndex = true;
                            }

                            while((match = urlBlockRegex.exec(text)) !== null) {
                                const block = match[1];
                                const locMatch = /<loc>(.*?)<\\/loc>/.exec(block);
                                if (locMatch) {
                                    const loc = locMatch[1];
                                    if (!pageDataMap.has(loc)) {
                                        const lastmodMatch = /<lastmod>(.*?)<\\/lastmod>/.exec(block);
                                        const lastmod = lastmodMatch ? lastmodMatch[1] : null;
                                        pageDataMap.set(loc, { lastmod });
                                    }
                                }
                            }

                            if (!isSitemapIndex && pageDataMap.size === initialUrlCount) {
                                self.postMessage({ type: 'CRAWL_UPDATE', payload: { message: \`Using fallback parser for: \${currentSitemapUrl.substring(0, 100)}...\` } });
                                const genericLocRegex = /<loc>(.*?)<\\/loc>/g;
                                while((match = genericLocRegex.exec(text)) !== null) {
                                    const loc = match[1].trim();
                                    if (loc.startsWith('http') && !pageDataMap.has(loc)) {
                                        pageDataMap.set(loc, { lastmod: null });
                                    }
                                }
                            }
                        }

                        const discoveredPages = Array.from(pageDataMap.entries()).map(([url, data]) => {
                            const currentDate = new Date();
                            let daysOld = null;
                            if (data.lastmod) {
                                const lastModDate = new Date(data.lastmod);
                                if (!isNaN(lastModDate.getTime())) {
                                    daysOld = Math.round((currentDate.getTime() - lastModDate.getTime()) / (1000 * 3600 * 24));
                                }
                            }
                            return {
                                id: url,
                                title: url, // Use URL as initial title
                                slug: extractSlugFromUrl(url),
                                lastMod: data.lastmod,
                                wordCount: null,
                                crawledContent: null,
                                healthScore: null,
                                updatePriority: null,
                                justification: null,
                                daysOld: daysOld,
                                isStale: false, // Will be calculated after content analysis
                                publishedState: 'none'
                            };
                        });

                        if (discoveredPages.length === 0) {
                             self.postMessage({ type: 'CRAWL_COMPLETE', payload: { pages: [], message: 'Crawl complete, but no page URLs were found.' } });
                             return;
                        }

                        self.postMessage({ type: 'CRAWL_COMPLETE', payload: { pages: discoveredPages, message: \`Discovery successful! Found \${discoveredPages.length} pages. Click 'Analyze Health' to process content.\` } });

                    } catch (error) {
                        self.postMessage({ type: 'CRAWL_ERROR', payload: { message: \`An error occurred during crawl: \${error.message}\` } });
                    }
                }
            });
        `;
        const blob = new Blob([workerCode], { type: 'application/javascript' });
        workerRef.current = new Worker(URL.createObjectURL(blob));

        workerRef.current.onmessage = (e) => {
            const { type, payload } = e.data;
            switch (type) {
                case 'CRAWL_UPDATE':
                    if (payload.message) setCrawlMessage(payload.message);
                    break;
                case 'CRAWL_COMPLETE':
                    setCrawlMessage(payload.message || 'Crawl complete.');
                    setExistingPages(payload.pages || []);
                    setIsCrawling(false);
                    break;
                case 'CRAWL_ERROR':
                    setCrawlMessage(payload.message);
                    setIsCrawling(false);
                    break;
            }
        };

        return () => {
            workerRef.current?.terminate();
        };
    }, []);

    // Clear hub page selection when filters change to avoid confusion
    useEffect(() => {
        setSelectedHubPages(new Set());
    }, [hubSearchFilter, hubStatusFilter]);

     const filteredAndSortedHubPages = useMemo(() => {
        let filtered = [...existingPages];

        // Status filter
        if (hubStatusFilter !== 'All') {
            filtered = filtered.filter(page => page.updatePriority === hubStatusFilter);
        }

        // Search filter
        if (hubSearchFilter) {
            filtered = filtered.filter(page =>
                page.title.toLowerCase().includes(hubSearchFilter.toLowerCase()) ||
                page.id.toLowerCase().includes(hubSearchFilter.toLowerCase())
            );
        }

        // Sorting
        if (hubSortConfig.key) {
            filtered.sort((a, b) => {
                 if (hubSortConfig.key === 'default') {
                    // 1. Stale content first (true is "smaller" so it comes first with asc)
                    if (a.isStale !== b.isStale) {
                        return a.isStale ? -1 : 1;
                    }
                    // 2. Older content first
                    if (a.daysOld !== b.daysOld) {
                        return (b.daysOld ?? 0) - (a.daysOld ?? 0);
                    }
                    // 3. Thinner content first
                    return (a.wordCount ?? 0) - (b.wordCount ?? 0);
                }

                let valA = a[hubSortConfig.key as keyof typeof a];
                let valB = b[hubSortConfig.key as keyof typeof b];

                // Handle boolean sorting for 'isStale'
                if (typeof valA === 'boolean' && typeof valB === 'boolean') {
                    if (valA === valB) return 0;
                    if (hubSortConfig.direction === 'asc') {
                        return valA ? -1 : 1; // true comes first
                    }
                    return valA ? 1 : -1; // false comes first
                }

                // Handle null or undefined values for sorting
                if (valA === null || valA === undefined) valA = hubSortConfig.direction === 'asc' ? Infinity : -Infinity;
                if (valB === null || valB === undefined) valB = hubSortConfig.direction === 'asc' ? Infinity : -Infinity;

                if (valA < valB) {
                    return hubSortConfig.direction === 'asc' ? -1 : 1;
                }
                if (valA > valB) {
                    return hubSortConfig.direction === 'asc' ? 1 : -1;
                }
                return 0;
            });
        }


        return filtered;
    }, [existingPages, hubSearchFilter, hubStatusFilter, hubSortConfig]);

    const validateApiKey = useCallback(debounce(async (provider: string, key: string) => {
        if (!key) {
            setApiKeyStatus(prev => ({ ...prev, [provider]: 'idle' }));
            setApiClients(prev => ({ ...prev, [provider]: null }));
            return;
        }

        setApiKeyStatus(prev => ({ ...prev, [provider]: 'validating' }));

        try {
            let client;
            let isValid = false;
            switch (provider) {
                case 'gemini':
                    client = new GoogleGenAI({ apiKey: key });
                    await callAiWithRetry(() =>
                        (client as GoogleGenAI).models.generateContent({ model: AI_MODELS.GEMINI_FLASH, contents: 'test' })
                    );
                    isValid = true;
                    break;
                case 'openai':
                    client = new OpenAI({ apiKey: key, dangerouslyAllowBrowser: true });
                    await callAiWithRetry(() => client.models.list());
                    isValid = true;
                    break;
                case 'anthropic':
                    client = new Anthropic({ apiKey: key });
                    await callAiWithRetry(() => client.messages.create({
                        model: AI_MODELS.ANTHROPIC_HAIKU,
                        max_tokens: 1,
                        messages: [{ role: "user", content: "test" }],
                    }));
                    isValid = true;
                    break;
                 case 'openrouter':
                    client = new OpenAI({
                        baseURL: "https://openrouter.ai/api/v1",
                        apiKey: key,
                        dangerouslyAllowBrowser: true,
                        defaultHeaders: {
                            'HTTP-Referer': window.location.href,
                            'X-Title': 'WP Content Optimizer Pro',
                        }
                    });
                    await callAiWithRetry(() => client.chat.completions.create({
                        model: 'google/gemini-2.5-flash',
                        messages: [{ role: "user", content: "test" }],
                        max_tokens: 1
                    }));
                    isValid = true;
                    break;
                case 'groq':
                    client = new OpenAI({
                        baseURL: "https://api.groq.com/openai/v1",
                        apiKey: key,
                        dangerouslyAllowBrowser: true,
                    });
                    await callAiWithRetry(() => (client as OpenAI).chat.completions.create({
                        model: AI_MODELS.GROQ_MODELS[1], // Use a small model for testing
                        messages: [{ role: "user", content: "test" }],
                        max_tokens: 1
                    }));
                    isValid = true;
                    break;
                 case 'serper':
                    const serperResponse = await fetchWithProxies("https://google.serper.dev/search", {
                        method: 'POST',
                        headers: {
                            'X-API-KEY': key,
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ q: 'test' })
                    });
                    if (serperResponse.ok) {
                        isValid = true;
                    } else {
                        const errorBody = await serperResponse.json().catch(() => ({ message: `Serper validation failed with status ${serperResponse.status}` }));
                        throw new Error(errorBody.message || `Serper validation failed with status ${serperResponse.status}`);
                    }
                    break;
            }

            if (isValid) {
                setApiKeyStatus(prev => ({ ...prev, [provider]: 'valid' }));
                if (client) {
                     setApiClients(prev => ({ ...prev, [provider]: client as any }));
                }
                setEditingApiKey(null);
            } else {
                 throw new Error("Validation check failed.");
            }
        } catch (error) {
            console.error(`${provider} API key validation failed:`, error);
            setApiKeyStatus(prev => ({ ...prev, [provider]: 'invalid' }));
            setApiClients(prev => ({ ...prev, [provider]: null }));
        }
    }, 500), []);
    
     useEffect(() => {
        Object.entries(apiKeys).forEach(([key, value]) => {
            if (value) {
                validateApiKey(key.replace('ApiKey', ''), value);
            }
        });
    }, []); // Run only on initial mount to validate saved keys

    const handleApiKeyChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        const { name, value } = e.target;
        const provider = name.replace('ApiKey', '');
        setApiKeys(prev => ({ ...prev, [name]: value }));
        validateApiKey(provider, value);
    };
    
    const handleOpenrouterModelsChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setOpenrouterModels(e.target.value.split('\n').map(m => m.trim()).filter(Boolean));
    };


    const handleNextStep = () => setCurrentStep(prev => prev + 1);
    const handlePrevStep = () => setCurrentStep(prev => prev - 1);

    const handleHubSort = (key: string) => {
        let direction: 'asc' | 'desc' = 'asc';
        if (hubSortConfig.key === key && hubSortConfig.direction === 'asc') {
            direction = 'desc';
        }
        setHubSortConfig({ key, direction });
    };

    const stopHealthAnalysisRef = useRef(false);
    const handleStopHealthAnalysis = () => {
        stopHealthAnalysisRef.current = true;
    };

    const handleAnalyzeContentHealth = async () => {
        const pagesToAnalyze = existingPages.filter(p => !p.crawledContent);
        if (pagesToAnalyze.length === 0) {
            alert("No new pages available to analyze. All discovered pages have already been processed.");
            return;
        }

        const client = apiClients[selectedModel as keyof typeof apiClients];
        if (!client) {
            alert("API client not available. Please check your API key in Step 1.");
            return;
        }
        
        stopHealthAnalysisRef.current = false;
        setIsAnalyzingHealth(true);
        setHealthAnalysisProgress({ current: 0, total: pagesToAnalyze.length });

        try {
            await processConcurrently(
                pagesToAnalyze,
                async (page) => {
                    const cacheKey = `health-analysis-${page.id}`;
                    const cached = sessionStorage.getItem(cacheKey);
                    if(cached) {
                        const parsed = JSON.parse(cached);
                        setExistingPages(prev => prev.map(p => p.id === page.id ? { ...p, ...parsed } : p));
                        return;
                    }

                    try {
                        // --- STAGE 1: Fetch Page Content ---
                        let pageHtml = '';
                        try {
                            const pageResponse = await fetchWithProxies(page.id);
                            pageHtml = await pageResponse.text();
                        } catch (fetchError: any) {
                             throw new Error(`Failed to fetch page content: ${fetchError.message}`);
                        }

                        const titleMatch = pageHtml.match(/<title>([\s\S]*?)<\/title>/i);
                        const title = titleMatch ? titleMatch[1] : 'Untitled Page';

                        let bodyText = pageHtml
                            .replace(/<script[\s\S]*?<\/script>/gi, '')
                            .replace(/<style[\s\S]*?<\/style>/gi, '')
                            .replace(/<nav[\s\S]*?<\/nav>/gi, '')
                            .replace(/<footer[\s\S]*?<\/footer>/gi, '')
                            .replace(/<header[\s\S]*?<\/header>/gi, '')
                            .replace(/<aside[\s\S]*?<\/aside>/gi, '')
                            .replace(/<[^>]+>/g, ' ')
                            .replace(/\s+/g, ' ')
                            .trim();
                        
                        const wordCount = bodyText.split(/\s+/).filter(Boolean).length;
                        
                        const currentYear = new Date().getFullYear();
                        const yearInTitleMatch = title.match(/\b(201[5-9]|202[0-3])\b/);
                        const isStale = yearInTitleMatch ? parseInt(yearInTitleMatch[0], 10) < currentYear : false;

                        setExistingPages(prev => prev.map(p => p.id === page.id ? { ...p, title, wordCount, crawledContent: bodyText, isStale } : p));
                        
                        // --- STAGE 2: AI Health Analysis ---
                        if (wordCount < 100) {
                            throw new Error("Content is too thin for analysis.");
                        }

                        const template = PROMPT_TEMPLATES.content_health_analyzer;
                        const contentSnippet = bodyText.substring(0, 12000); // Use a generous snippet
                        const userPrompt = template.userPrompt(contentSnippet);

                        let responseText: string | null = '';
                        switch (selectedModel) {
                            case 'gemini':
                                const geminiClient = apiClients.gemini;
                                if (!geminiClient) throw new Error("Gemini client not initialized");
                                const geminiResponse = await callAiWithRetry(() => geminiClient.models.generateContent({
                                    model: AI_MODELS.GEMINI_FLASH,
                                    contents: userPrompt,
                                    config: { systemInstruction: template.systemInstruction, responseMimeType: "application/json" }
                                }));
                                responseText = geminiResponse.text;
                                break;
                            case 'openai':
                                const openaiClient = apiClients.openai;
                                if (!openaiClient) throw new Error("OpenAI client not initialized");
                                const openaiResponse = await callAiWithRetry(() => openaiClient.chat.completions.create({
                                    model: AI_MODELS.OPENAI_GPT4_TURBO,
                                    messages: [{ role: "system", content: template.systemInstruction }, { role: "user", content: userPrompt }],
                                    response_format: { type: "json_object" },
                                }));
                                responseText = openaiResponse.choices[0].message.content ?? '';
                                break;
                            case 'openrouter':
                                let openrouterResponseText: string | null = '';
                                let lastError: Error | null = null;
                                const openrouterClient = apiClients.openrouter;
                                if (!openrouterClient) throw new Error("OpenRouter client not initialized");
                                for (const modelName of openrouterModels) {
                                    try {
                                        console.log(`[OpenRouter] Attempting health analysis with model: ${modelName}`);
                                        const response = await callAiWithRetry(() => openrouterClient.chat.completions.create({
                                            model: modelName,
                                            messages: [{ role: "system", content: template.systemInstruction }, { role: "user", content: userPrompt }],
                                            response_format: { type: "json_object" },
                                        }));

                                        const content = response.choices[0].message.content ?? '';
                                        if (!content) throw new Error("Empty response from model.");

                                        extractJson(content);

                                        openrouterResponseText = content;
                                        lastError = null;
                                        break; // Success
                                    } catch (error) {
                                        console.error(`OpenRouter model '${modelName}' failed during health analysis. Trying next...`, error);
                                        lastError = error as Error;
                                    }
                                }
                                if (lastError && !openrouterResponseText) throw lastError;
                                responseText = openrouterResponseText;
                                break;
                            case 'groq':
                                const groqClient = apiClients.groq;
                                if (!groqClient) throw new Error("Groq client not initialized");
                                const groqResponse = await callAiWithRetry(() => groqClient.chat.completions.create({
                                    model: selectedGroqModel,
                                    messages: [{ role: "system", content: template.systemInstruction }, { role: "user", content: userPrompt }],
                                    response_format: { type: "json_object" },
                                }));
                                responseText = groqResponse.choices[0].message.content ?? '';
                                break;
                            case 'anthropic':
                                const anthropicClient = apiClients.anthropic;
                                if (!anthropicClient) throw new Error("Anthropic client not initialized");
                                const anthropicResponse = await callAiWithRetry(() => anthropicClient.messages.create({
                                    model: AI_MODELS.ANTHROPIC_HAIKU,
                                    max_tokens: 4096,
                                    system: template.systemInstruction,
                                    messages: [{ role: "user", content: userPrompt }],
                                }));
                                responseText = anthropicResponse.content.map(c => c.text).join("");
                                break;
                        }

                        const parsedJson = JSON.parse(extractJson(responseText!));
                        const { healthScore, updatePriority, justification } = parsedJson;
                        sessionStorage.setItem(cacheKey, JSON.stringify({ title, wordCount, crawledContent: bodyText, isStale, healthScore, updatePriority, justification }));
                        setExistingPages(prev => prev.map(p => p.id === page.id ? { ...p, healthScore, updatePriority, justification } : p));
                    } catch (error: any) {
                        console.error(`Failed to analyze content for ${page.id}:`, error);
                        setExistingPages(prev => prev.map(p => p.id === page.id ? { ...p, healthScore: 0, updatePriority: 'Error', justification: error.message.substring(0, 100) } : p));
                    }
                },
                8, // Increased concurrency for page fetching
                (completed, total) => {
                    setHealthAnalysisProgress({ current: completed, total: total });
                },
                () => stopHealthAnalysisRef.current
            );
        } catch(error) {
            console.error("Content health analysis process was interrupted or failed:", error);
        } finally {
            setIsAnalyzingHealth(false);
        }
    };


    const handlePlanRewrite = (page: SitemapPage) => {
        const newItem: ContentItem = { id: page.title, title: page.title, type: 'standard', originalUrl: page.id, status: 'idle', statusText: 'Ready to Rewrite', generatedContent: null, crawledContent: page.crawledContent };
        dispatch({ type: 'SET_ITEMS', payload: [newItem] });
        handleNextStep();
    };

    const handleCreatePillar = (page: SitemapPage) => {
        const newItem: ContentItem = { id: page.title, title: page.title, type: 'pillar', originalUrl: page.id, status: 'idle', statusText: 'Ready to Generate', generatedContent: null, crawledContent: page.crawledContent };
        dispatch({ type: 'SET_ITEMS', payload: [newItem] });
        handleNextStep();
    };

    const handleToggleHubPageSelect = (pageId: string) => {
        setSelectedHubPages(prev => {
            const newSet = new Set(prev);
            if (newSet.has(pageId)) {
                newSet.delete(pageId);
            } else {
                newSet.add(pageId);
            }
            return newSet;
        });
    };

    const handleToggleHubPageSelectAll = () => {
        if (selectedHubPages.size === filteredAndSortedHubPages.length) {
            setSelectedHubPages(new Set());
        } else {
            setSelectedHubPages(new Set(filteredAndSortedHubPages.map(p => p.id)));
        }
    };

    const handleRewriteSelected = () => {
        const selectedPages = existingPages.filter(p => selectedHubPages.has(p.id));
        if (selectedPages.length === 0) return;

        const newItems: ContentItem[] = selectedPages.map(page => ({
            id: page.title,
            title: page.title,
            type: 'standard',
            originalUrl: page.id,
            status: 'idle',
            statusText: 'Ready to Rewrite',
            generatedContent: null,
            crawledContent: page.crawledContent
        }));
        dispatch({ type: 'SET_ITEMS', payload: newItems });
        setSelectedHubPages(new Set());
        handleNextStep();
    };

    const handleCreatePillarSelected = () => {
        const selectedPages = existingPages.filter(p => selectedHubPages.has(p.id));
        if (selectedPages.length === 0) return;

        const newItems: ContentItem[] = selectedPages.map(page => ({
            id: page.title,
            title: page.title,
            type: 'pillar',
            originalUrl: page.id,
            status: 'idle',
            statusText: 'Ready to Generate',
            generatedContent: null,
            crawledContent: page.crawledContent
        }));
        dispatch({ type: 'SET_ITEMS', payload: newItems });
        setSelectedHubPages(new Set());
        handleNextStep();
    };


    const handleCrawlSitemap = async () => {
        if (!sitemapUrl) {
            setCrawlMessage('Please enter a sitemap URL.');
            return;
        }

        setIsCrawling(true);
        setCrawlMessage('');
        setCrawlProgress({ current: 0, total: 0 });
        setExistingPages([]);
        
        workerRef.current?.postMessage({ type: 'CRAWL_SITEMAP', payload: { sitemapUrl } });
    };
    
    const handleGenerateClusterPlan = async () => {
        setIsGenerating(true);
        dispatch({ type: 'SET_ITEMS', payload: [] });
        const client = apiClients[selectedModel as keyof typeof apiClients];
        if (!client) {
             dispatch({ type: 'UPDATE_STATUS', payload: { id: 'cluster-planner', status: 'error', statusText: 'API Client not initialized.' } });
             setIsGenerating(false);
            return;
        }

        try {
            const template = PROMPT_TEMPLATES.cluster_planner;
            let systemInstruction = template.systemInstruction;
            if (geoTargeting.enabled && geoTargeting.location) {
                systemInstruction = systemInstruction.replace('{{GEO_TARGET_INSTRUCTIONS}}', `All titles must be geo-targeted for "${geoTargeting.location}".`);
            } else {
                 systemInstruction = systemInstruction.replace('{{GEO_TARGET_INSTRUCTIONS}}', '');
            }
            
            const userPrompt = template.userPrompt(topic);
            let responseText: string | null = '';
            
             switch (selectedModel) {
                case 'gemini':
                    const geminiClient = apiClients.gemini;
                    if (!geminiClient) throw new Error("Gemini client not initialized");
                    const geminiResponse = await callAiWithRetry(() => geminiClient.models.generateContent({
                        model: AI_MODELS.GEMINI_FLASH,
                        contents: userPrompt,
                        config: { systemInstruction, responseMimeType: "application/json" }
                    }));
                    responseText = geminiResponse.text;
                    break;
                case 'openai':
                     const openaiClient = apiClients.openai;
                    if (!openaiClient) throw new Error("OpenAI client not initialized");
                    const openaiResponse = await callAiWithRetry(() => openaiClient.chat.completions.create({
                        model: AI_MODELS.OPENAI_GPT4_TURBO,
                        messages: [{ role: "system", content: systemInstruction }, { role: "user", content: userPrompt }],
                        response_format: { type: "json_object" },
                    }));
                    responseText = openaiResponse.choices[0].message.content ?? '';
                    break;
                case 'openrouter':
                    let openrouterResponseText: string | null = '';
                    let lastError: Error | null = null;
                    const openrouterClient = apiClients.openrouter;
                    if (!openrouterClient) throw new Error("OpenRouter client not initialized");
                    
                    for (const modelName of openrouterModels) {
                        try {
                            console.log(`[OpenRouter] Attempting cluster plan with model: ${modelName}`);
                            const response = await callAiWithRetry(() => openrouterClient.chat.completions.create({
                                model: modelName,
                                messages: [{ role: "system", content: systemInstruction }, { role: "user", content: userPrompt }],
                                response_format: { type: "json_object" },
                            }));
                            
                            const content = response.choices[0].message.content ?? '';
                            if (!content) throw new Error("Empty response from model.");
                            
                            extractJson(content); 

                            openrouterResponseText = content;
                            lastError = null;
                            break; // Success
                        } catch (error) {
                            console.error(`OpenRouter model '${modelName}' failed for cluster plan. Trying next...`, error);
                            lastError = error as Error;
                        }
                    }
                    if (lastError && !openrouterResponseText) throw lastError;
                    responseText = openrouterResponseText;
                    break;
                 case 'groq':
                    const groqClient = apiClients.groq;
                    if (!groqClient) throw new Error("Groq client not initialized");
                    const groqResponse = await callAiWithRetry(() => groqClient.chat.completions.create({
                        model: selectedGroqModel,
                        messages: [{ role: "system", content: systemInstruction }, { role: "user", content: userPrompt }],
                        response_format: { type: "json_object" },
                    }));
                    responseText = groqResponse.choices[0].message.content ?? '';
                    break;
                 case 'anthropic':
                    const anthropicClient = apiClients.anthropic;
                    if (!anthropicClient) throw new Error("Anthropic client not initialized");
                    const anthropicResponse = await callAiWithRetry(() => anthropicClient.messages.create({
                        model: AI_MODELS.ANTHROPIC_OPUS,
                        max_tokens: 4096,
                        system: systemInstruction,
                        messages: [{ role: "user", content: userPrompt }],
                    }));
                    responseText = anthropicResponse.content.map(c => c.text).join("");
                    break;
            }

            const parsedJson = JSON.parse(extractJson(responseText!));
            const newItems: Partial<ContentItem>[] = [
                { id: parsedJson.pillarTitle, title: parsedJson.pillarTitle, type: 'pillar' },
                ...parsedJson.clusterTitles.map((title: string) => ({ id: title, title, type: 'cluster' }))
            ];
            dispatch({ type: 'SET_ITEMS', payload: newItems });
            handleNextStep();

        } catch (error: any) {
            console.error("Error generating cluster plan:", error);
            const errorItem: ContentItem = {
                id: 'error-item', title: 'Failed to Generate Plan', type: 'standard', status: 'error',
                statusText: `An error occurred: ${error.message}`, generatedContent: null, crawledContent: null
            };
             dispatch({ type: 'SET_ITEMS', payload: [errorItem] });
        } finally {
            setIsGenerating(false);
        }
    };
    
    const handleGenerateSingleFromKeyword = () => {
        if (!primaryKeyword) return;
        const newItem: Partial<ContentItem> = { id: primaryKeyword, title: primaryKeyword, type: 'standard' };
        dispatch({ type: 'SET_ITEMS', payload: [newItem] });
        handleNextStep();
    };

    // --- Image Generation Logic ---
    const handleGenerateImages = async () => {
        const geminiClient = apiClients.gemini;
        if (!geminiClient || apiKeyStatus.gemini !== 'valid') {
            setImageGenerationError('Please enter a valid Gemini API key in Step 1 to generate images.');
            return;
        }
        if (!imagePrompt) {
            setImageGenerationError('Please enter a prompt to generate an image.');
            return;
        }

        setIsGeneratingImages(true);
        setGeneratedImages([]);
        setImageGenerationError('');

        try {
            const geminiResponse = await callAiWithRetry(() => geminiClient.models.generateImages({
                model: AI_MODELS.GEMINI_IMAGEN,
                prompt: imagePrompt,
                config: {
                    numberOfImages: numImages,
                    outputMimeType: 'image/jpeg',
                    aspectRatio: aspectRatio as "1:1" | "16:9" | "9:16" | "4:3" | "3:4",
                },
            }));
            const imagesData = geminiResponse.generatedImages.map(img => ({
                src: `data:image/jpeg;base64,${img.image.imageBytes}`,
                prompt: imagePrompt
            }));
            
            setGeneratedImages(imagesData);

        } catch (error: any) {
            console.error("Image generation failed:", error);
            setImageGenerationError(`An error occurred: ${error.message}`);
        } finally {
            setIsGeneratingImages(false);
        }
    };

    const handleDownloadImage = (base64Data: string, prompt: string) => {
        const link = document.createElement('a');
        link.href = base64Data;
        const safePrompt = prompt.substring(0, 30).replace(/[^a-z0-9]/gi, '_').toLowerCase();
        link.download = `generated-image-${safePrompt}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const handleCopyText = (text: string) => {
        navigator.clipboard.writeText(text).catch(err => {
            console.error('Failed to copy text: ', err);
        });
    };

    // --- Step 3 Logic ---
    const handleToggleSelect = (itemId: string) => {
        setSelectedItems(prev => {
            const newSet = new Set(prev);
            if (newSet.has(itemId)) {
                newSet.delete(itemId);
            } else {
                newSet.add(itemId);
            }
            return newSet;
        });
    };

    const handleToggleSelectAll = () => {
        if (selectedItems.size === filteredAndSortedItems.length) {
            setSelectedItems(new Set());
        } else {
            setSelectedItems(new Set(filteredAndSortedItems.map(item => item.id)));
        }
    };
    
     const filteredAndSortedItems = useMemo(() => {
        let sorted = [...items];
        if (sortConfig.key) {
            sorted.sort((a, b) => {
                const valA = a[sortConfig.key as keyof typeof a];
                const valB = b[sortConfig.key as keyof typeof a];

                if (valA < valB) {
                    return sortConfig.direction === 'asc' ? -1 : 1;
                }
                if (valA > valB) {
                    return sortConfig.direction === 'asc' ? 1 : -1;
                }
                return 0;
            });
        }
        if (filter) {
            return sorted.filter(item => item.title.toLowerCase().includes(filter.toLowerCase()));
        }
        return sorted;
    }, [items, filter, sortConfig]);

    const handleSort = (key: string) => {
        let direction: 'asc' | 'desc' = 'asc';
        if (sortConfig.key === key && sortConfig.direction === 'asc') {
            direction = 'desc';
        }
        setSortConfig({ key, direction });
    };

    const handleGenerateSingle = (item: ContentItem) => {
        stopGenerationRef.current.delete(item.id);
        setIsGenerating(true);
        setGenerationProgress({ current: 0, total: 1 });
        generateContent([item]);
    };

    const handleGenerateSelected = () => {
        stopGenerationRef.current.clear();
        const itemsToGenerate = items.filter(item => selectedItems.has(item.id));
        if (itemsToGenerate.length > 0) {
            setIsGenerating(true);
            setGenerationProgress({ current: 0, total: itemsToGenerate.length });
            generateContent(itemsToGenerate);
        }
    };
    
     const handleStopGeneration = (itemId: string | null = null) => {
        if (itemId) {
            stopGenerationRef.current.add(itemId);
             dispatch({
                type: 'UPDATE_STATUS',
                payload: { id: itemId, status: 'idle', statusText: 'Stopped by user' }
            });
        } else {
            // Stop all
            items.forEach(item => {
                if (item.status === 'generating') {
                    stopGenerationRef.current.add(item.id);
                     dispatch({
                        type: 'UPDATE_STATUS',
                        payload: { id: item.id, status: 'idle', statusText: 'Stopped by user' }
                    });
                }
            });
            setIsGenerating(false);
        }
    };

    const generateImageWithFallback = async (prompt: string): Promise<string | null> => {
        // Priority 1: OpenAI DALL-E 3
        if (apiClients.openai && apiKeyStatus.openai === 'valid') {
            try {
                console.log("Attempting image generation with OpenAI DALL-E 3...");
                const openaiImgResponse = await callAiWithRetry(() => apiClients.openai!.images.generate({ model: AI_MODELS.OPENAI_DALLE3, prompt, n: 1, size: '1792x1024', response_format: 'b64_json' }));
                const base64Image = openaiImgResponse.data[0].b64_json;
                if (base64Image) {
                    console.log("OpenAI image generation successful.");
                    return `data:image/png;base64,${base64Image}`;
                }
            } catch (error) {
                console.warn("OpenAI image generation failed, falling back to Gemini.", error);
            }
        }

        // Priority 2: Gemini Imagen
        if (apiClients.gemini && apiKeyStatus.gemini === 'valid') {
            try {
                 console.log("Attempting image generation with Google Gemini Imagen...");
                 const geminiImgResponse = await callAiWithRetry(() => apiClients.gemini!.models.generateImages({ model: AI_MODELS.GEMINI_IMAGEN, prompt: prompt, config: { numberOfImages: 1, outputMimeType: 'image/jpeg', aspectRatio: '16:9' } }));
                 const base64Image = geminiImgResponse.generatedImages[0].image.imageBytes;
                 if (base64Image) {
                    console.log("Gemini image generation successful.");
                    return `data:image/jpeg;base64,${base64Image}`;
                 }
            } catch (error) {
                 console.error("Gemini image generation also failed.", error);
            }
        }
        
        console.error("All image generation services failed or are unavailable.");
        return null;
    };
    
    const callAI = useCallback(async (
        promptKey: keyof typeof PROMPT_TEMPLATES,
        promptArgs: any[],
        responseFormat: 'json' | 'html' = 'json'
    ): Promise<string> => {
        const client = apiClients[selectedModel as keyof typeof apiClients];
        if (!client) throw new Error(`API Client for '${selectedModel}' not initialized.`);

        const template = PROMPT_TEMPLATES[promptKey];
        // Geo-targeting replacement is only relevant for the cluster planner
        const systemInstruction = (promptKey === 'cluster_planner') 
            ? template.systemInstruction.replace('{{GEO_TARGET_INSTRUCTIONS}}', (geoTargeting.enabled && geoTargeting.location) ? `All titles must be geo-targeted for "${geoTargeting.location}".` : '')
            : template.systemInstruction;
            
        // @ts-ignore
        const userPrompt = template.userPrompt(...promptArgs);
        
        let responseText: string | null = '';

        switch (selectedModel) {
            case 'gemini':
                const geminiResponse = await callAiWithRetry(() => (client as GoogleGenAI).models.generateContent({
                    model: AI_MODELS.GEMINI_FLASH,
                    contents: userPrompt,
                    config: { systemInstruction, responseMimeType: responseFormat === 'json' ? "application/json" : "text/plain" }
                }));
                responseText = geminiResponse.text;
                break;
            case 'openai':
                const openaiResponse = await callAiWithRetry(() => (client as OpenAI).chat.completions.create({
                    model: AI_MODELS.OPENAI_GPT4_TURBO,
                    messages: [{ role: "system", content: systemInstruction }, { role: "user", content: userPrompt }],
                    ...(responseFormat === 'json' && { response_format: { type: "json_object" } })
                }));
                responseText = openaiResponse.choices[0].message.content;
                break;
            case 'openrouter':
                let lastError: Error | null = null;
                for (const modelName of openrouterModels) {
                    try {
                        console.log(`[OpenRouter] Attempting '${promptKey}' with model: ${modelName}`);
                        const response = await callAiWithRetry(() => (client as OpenAI).chat.completions.create({
                            model: modelName,
                            messages: [{ role: "system", content: systemInstruction }, { role: "user", content: userPrompt }],
                             ...(responseFormat === 'json' && { response_format: { type: "json_object" } })
                        }));
                        const content = response.choices[0].message.content;
                        if (!content) throw new Error("Empty response from model.");
                        responseText = content;
                        lastError = null; // Clear error on success
                        break; // Success
                    } catch (error: any) {
                        console.error(`OpenRouter model '${modelName}' failed for '${promptKey}'. Trying next...`, error);
                        lastError = error;
                    }
                }
                if (lastError && !responseText) throw lastError;
                break;
            case 'groq':
                 const groqResponse = await callAiWithRetry(() => (client as OpenAI).chat.completions.create({
                    model: selectedGroqModel,
                    messages: [{ role: "system", content: systemInstruction }, { role: "user", content: userPrompt }],
                    ...(responseFormat === 'json' && { response_format: { type: "json_object" } })
                }));
                responseText = groqResponse.choices[0].message.content;
                break;
            case 'anthropic':
                const anthropicResponse = await callAiWithRetry(() => (client as Anthropic).messages.create({
                    model: promptKey.includes('section') ? AI_MODELS.ANTHROPIC_HAIKU : AI_MODELS.ANTHROPIC_OPUS,
                    max_tokens: 4096,
                    system: systemInstruction,
                    messages: [{ role: "user", content: userPrompt }],
                }));
                responseText = anthropicResponse.content.map(c => c.text).join("");
                break;
        }

        if (!responseText) {
            throw new Error(`AI returned an empty response for the '${promptKey}' stage.`);
        }

        return responseText;
    }, [apiClients, selectedModel, geoTargeting, openrouterModels, selectedGroqModel]);


    const generateContent = useCallback(async (itemsToGenerate: ContentItem[]) => {
        let generatedCount = 0;

        for (const item of itemsToGenerate) {
            if (stopGenerationRef.current.has(item.id)) continue;

            dispatch({ type: 'UPDATE_STATUS', payload: { id: item.id, status: 'generating', statusText: 'Initializing...' } });

            let rawResponseForDebugging: any = null;
            let processedContent: GeneratedContent | null = null;
             
            try {
                let semanticKeywords: string[] | null = null;
                let serpData: any[] | null = null;
                let youtubeVideos: any[] | null = null;
                
                const isPillar = item.type === 'pillar';

                // --- STAGE 1: SERP & Keyword Intelligence ---
                if (apiKeys.serperApiKey && apiKeyStatus.serper === 'valid') {
                    dispatch({ type: 'UPDATE_STATUS', payload: { id: item.id, status: 'generating', statusText: 'Stage 1/4: Fetching SERP Data...' } });
                    const cacheKey = `serp-${item.title}`;
                    const cachedSerp = apiCache.get(cacheKey);

                    if (cachedSerp) {
                         serpData = cachedSerp.serpData;
                         youtubeVideos = cachedSerp.youtubeVideos;
                    } else {
                        try {
                            const serperResponse = await fetchWithProxies("https://google.serper.dev/search", {
                                method: 'POST',
                                headers: { 'X-API-KEY': apiKeys.serperApiKey, 'Content-Type': 'application/json' },
                                body: JSON.stringify({ q: item.title })
                            });
                            if (!serperResponse.ok) throw new Error(`Serper API failed with status ${serperResponse.status}`);
                            const serperJson = await serperResponse.json();
                            serpData = serperJson.organic ? serperJson.organic.slice(0, 10) : [];
                            
                            const videoCandidates = new Map<string, any>();
                            const videoQueries = [`"${item.title}" tutorial`, `how to ${item.title}`, item.title];

                            for (const query of videoQueries) {
                                if (videoCandidates.size >= 10) break;
                                try {
                                    const videoResponse = await fetchWithProxies("https://google.serper.dev/videos", {
                                        method: 'POST', headers: { 'X-API-KEY': apiKeys.serperApiKey, 'Content-Type': 'application/json' }, body: JSON.stringify({ q: query })
                                    });
                                    if (videoResponse.ok) {
                                        const json = await videoResponse.json();
                                        for (const v of (json.videos || [])) {
                                            const videoId = extractYouTubeID(v.link);
                                            if (videoId && !videoCandidates.has(videoId)) videoCandidates.set(videoId, { ...v, videoId });
                                        }
                                    }
                                } catch (e) { console.warn(`Video search failed for "${query}".`, e); }
                            }
                            youtubeVideos = getUniqueYoutubeVideos(Array.from(videoCandidates.values()), YOUTUBE_EMBED_COUNT);
                            apiCache.set(cacheKey, { serpData, youtubeVideos });
                        } catch (serpError) {
                            console.error("Failed to fetch SERP data:", serpError);
                        }
                    }
                }

                dispatch({ type: 'UPDATE_STATUS', payload: { id: item.id, status: 'generating', statusText: 'Stage 1/4: Analyzing Topic...' } });
                const skCacheKey = `sk-${item.title}`;
                if (apiCache.get(skCacheKey)) {
                    semanticKeywords = apiCache.get(skCacheKey);
                } else {
                    const skResponseText = await callAI('semantic_keyword_generator', [item.title]);
                    const parsedSk = JSON.parse(extractJson(skResponseText));
                    semanticKeywords = parsedSk.semanticKeywords;
                    apiCache.set(skCacheKey, semanticKeywords);
                }

                if (stopGenerationRef.current.has(item.id)) break;

                // --- STAGE 2: Generate Metadata and Outline ---
                dispatch({ type: 'UPDATE_STATUS', payload: { id: item.id, status: 'generating', statusText: 'Stage 2/4: Generating Article Outline...' } });
                const outlineResponseText = await callAI('content_meta_and_outline', [item.title, semanticKeywords, serpData, existingPages, item.crawledContent]);
                rawResponseForDebugging = outlineResponseText; // Save for debugging if JSON parse fails
                const metaAndOutline = JSON.parse(extractJson(outlineResponseText));

                // --- STAGE 3: Generate Content Section-by-Section ---
                let contentParts: string[] = [];
                contentParts.push(metaAndOutline.introduction);
                contentParts.push(`<h3>Key Takeaways</h3><ul>${metaAndOutline.keyTakeaways.map((li: string) => `<li>${li}</li>`).join('')}</ul>`);
                
                if (metaAndOutline.imageDetails?.[0]?.placeholder) contentParts.push(`<p>${metaAndOutline.imageDetails[0].placeholder}</p>`);

                for (let i = 0; i < metaAndOutline.outline.length; i++) {
                    const heading = metaAndOutline.outline[i];
                    if (stopGenerationRef.current.has(item.id)) break;
                    dispatch({ type: 'UPDATE_STATUS', payload: { id: item.id, status: 'generating', statusText: `Stage 3/4: Writing Section ${i + 1}/${metaAndOutline.outline.length}` } });
                    
                    const sectionHtml = await callAI('write_article_section', [item.title, metaAndOutline.title, heading, existingPages], 'html');
                    contentParts.push(`<h2>${heading}</h2>\n${sectionHtml}`);

                    if (i === 2 && youtubeVideos?.[0]) {
                        const vid = youtubeVideos[0];
                        contentParts.push(`<div class="video-container"><iframe width="100%" height="410" src="${vid.embedUrl}" frameborder="0" allowfullscreen title="${vid.title}"></iframe></div>`);
                    }
                    if (i === 6 && youtubeVideos?.[1]) {
                        const vid = youtubeVideos[1];
                        contentParts.push(`<div class="video-container"><iframe width="100%" height="410" src="${vid.embedUrl}" frameborder="0" allowfullscreen title="${vid.title}"></iframe></div>`);
                    }
                    if (i === 4 && metaAndOutline.imageDetails?.[1]?.placeholder) {
                        contentParts.push(`<p>${metaAndOutline.imageDetails[1].placeholder}</p>`);
                    }
                }
                if (stopGenerationRef.current.has(item.id)) break;
                
                contentParts.push(`<h2>Frequently Asked Questions</h2>`);
                for (const faq of metaAndOutline.faqSection) {
                     if (stopGenerationRef.current.has(item.id)) break;
                     const answerHtml = await callAI('write_faq_answer', [faq.question], 'html');
                     contentParts.push(`<h3>${faq.question}</h3>\n${answerHtml}`);
                }
                
                contentParts.push(metaAndOutline.conclusion);
                contentParts.push('[REFERENCES_PLACEHOLDER]');
                
                let finalContentHtml = contentParts.join('\n\n');

                // --- STAGE 4: Final Assembly & Post-Processing ---
                processedContent = normalizeGeneratedContent(metaAndOutline, item.title);
                processedContent.content = finalContentHtml;
                processedContent.primaryKeyword = item.title;
                processedContent.semanticKeywords = semanticKeywords || [];
                
                // QUALITY GATES
                enforceWordCount(processedContent.content, isPillar ? TARGET_MIN_WORDS_PILLAR : TARGET_MIN_WORDS, isPillar ? TARGET_MAX_WORDS_PILLAR : TARGET_MAX_WORDS);
                checkHumanWritingScore(processedContent.content);
                
                // LINKING & EMBED PROTOCOL
                processedContent.content = validateAndRepairInternalLinks(processedContent.content, existingPages);
                processedContent.content = enforceInternalLinkQuota(processedContent.content, existingPages, processedContent.primaryKeyword, MIN_INTERNAL_LINKS);
                processedContent.content = processInternalLinks(processedContent.content, existingPages);
                
                if (youtubeVideos) {
                    processedContent.content = enforceUniqueVideoEmbeds(processedContent.content, youtubeVideos);
                }
                processedContent.content = processedContent.content.replace(/<iframe[^>]+src="https:\/\/www\.youtube\.com\/embed\/[^>]+>/g, (match) => {
                    return match.replace(/width="[^"]*"/, 'width="100%"').replace(/height="[^"]*"/, 'height="410"');
                });
                
                if (processedContent.content.includes('[REFERENCES_PLACEHOLDER]')) {
                    let referencesHtml = '<h2>References</h2><ul>';
                    if (serpData && serpData.length > 0) {
                        serpData.slice(0, 8).forEach(ref => {
                            if (ref.link && ref.title) referencesHtml += `<li><a href="${ref.link}" target="_blank" rel="noopener noreferrer">${ref.title}</a></li>`;
                        });
                    }
                    referencesHtml += '</ul>';
                    processedContent.content = processedContent.content.replace('[REFERENCES_PLACEHOLDER]', referencesHtml);
                }
                
                // Conditional Image Generation
                if (apiKeyStatus.openai === 'valid' || apiKeyStatus.gemini === 'valid') {
                    for (let i = 0; i < processedContent.imageDetails.length; i++) {
                        const imageDetail = processedContent.imageDetails[i];
                        if (stopGenerationRef.current.has(item.id)) break;
                        dispatch({ type: 'UPDATE_STATUS', payload: { id: item.id, status: 'generating', statusText: `Stage 4/4: Generating Image ${i + 1}/${processedContent.imageDetails.length}...` } });
                        
                        const generatedImageSrc = await generateImageWithFallback(imageDetail.prompt);
                        
                        if (generatedImageSrc) {
                            processedContent.imageDetails[i].generatedImageSrc = generatedImageSrc;
                            const imageHtml = `<figure class="wp-block-image size-large"><img src="${generatedImageSrc}" alt="${imageDetail.altText}" title="${imageDetail.title}"/><figcaption>${imageDetail.altText}</figcaption></figure>`;
                            if (processedContent.content.includes(imageDetail.placeholder)) {
                                processedContent.content = processedContent.content.replace(new RegExp(escapeRegExp(imageDetail.placeholder), 'g'), imageHtml);
                            }
                        } else {
                             if (imageDetail.placeholder) {
                                processedContent.content = processedContent.content.replace(new RegExp(escapeRegExp(imageDetail.placeholder), 'g'), `<!-- Image generation failed for prompt: "${imageDetail.prompt}" -->`);
                             }
                        }
                    }
                } else {
                    console.warn("No valid image generation API key found. Stripping image placeholders.");
                    processedContent.content = processedContent.content.replace(/\[IMAGE_\d+_PLACEHOLDER\]/g, '');
                }

                if (stopGenerationRef.current.has(item.id)) break;

                dispatch({ type: 'UPDATE_STATUS', payload: { id: item.id, status: 'generating', statusText: 'Stage 4/4: Finalizing...' }});
                dispatch({ type: 'SET_CONTENT', payload: { id: item.id, content: processedContent } });

            } catch (error: any) {
                if (error instanceof ContentTooShortError) {
                    const partialContent = { ...processedContent, content: error.content } as GeneratedContent;
                    console.warn(`Content for "${item.title}" was too short, but is being saved for review.`, error);
                    dispatch({ type: 'SET_CONTENT', payload: { id: item.id, content: partialContent } });
                    dispatch({ type: 'UPDATE_STATUS', payload: { id: item.id, status: 'error', statusText: `Quality Check Failed: ${error.message}` } });
                } else {
                    console.error(`Error generating content for "${item.title}":`, error);
                    console.log(`[DEBUG] Raw AI Response for "${item.title}":`, rawResponseForDebugging);
                    dispatch({ type: 'UPDATE_STATUS', payload: { id: item.id, status: 'error', statusText: `Error: ${error.message.substring(0, 100)}...` } });
                }
            } finally {
                generatedCount++;
                setGenerationProgress({ current: generatedCount, total: itemsToGenerate.length });
            }
        }

        setIsGenerating(false);
    }, [apiKeys, apiKeyStatus, callAI, existingPages]);
    
    // --- WordPress Publishing Logic ---

    const getWpConnectionErrorHelp = (errorCode: 'CORS' | 'AUTH' | 'UNREACHABLE' | 'GENERAL', details: string) => {
        switch (errorCode) {
            case 'CORS':
                return (
                    <div className="cors-instructions">
                        <h4><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z" clipRule="evenodd" /></svg>Connection Blocked by Server (CORS)</h4>
                        <p><strong>Your credentials are likely correct</strong>, but your WordPress server is blocking the connection for security reasons. This is a common server configuration issue that you need to resolve on your website.</p>
                        <p><strong>Primary Solution:</strong></p>
                        <ol>
                            <li>Edit the <strong>.htaccess</strong> file in the main directory of your WordPress installation.</li>
                            <li>Add the following line to the <strong>very top</strong> of the file:
                            <pre><code>SetEnvIf Authorization "(.*)" HTTP_AUTHORIZATION=$1</code></pre>
                            </li>
                        </ol>
                        <p><strong>Other things to check:</strong></p>
                        <ul>
                            <li>Temporarily disable security plugins (like Wordfence, iThemes Security) to see if they are interfering.</li>
                            <li>Ensure your WordPress User is an <strong>Administrator</strong> or <strong>Editor</strong>.</li>
                        </ul>
                    </div>
                );
            case 'AUTH':
                return <p><strong>Authentication Failed (401).</strong> Please double-check your WordPress Username and your Application Password. Ensure the password was copied correctly and is not revoked.</p>;
            case 'UNREACHABLE':
                 return <p><strong>Connection Failed.</strong> Could not reach the WordPress REST API. Please verify your <strong>WordPress URL</strong> is correct and that the site is online. Firewalls or security plugins could also be blocking all access.</p>;
            default:
                 return <p><strong>An unexpected error occurred:</strong> {details}</p>
        }
    };

    const publishItem = async (
        itemToPublish: ContentItem,
        currentWpPassword: string
    ): Promise<{ success: boolean; message: React.ReactNode; link?: string }> => {
        if (!wpConfig.url || !wpConfig.username || !currentWpPassword) {
            return { success: false, message: 'WordPress URL, Username, and Application Password are required.' };
        }
        if (!itemToPublish.generatedContent) {
            return { success: false, message: 'No content available to publish.' };
        }

        const { url, username } = wpConfig;
        const isUpdate = !!itemToPublish.originalUrl;
        let content = itemToPublish.generatedContent.content;
        const authHeader = `Basic ${btoa(`${username}:${currentWpPassword}`)}`;
        const apiUrlBase = url.replace(/\/+$/, '') + '/wp-json/wp/v2';

        // 1. Upload images and replace base64 src with WordPress URLs
        try {
            for (const imageDetail of itemToPublish.generatedContent.imageDetails) {
                if (imageDetail.generatedImageSrc) {
                    // Convert base64 to Blob
                    const imageResponse = await fetch(imageDetail.generatedImageSrc);
                    const blob = await imageResponse.blob();
                    
                    // Sanitize filename
                    const safeFilename = imageDetail.title
                        .replace(/[^a-z0-9]/gi, '-')
                        .toLowerCase()
                        .substring(0, 50) + '.jpg';

                    const formData = new FormData();
                    formData.append('file', blob, safeFilename);
                    formData.append('title', imageDetail.title);
                    formData.append('alt_text', imageDetail.altText);
                    formData.append('caption', imageDetail.altText);

                    // Use fetchWordPressWithRetry which now correctly handles auth
                    const mediaResponse = await fetchWordPressWithRetry(`${apiUrlBase}/media`, {
                        method: 'POST',
                        headers: {
                            'Authorization': authHeader,
                            // NOTE: Do NOT set 'Content-Type' for FormData, browser does it with boundary
                        },
                        body: formData
                    });

                    if (!mediaResponse.ok) {
                        if (mediaResponse.status === 401) throw new Error('Authentication failed (401). Check credentials.');
                        if (mediaResponse.status === 403) throw new Error('Forbidden (403). User may lack permissions or be blocked by security.');
                        const errorJson = await mediaResponse.json().catch(() => ({}));
                        throw new Error(errorJson.message || `Media upload failed with HTTP ${mediaResponse.status}`);
                    }

                    const mediaJson = await mediaResponse.json();
                    const wpImageUrl = mediaJson.source_url;

                    const newImgTag = `<img src="${wpImageUrl}" alt="${imageDetail.altText}" class="wp-image-${mediaJson.id}" title="${imageDetail.title}"/>`;
                    const oldImgTagRegex = new RegExp(`<img[^>]*src="${escapeRegExp(imageDetail.generatedImageSrc)}"[^>]*>`, 'g');
                    
                    if (oldImgTagRegex.test(content)) {
                        content = content.replace(oldImgTagRegex, newImgTag);
                    } else if (content.includes(imageDetail.placeholder)) {
                        const newFigureBlock = `<figure class="wp-block-image size-large">${newImgTag}<figcaption>${imageDetail.altText}</figcaption></figure>`;
                        content = content.replace(new RegExp(escapeRegExp(imageDetail.placeholder), 'g'), newFigureBlock);
                    }
                }
            }
// FIX: Changed catch parameter from `e: unknown` to `error: any` to allow direct property access and fix type errors.
        } catch (error: any) {
            if (error.name === 'TypeError' || error.message.toLowerCase().includes('cors')) {
                return { success: false, message: getWpConnectionErrorHelp('CORS', error.message) };
            }
            if (error.message.includes('401')) {
                 return { success: false, message: getWpConnectionErrorHelp('AUTH', error.message) };
            }
            return { success: false, message: getWpConnectionErrorHelp('GENERAL', `Media upload failed: ${error.message}`) };
        }

        // 2. Find Post ID for updates
        let postId: number | null = null;
        if (isUpdate) {
            try {
                const slug = extractSlugFromUrl(itemToPublish.originalUrl!);
                const postSearchResponse = await fetchWordPressWithRetry(`${apiUrlBase}/posts?slug=${slug}`, {
                    method: 'GET',
                    headers: { 'Authorization': authHeader }
                });
                 if (!postSearchResponse.ok) {
                    const errorJson = await postSearchResponse.json().catch(() => ({}));
                    throw new Error(errorJson.message || `Could not find existing post (HTTP ${postSearchResponse.status})`);
                }
                const posts = await postSearchResponse.json();
                if (posts.length > 0) {
                    postId = posts[0].id;
                } else {
                    console.warn(`Could not find existing post with slug "${slug}". A new post will be created.`);
                }
// FIX: Changed catch parameter from `e: unknown` to `error: any` to allow direct property access and fix type errors.
            } catch (error: any) {
                return { success: false, message: `Failed to find original post: ${error.message}` };
            }
        }

        // 3. Prepare post data
        const postData = {
            title: itemToPublish.generatedContent.title,
            content: content,
            status: 'publish',
            slug: itemToPublish.generatedContent.slug,
            excerpt: itemToPublish.generatedContent.metaDescription
        };

        // 4. Create or Update Post
        const endpoint = postId ? `${apiUrlBase}/posts/${postId}` : `${apiUrlBase}/posts`;
        try {
            const response = await fetchWordPressWithRetry(endpoint, {
                method: 'POST',
                headers: { 'Authorization': authHeader, 'Content-Type': 'application/json' },
                body: JSON.stringify(postData)
            });

            if (!response.ok) {
                const errorJson = await response.json().catch(() => ({}));
                return { success: false, message: `WP Error: ${errorJson.message || response.statusText}` };
            }

            const responseJson = await response.json();
            const actionText = postId ? 'updated' : 'published';
            const message = (
                <span>
                    Successfully {actionText}!
                    <a href={responseJson.link} target="_blank" rel="noopener noreferrer">
                        View Post
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M4.25 5.5a.75.75 0 00-.75.75v8.5c0 .414.336.75.75.75h8.5a.75.75 0 00.75-.75v-4a.75.75 0 011.5 0v4A2.25 2.25 0 0112.75 17H4.25A2.25 2.25 0 012 14.75v-8.5A2.25 2.25 0 014.25 4h5a.75.75 0 010 1.5h-5z" clipRule="evenodd" /><path fillRule="evenodd" d="M6.194 12.753a.75.75 0 001.06.053L16.5 4.44v2.81a.75.75 0 001.5 0v-4.5a.75.75 0 00-.75-.75h-4.5a.75.75 0 000 1.5h2.553l-9.056 8.19a.75.75 0 00.053 1.06z" clipRule="evenodd" /></svg>
                    </a>
                </span>
            );

            return { success: true, message: message, link: responseJson.link };
// FIX: Changed catch parameter from `e: unknown` to `error: any` to allow direct property access and fix type errors.
        } catch (error: any) {
             if (error.name === 'TypeError' || error.message.toLowerCase().includes('cors')) {
                return { success: false, message: getWpConnectionErrorHelp('CORS', error.message) };
            }
             return { success: false, message: getWpConnectionErrorHelp('GENERAL', error.message) };
        }
    };
    
    const verifyWpConnection = async () => {
        if (!wpConfig.url || !wpConfig.username || !wpPassword) {
            setWpConnectionStatus('invalid');
            setWpConnectionMessage('URL, Username, and Application Password are required.');
            return;
        }
        setWpConnectionStatus('verifying');
        setWpConnectionMessage('');

        const apiUrlBase = wpConfig.url.replace(/\/+$/, '') + '/wp-json';
        const authHeader = `Basic ${btoa(`${wpConfig.username}:${wpPassword}`)}`;

        try {
            // --- STAGE 1: Authenticated Request ---
            const usersResponse = await fetchWordPressWithRetry(`${apiUrlBase}/wp/v2/users/me?context=edit`, {
                method: 'GET',
                headers: { 'Authorization': authHeader }
            });

            if (!usersResponse.ok) {
                if (usersResponse.status === 401) {
                    throw { type: 'AUTH_ERROR', message: 'Authentication failed (401). Incorrect Username or Application Password.' };
                }
                const errorJson = await usersResponse.json().catch(() => ({}));
                throw { type: 'GENERAL_API_ERROR', message: errorJson.message || `API returned HTTP ${usersResponse.status}` };
            }

            const data = await usersResponse.json();
            const canPublish = data.capabilities?.publish_posts;
            if (!canPublish) {
                setWpConnectionStatus('invalid');
                setWpConnectionMessage(`Connected as ${data.name}, but user lacks permission to publish posts. Please use an Administrator or Editor account.`);
                return;
            }

            setWpConnectionStatus('valid');
            setWpConnectionMessage(`Success! Connected as ${data.name} with publish permissions.`);

        } catch (error: any) {
            // --- STAGE 2: Diagnose the Failure ---
            if (error.type === 'AUTH_ERROR') {
                setWpConnectionStatus('invalid');
                setWpConnectionMessage(getWpConnectionErrorHelp('AUTH', error.message));
                return;
            }
            if (error.name === 'TypeError' || error.message?.toLowerCase().includes('failed to fetch')) {
                // This is a network-level error, could be CORS or offline.
                // Let's check if the server is reachable at all without authentication.
                try {
                    const publicResponse = await fetchWordPressWithRetry(apiUrlBase, { method: 'GET' });
                    if (publicResponse.ok) {
                        // Server is online, so the authenticated request was specifically blocked. This is a classic CORS issue.
                        setWpConnectionStatus('invalid');
                        setWpConnectionMessage(getWpConnectionErrorHelp('CORS', error.message));
                    } else {
                        // The server is online but the base REST API endpoint is not working.
                        setWpConnectionStatus('invalid');
                        setWpConnectionMessage(getWpConnectionErrorHelp('UNREACHABLE', `The REST API is not working correctly (HTTP ${publicResponse.status}).`));
                    }
                } catch (publicError) {
                    // We can't even reach the server publicly.
                    setWpConnectionStatus('invalid');
                    setWpConnectionMessage(getWpConnectionErrorHelp('UNREACHABLE', 'The WordPress URL is incorrect or the site is offline.'));
                }
                return;
            }

            // General fallback
            setWpConnectionStatus('invalid');
            setWpConnectionMessage(getWpConnectionErrorHelp('GENERAL', error.message));
        }
    };


    // --- Review Modal Logic ---
    const handleOpenReview = (item: ContentItem) => {
        if (item.generatedContent) {
            setSelectedItemForReview(item);
        }
    };
    
    const handleCloseReview = () => setSelectedItemForReview(null);

    const handleSaveChanges = (itemId: string, updatedSeo: { title: string; metaDescription: string; slug: string; }, updatedContent: string) => {
        const itemToUpdate = items.find(i => i.id === itemId);
        if (itemToUpdate && itemToUpdate.generatedContent) {
            const updatedGeneratedContent: GeneratedContent = {
                ...itemToUpdate.generatedContent,
                ...updatedSeo,
                content: updatedContent
            };
            dispatch({ type: 'SET_CONTENT', payload: { id: itemId, content: updatedGeneratedContent } });
        }
        alert('Changes saved locally!');
    };

    const handlePublishSuccess = (originalUrl: string) => {
        setExistingPages(prev => prev.map(p => 
            p.id === originalUrl ? { ...p, publishedState: 'updated' } : p
        ));
    };

    // --- Render Logic ---
    
    const getWordCountClass = (count: number | null) => {
        if (count === null || count < 0) return '';
        if (count < 800) return 'thin-content';
        return '';
    };

    const getAgeClass = (days: number | null) => {
        if (days === null) return '';
        if (days > 730) return 'age-ancient'; // Over 2 years
        if (days > 365) return 'age-old'; // Over 1 year
        return '';
    };

    const calculateGenerationProgress = (statusText: string): string => {
        if (statusText.includes("Stage 1/4")) return "10%";
        if (statusText.includes("Stage 2/4")) return "25%";
        if (statusText.includes("Stage 3/4")) {
            const match = statusText.match(/Writing Section (\d+)\/(\d+)/);
            if (match) {
                const current = parseInt(match[1], 10);
                const total = parseInt(match[2], 10);
                if (total > 0) {
                    // Stage 3 is from 25% to 75%
                    return `${25 + (current / total) * 50}%`;
                }
            }
            return "50%";
        }
        if (statusText.includes("Stage 4/4")) {
             if (statusText.includes("Generating Image")) {
                const match = statusText.match(/Image (\d+)\/(\d+)/);
                 if (match) {
                    const current = parseInt(match[1], 10);
                    const total = parseInt(match[2], 10);
                    if (total > 0) {
                        // Stage 4 is from 75% to 100%
                        return `${75 + (current / total) * 25}%`;
                    }
                 }
             }
            if (statusText.includes("Finalizing")) return "99%";
            return "75%";
        }
        return "5%";
    };


    const renderStep = () => {
        switch (currentStep) {
            case 1:
                return (
                    <div className="setup-container">
                        <div className="setup-welcome">
                            <h1 className="usp-headline">Welcome to WP Content Optimizer Pro</h1>
                             <p className="usp-subheadline">
                                Your all-in-one solution for planning, generating, and optimizing high-quality SEO content at scale.
                                Let's get your API keys configured to unlock the power of AI.
                            </p>
                            <div className="features-grid">
                                <div className="feature">
                                    <div className="feature-icon"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" /></svg></div>
                                    <div className="feature-content">
                                        <h3>Multi-Provider AI</h3>
                                        <p>Connect to Gemini, OpenAI, Anthropic, or OpenRouter to use the best models for your needs.</p>
                                    </div>
                                </div>
                                 <div className="feature">
                                    <div className="feature-icon"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.286zm0 13.036h.008v.008h-.008v-.008z" /></svg></div>
                                    <div className="feature-content">
                                        <h3>SEO Optimized</h3>
                                        <p>Every article is crafted based on SEO best practices to help you rank higher on Google.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="setup-form">
                             <fieldset className="config-fieldset">
                                <legend>AI Provider</legend>
                                <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
                                    <option value="gemini">Google Gemini</option>
                                    <option value="openai">OpenAI</option>
                                    <option value="anthropic">Anthropic</option>
                                    <option value="openrouter">OpenRouter</option>
                                    <option value="groq">Groq</option>
                                </select>
                            </fieldset>
                             <fieldset className="config-fieldset full-width">
                                <legend>SERP Data Provider (Required)</legend>
                                <p className="help-text" style={{margin: '0 0 1rem 0'}}>
                                    An API key from <a href="https://serper.dev" target="_blank" rel="noopener noreferrer">Serper.dev</a> is required for real-time Google data, ensuring content accuracy and high-quality external links.
                                </p>
                                <ApiKeyInput provider="serper" value={apiKeys.serperApiKey} onChange={handleApiKeyChange} status={apiKeyStatus.serper} name="serperApiKey" placeholder="Enter your Serper.dev API Key" isEditing={editingApiKey === 'serper'} onEdit={() => setEditingApiKey('serper')} />
                            </fieldset>
                            <div className="config-forms-wrapper">
                                <fieldset className="config-fieldset">
                                    <legend>Google Gemini</legend>
                                    <ApiKeyInput provider="gemini" value={apiKeys.geminiApiKey} onChange={handleApiKeyChange} status={apiKeyStatus.gemini} name="geminiApiKey" placeholder="Enter your Google Gemini API Key" isEditing={editingApiKey === 'gemini'} onEdit={() => setEditingApiKey('gemini')} />
                                     <p className="help-text">
                                        Used for text and image generation. Recommended for high-quality, cost-effective results.
                                    </p>
                                </fieldset>
                                 <fieldset className="config-fieldset">
                                    <legend>OpenAI</legend>
                                    <ApiKeyInput provider="openai" value={apiKeys.openaiApiKey} onChange={handleApiKeyChange} status={apiKeyStatus.openai} name="openaiApiKey" placeholder="Enter your OpenAI API Key" isEditing={editingApiKey === 'openai'} onEdit={() => setEditingApiKey('openai')} />
                                </fieldset>
                                <fieldset className="config-fieldset">
                                    <legend>Anthropic</legend>
                                    <ApiKeyInput provider="anthropic" value={apiKeys.anthropicApiKey} onChange={handleApiKeyChange} status={apiKeyStatus.anthropic} name="anthropicApiKey" placeholder="Enter your Anthropic API Key" isEditing={editingApiKey === 'anthropic'} onEdit={() => setEditingApiKey('anthropic')} />
                                </fieldset>
                                <fieldset className="config-fieldset">
                                     <legend>OpenRouter</legend>
                                     <ApiKeyInput provider="openrouter" value={apiKeys.openrouterApiKey} onChange={handleApiKeyChange} status={apiKeyStatus.openrouter} name="openrouterApiKey" placeholder="Enter your OpenRouter API Key" isEditing={editingApiKey === 'openrouter'} onEdit={() => setEditingApiKey('openrouter')} />
                                     {selectedModel === 'openrouter' && (
                                         <div className="form-group" style={{marginTop: '1rem'}}>
                                            <label htmlFor="openrouterModels">Custom Model Names (one per line, for fallback)</label>
                                             <ApiKeyInput
                                                provider="openrouterModels"
                                                isTextArea={true}
                                                value={openrouterModels.join('\n')}
                                                onChange={handleOpenrouterModelsChange}
                                                status={'idle'}
                                                name="openrouterModels"
                                                placeholder={"google/gemini-2.5-flash\nanthropic/claude-3-haiku\nmicrosoft/wizardlm-2-8x22b\nopenrouter/auto"}
                                                isEditing={true} onEdit={()=>{}}
                                            />
                                            <p className="help-text" style={{marginTop: '0.5rem'}}>
                                                List models by priority. The tool will try them in order if one fails. Prioritize reliable models over free/unstable ones to avoid errors.
                                            </p>
                                         </div>
                                     )}
                                </fieldset>
                                <fieldset className="config-fieldset">
                                    <legend>Groq</legend>
                                    <ApiKeyInput provider="groq" value={apiKeys.groqApiKey} onChange={handleApiKeyChange} status={apiKeyStatus.groq} name="groqApiKey" placeholder="Enter your Groq API Key" isEditing={editingApiKey === 'groq'} onEdit={() => setEditingApiKey('groq')} />
                                     {selectedModel === 'groq' && (
                                        <div className="form-group" style={{marginTop: '1rem'}}>
                                            <label htmlFor="selectedGroqModel">Groq Model</label>
                                            <input
                                                list="groq-models-list"
                                                id="selectedGroqModel"
                                                name="selectedGroqModel"
                                                value={selectedGroqModel}
                                                onChange={e => setSelectedGroqModel(e.target.value)}
                                                placeholder="Select or type a model name..."
                                            />
                                            <datalist id="groq-models-list">
                                                {AI_MODELS.GROQ_MODELS.map(model => (
                                                    <option key={model} value={model} />
                                                ))}
                                            </datalist>
                                            <p className="help-text" style={{marginTop: '0.5rem'}}>
                                                Select a model from the list or type a custom model ID. Llama 3.1 70B offers the highest quality.
                                            </p>
                                        </div>
                                    )}
                                </fieldset>
                                <fieldset className="config-fieldset full-width">
                                    <legend>Advanced Settings</legend>
                                    <div className="form-group" style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: 0 }}>
                                        <input
                                            type="checkbox"
                                            id="geoEnabled"
                                            checked={geoTargeting.enabled}
                                            onChange={e => setGeoTargeting(p => ({ ...p, enabled: e.target.checked }))}
                                            style={{ width: 'auto' }}
                                        />
                                        <label htmlFor="geoEnabled" style={{ marginBottom: 0, flexGrow: 1 }}>
                                            Enable Geo-Targeting
                                        </label>
                                        <input
                                            type="text"
                                            value={geoTargeting.location}
                                            onChange={e => setGeoTargeting(p => ({ ...p, location: e.target.value }))}
                                            placeholder="e.g., 'New York City'"
                                            disabled={!geoTargeting.enabled}
                                            style={{ flexGrow: 2 }}
                                        />
                                    </div>
                                </fieldset>
                            </div>
                             <button className="btn" onClick={handleNextStep} disabled={apiKeyStatus[selectedModel as keyof typeof apiKeyStatus] !== 'valid' || apiKeyStatus.serper !== 'valid'}>
                                {apiKeyStatus[selectedModel as keyof typeof apiKeyStatus] !== 'valid' ? `Enter a Valid ${selectedModel.charAt(0).toUpperCase() + selectedModel.slice(1)} Key` : apiKeyStatus.serper !== 'valid' ? 'Enter a Valid Serper Key' : 'Proceed to Content Strategy'}
                            </button>
                        </div>
                    </div>
                );
            case 2:
                const priorityOptions = ['All', 'Critical', 'High', 'Medium', 'Healthy', 'Error'];
                return (
                    <div className="step-container full-width">
                        <div className="content-mode-toggle">
                             <button
                                className={contentMode === 'bulk' ? 'active' : ''}
                                onClick={() => setContentMode('bulk')}
                            >
                                Bulk Pillar & Cluster
                            </button>
                            <button
                                className={contentMode === 'single' ? 'active' : ''}
                                onClick={() => setContentMode('single')}
                            >
                                Single Article
                            </button>
                             <button
                                className={contentMode === 'imageGenerator' ? 'active' : ''}
                                onClick={() => setContentMode('imageGenerator')}
                            >
                                AI Image Generator
                            </button>
                        </div>

                        {contentMode === 'imageGenerator' && (
                            <div className="image-generator-container">
                                <div className="content-creation-hub">
                                    <h2>AI Image Generator</h2>
                                    <p>Describe the image you want to create. Be as specific as possible for the best results. Powered by Google's Imagen 4 model.</p>
                                    <div className="image-generator-form">
                                        <div className="form-group">
                                            <textarea
                                                id="imagePrompt"
                                                value={imagePrompt}
                                                onChange={(e) => setImagePrompt(e.target.value)}
                                                placeholder="e.g., A photo of an astronaut riding a horse on Mars, cinematic lighting"
                                                rows={4}
                                            />
                                        </div>
                                        <div className="image-controls-grid">
                                            <div className="form-group">
                                                <label htmlFor="numImages">Number of Images</label>
                                                <select id="numImages" value={numImages} onChange={e => setNumImages(Number(e.target.value))}>
                                                    {[1, 2, 3, 4].map(n => <option key={n} value={n}>{n}</option>)}
                                                </select>
                                            </div>
                                            <div className="form-group">
                                                <label htmlFor="aspectRatio">Aspect Ratio</label>
                                                <select id="aspectRatio" value={aspectRatio} onChange={e => setAspectRatio(e.target.value)}>
                                                    <option value="1:1">Square (1:1)</option>
                                                    <option value="16:9">Widescreen (16:9)</option>
                                                    <option value="9:16">Portrait (9:16)</option>
                                                    <option value="4:3">Landscape (4:3)</option>
                                                    <option value="3:4">Tall (3:4)</option>
                                                </select>
                                            </div>
                                        </div>
                                        <button className="btn" onClick={handleGenerateImages} disabled={isGeneratingImages} style={{width: '100%'}}>
                                            {isGeneratingImages ? 'Generating...' : `Generate ${numImages} Image${numImages > 1 ? 's' : ''}`}
                                        </button>
                                        {imageGenerationError && <p className="result error" style={{textAlign: 'left'}}>{imageGenerationError}</p>}
                                    </div>
                                </div>
                                <div className="image-gallery">
                                    {isGeneratingImages && Array.from({ length: numImages }).map((_, i) => (
                                        <div key={i} className="gallery-placeholder">
                                            <div className="key-status-spinner"></div>
                                            <span>Generating...</span>
                                        </div>
                                    ))}
                                    {generatedImages.map((image, index) => (
                                        <div key={index} className="image-card">
                                            <img src={image.src} alt={image.prompt} />
                                            <div className="image-card-overlay">
                                                <p className="image-card-prompt">{image.prompt}</p>
                                                <div className="image-card-actions">
                                                    <button className="btn btn-small btn-secondary" onClick={() => handleDownloadImage(image.src, image.prompt)}>Download</button>
                                                    <button className="btn btn-small btn-secondary" onClick={() => handleCopyText(image.prompt)}>Copy Prompt</button>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {contentMode === 'bulk' && (
                            <>
                                <div className="content-creation-hub">
                                    <h2>Create a Pillar & Cluster Content Plan</h2>
                                    <p>Enter a broad topic, and we'll generate a complete content plan with a main pillar article and several supporting cluster articles to establish topical authority.</p>
                                    <div className="content-form">
                                        <div className="form-group">
                                            <label htmlFor="topic" className="sr-only">Topic</label>
                                            <input type="text" id="topic" value={topic} onChange={(e) => setTopic(e.target.value)} placeholder="e.g., 'Digital Marketing for Small Businesses'" />
                                        </div>
                                        <button className="btn" onClick={handleGenerateClusterPlan} disabled={isGenerating || !topic}>
                                            {isGenerating ? 'Generating Plan...' : 'Generate Content Plan'}
                                        </button>
                                    </div>
                                </div>
                                <div style={{textAlign: 'center', margin: '2rem 0', fontWeight: 500, color: 'var(--text-light-color)'}}>OR</div>
                                <div className="content-creation-hub content-hub-section">
                                    <div className="content-hub-title" style={{justifyContent: 'center', marginBottom: '0.5rem'}}>
                                        <h3>Content Health Hub</h3>
                                        <div className="tooltip-container">
                                            <span className="info-icon">i</span>
                                            <div className="tooltip-content">
                                                <h4>How It Works</h4>
                                                <p>This tool analyzes your existing content to identify pages that need an update.</p>
                                                <ol>
                                                    <li><strong>Crawl Sitemap:</strong> We'll find all your public pages.</li>
                                                    <li><strong>Analyze Health:</strong> We'll use AI to score each page's SEO health.</li>
                                                    <li><strong>Rewrite or Re-Pillar:</strong> Turn low-scoring content into high-ranking assets.</li>
                                                </ol>
                                            </div>
                                        </div>
                                    </div>
                                    <p>Crawl your website to identify underperforming content and turn it into high-ranking assets.</p>
                                    <div className="sitemap-input-group">
                                        <div className="form-group">
                                            <label htmlFor="sitemapUrl">Sitemap URL</label>
                                            <input type="text" id="sitemapUrl" value={sitemapUrl} onChange={(e) => setSitemapUrl(e.target.value)} placeholder="https://yourwebsite.com/sitemap_index.xml" />
                                        </div>
                                         <button className="btn" onClick={handleCrawlSitemap} disabled={isCrawling}>
                                            {isCrawling ? 'Crawling...' : 'Crawl Sitemap'}
                                        </button>
                                        {crawlMessage && <p className={`result ${crawlMessage.toLowerCase().includes('error') ? 'error' : 'success'}`}>{crawlMessage}</p>}
                                    </div>
                                </div>
                            </>
                        )}

                        {contentMode === 'single' && (
                            <div className="content-creation-hub">
                                <h2>Generate a Single Article</h2>
                                <p>Enter a primary keyword to generate a single, highly-optimized blog post from scratch.</p>
                                <div className="content-form">
                                    <div className="form-group">
                                        <label htmlFor="primaryKeyword" className="sr-only">Primary Keyword</label>
                                        <input type="text" id="primaryKeyword" value={primaryKeyword} onChange={(e) => setPrimaryKeyword(e.target.value)} placeholder="e.g., 'Best SEO Tools for 2025'" />
                                    </div>
                                    <button className="btn" onClick={handleGenerateSingleFromKeyword} disabled={!primaryKeyword}>
                                        Plan Article
                                    </button>
                                </div>
                            </div>
                        )}
                        
                        {(contentMode === 'bulk' || contentMode === 'single') && (
                            <div className="wp-config-section" style={{marginTop: '2rem'}}>
                                <h3>WordPress Integration (Optional)</h3>
                                <p className="help-text">Enter your WordPress details to publish content directly from the app. You must install the <a href="https://wordpress.org/plugins/application-passwords/" target="_blank" rel="noopener noreferrer">Application Passwords</a> plugin to generate a password.</p>
                                <div className="wp-config-grid">
                                    <div className="form-group">
                                        <label htmlFor="wpUrl">WordPress URL</label>
                                        <input type="text" id="wpUrl" value={wpConfig.url} onChange={e => setWpConfig(prev => ({...prev, url: e.target.value}))} placeholder="https://yourwebsite.com" />
                                    </div>
                                    <div className="form-group">
                                        <label htmlFor="wpUsername">WordPress Username</label>
                                        <input type="text" id="wpUsername" value={wpConfig.username} onChange={e => setWpConfig(prev => ({...prev, username: e.target.value}))} placeholder="your-wp-username" />
                                    </div>
                                    <div className="form-group">
                                        <label htmlFor="wpPassword">Application Password</label>
                                        <input type="password" id="wpPassword" value={wpPassword} onChange={e => setWpPassword(e.target.value)} placeholder="xxxx xxxx xxxx xxxx xxxx xxxx" />
                                    </div>
                                </div>
                                <div style={{display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap'}}>
                                    <button className="btn btn-secondary" onClick={verifyWpConnection} disabled={wpConnectionStatus === 'verifying'}>
                                        {wpConnectionStatus === 'verifying' ? 'Verifying...' : 'Verify Connection'}
                                    </button>
                                    {wpConnectionMessage &&
                                        <div className={`wp-connection-status ${wpConnectionStatus}`}>
                                            {wpConnectionStatus === 'verifying' && <div className="spinner"></div>}
                                            {wpConnectionStatus === 'valid' && <svg className="success" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" /></svg>}
                                            {wpConnectionStatus === 'invalid' && <svg className="error" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" /></svg>}
                                            <div className={wpConnectionStatus === 'valid' ? 'success' : 'error'}>{wpConnectionMessage}</div>
                                        </div>
                                    }
                                </div>
                            </div>
                        )}

                         {existingPages.length > 0 && contentMode === 'bulk' && (
                             <div className="content-hub-section" style={{ marginTop: '2rem' }}>
                                <div className="table-toolbar">
                                    <div style={{display: 'flex', alignItems: 'center', gap: '1rem'}}>
                                        <button 
                                            className="btn" 
                                            onClick={handleAnalyzeContentHealth} 
                                            disabled={isAnalyzingHealth}
                                            title="Analyze content for pages that haven't been processed yet"
                                        >
                                            {isAnalyzingHealth ? `Analyzing... (${healthAnalysisProgress.current}/${healthAnalysisProgress.total})` : 'Analyze Content Health'}
                                        </button>
                                        {isAnalyzingHealth && <button className="btn btn-danger" onClick={handleStopHealthAnalysis}>Stop</button>}
                                    </div>
                                    <div className="hub-actions-and-filters">
                                        <div className="hub-filters">
                                            <select className="hub-status-filter" value={hubStatusFilter} onChange={(e) => setHubStatusFilter(e.target.value)}>
                                                {priorityOptions.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                                            </select>
                                            <input
                                                type="search"
                                                className="table-search-input"
                                                placeholder="Filter by title or URL..."
                                                value={hubSearchFilter}
                                                onChange={(e) => setHubSearchFilter(e.target.value)}
                                            />
                                        </div>
                                    </div>
                                </div>
                                
                                {selectedHubPages.size > 0 && (
                                    <div className="bulk-action-bar">
                                        <span>{selectedHubPages.size} page(s) selected</span>
                                        <div className="bulk-action-buttons">
                                            <button className="btn btn-small" onClick={handleRewriteSelected}>Plan Rewrite</button>
                                            <button className="btn btn-small btn-secondary" onClick={handleCreatePillarSelected}>Create Pillar</button>
                                        </div>
                                    </div>
                                )}

                                <div className="table-container">
                                    <table className={`content-table ${isMobile ? 'mobile-cards' : ''}`}>
                                        <thead>
                                            <tr>
                                                <th className="checkbox-cell">
                                                    <input type="checkbox"
                                                        checked={selectedHubPages.size > 0 && selectedHubPages.size === filteredAndSortedHubPages.length}
                                                        onChange={handleToggleHubPageSelectAll}
                                                        aria-label="Select all pages"
                                                    />
                                                </th>
                                                <th className="sortable" onClick={() => handleHubSort('title')}>Post Title</th>
                                                <th className="sortable numeric-cell" onClick={() => handleHubSort('healthScore')}>Health</th>
                                                <th className="sortable numeric-cell" onClick={() => handleHubSort('wordCount')}>Words</th>
                                                <th className="sortable numeric-cell" onClick={() => handleHubSort('daysOld')}>Age</th>
                                                <th className="actions-cell">Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {filteredAndSortedHubPages.map(page => (
                                                <tr key={page.id}>
                                                    <td data-label="Select" className="checkbox-cell">
                                                        <input type="checkbox"
                                                            checked={selectedHubPages.has(page.id)}
                                                            onChange={() => handleToggleHubPageSelect(page.id)}
                                                            aria-label={`Select ${page.title}`}
                                                        />
                                                    </td>
                                                    <td data-label="Post Title">
                                                        <div>
                                                            <a href={page.id} target="_blank" rel="noopener noreferrer" className="hub-post-link">
                                                                <div className="post-title">
                                                                     {page.isStale && <span className="stale-badge priority-critical">OUTDATED</span>}
                                                                     {page.publishedState === 'updated' && <span className="published-badge updated">✓ Rewritten</span>}
                                                                    {page.title}
                                                                </div>
                                                                <div className="post-url">{page.id.replace(/^(https?:\/\/)?(www\.)?/, '').substring(0, 50)}...</div>
                                                            </a>
                                                        </div>
                                                    </td>
                                                    <td data-label="Health" className="numeric-cell">
                                                        {page.healthScore !== null ? (
                                                            <div style={{display: 'flex', flexDirection: 'column', alignItems: 'flex-end'}}>
                                                                <span className={`priority-badge priority-${page.updatePriority?.toLowerCase()}`}>{page.healthScore} - {page.updatePriority}</span>
                                                                <span className="help-text" style={{margin: 0, textAlign: 'right'}}>{page.justification}</span>
                                                            </div>
                                                        ) : (
                                                            isAnalyzingHealth ? <div className="skeleton-loader" style={{width: '80px', height: '20px'}}></div> : <span className="help-text">Not analyzed</span>
                                                        )}
                                                    </td>
                                                    <td data-label="Words" className="numeric-cell">
                                                        <span className={getWordCountClass(page.wordCount)}>{page.wordCount ?? 'N/A'}</span>
                                                    </td>
                                                     <td data-label="Age" className="numeric-cell">
                                                        <span className={getAgeClass(page.daysOld)}>{page.daysOld !== null ? `${page.daysOld} days` : 'N/A'}</span>
                                                    </td>
                                                    <td data-label="Actions" className="actions-cell">
                                                        {page.crawledContent ? (
                                                            <div className="action-button-group">
                                                                <button className="btn btn-small" onClick={() => handlePlanRewrite(page)}>Plan Rewrite</button>
                                                                <button className="btn btn-small btn-secondary" onClick={() => handleCreatePillar(page)}>Create Pillar</button>
                                                            </div>
                                                        ) : <span className="help-text">Analyze first</span>}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        )}
                    </div>
                );
            case 3:
                const doneAndSelectedCount = items.filter(item => selectedItems.has(item.id) && item.status === 'done').length;
                return (
                    <div className="step-container full-width">
                        <div className="table-toolbar">
                             <div className="selection-toolbar-actions">
                                {isGenerating ? (
                                    <>
                                        <span>Generating in progress...</span>
                                        <button className="btn btn-danger" onClick={() => handleStopGeneration()}>Stop All</button>
                                    </>
                                ) : (
                                     <button className="btn" onClick={handleGenerateSelected} disabled={selectedItems.size === 0}>
                                        {`Generate ${selectedItems.size} Selected`}
                                    </button>
                                )}
                                <button
                                    className="btn btn-success"
                                    onClick={() => setIsBulkPublishModalOpen(true)}
                                    disabled={doneAndSelectedCount === 0}
                                    title="Publish selected and completed items to WordPress"
                                >
                                    🚀 Bulk Publish Selected ({doneAndSelectedCount})
                                </button>
                            </div>
                            <input
                                type="search"
                                className="table-search-input"
                                placeholder="Filter by title..."
                                value={filter}
                                onChange={(e) => setFilter(e.target.value)}
                            />
                        </div>
                        {isGenerating && (
                            <div className="generation-progress-bar">
                                <div className="progress-bar-fill" style={{ width: `${(generationProgress.current / generationProgress.total) * 100}%` }}></div>
                                <div className="progress-text">
                                    Overall Progress: {generationProgress.current} / {generationProgress.total} Complete
                                </div>
                            </div>
                        )}
                        <div className="table-container">
                            <table className={`content-table ${isMobile ? 'mobile-cards' : ''}`}>
                                <thead>
                                    <tr>
                                        <th className="checkbox-cell">
                                            <input type="checkbox"
                                                checked={selectedItems.size > 0 && selectedItems.size === filteredAndSortedItems.length}
                                                onChange={handleToggleSelectAll}
                                                aria-label="Select all items"
                                            />
                                        </th>
                                        <th className="sortable" onClick={() => handleSort('title')}>Title</th>
                                        <th className="sortable" onClick={() => handleSort('type')}>Type</th>
                                        <th className="sortable" onClick={() => handleSort('status')}>Status</th>
                                        <th className="actions-cell">Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {filteredAndSortedItems.map(item => (
                                        <tr key={item.id} className={`${item.type}-row ${selectedItems.has(item.id) ? 'selected' : ''} ${item.status === 'generating' ? 'is-generating' : ''}`}>
                                            <td data-label="Select" className="checkbox-cell">
                                                <input
                                                    type="checkbox"
                                                    checked={selectedItems.has(item.id)}
                                                    onChange={() => handleToggleSelect(item.id)}
                                                    aria-label={`Select ${item.title}`}
                                                />
                                            </td>
                                            <td data-label="Title">{item.title}</td>
                                            <td data-label="Type" style={{textTransform: 'capitalize'}}>{item.type}</td>
                                            <td data-label="Status">
                                                 {item.status === 'generating' ? (
                                                    <div className="generation-in-progress">
                                                        <div className="row-progress-details">
                                                            <span className="row-status-text">{item.statusText}</span>
                                                            <div className="row-progress-bar-container">
                                                                <div className="row-progress-bar-fill" style={{ width: calculateGenerationProgress(item.statusText) }}></div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                ) : (
                                                    <span className={`status-cell status-${item.status}`}>
                                                        {item.status === 'done' && <span className="status-icon">✓</span>}
                                                        {item.status === 'error' && <span className="status-icon">✗</span>}
                                                        {item.statusText}
                                                    </span>
                                                )}
                                            </td>
                                            <td data-label="Actions" className="actions-cell">
                                                {item.status === 'generating' ? (
                                                    <button className="btn btn-small stop-generation-btn-row" onClick={() => handleStopGeneration(item.id)}>Stop</button>
                                                ) : item.generatedContent ? (
                                                    <button className="btn btn-small" onClick={() => handleOpenReview(item)}>Review & Edit</button>
                                                ) : (
                                                    <button className="btn btn-small" onClick={() => handleGenerateSingle(item)}>Generate</button>
                                                )}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                );
            default:
                return <div>Invalid step</div>;
        }
    };

    return (
        <div className="container">
            <header>
                 <h1>WP Content Optimizer Pro</h1>
                 <ProgressBar currentStep={currentStep} onStepClick={setCurrentStep} />
            </header>
            <main>
                {renderStep()}
            </main>
            {isBulkPublishModalOpen &&
                <BulkPublishModal
                    items={items.filter(item => selectedItems.has(item.id) && item.status === 'done')}
                    onClose={() => setIsBulkPublishModalOpen(false)}
                    publishItem={publishItem}
                    wpPassword={wpPassword}
                    onPublishSuccess={handlePublishSuccess}
                />
            }
            {selectedItemForReview && (
                <ReviewModal
                    item={selectedItemForReview}
                    onClose={handleCloseReview}
                    onSaveChanges={handleSaveChanges}
                    wpConfig={wpConfig}
                    wpPassword={wpPassword}
                    onPublishSuccess={handlePublishSuccess}
                    publishItem={publishItem}
                />
            )}
             <footer className="app-footer">
                <p>WP Content Optimizer Pro v7.3</p>
                <p><a href="https://github.com/your-repo" target="_blank" rel="noopener noreferrer">View on GitHub</a> | Report an Issue</p>
            </footer>
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<React.StrictMode><App /></React.StrictMode>);