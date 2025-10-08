
import { GeneratedContent } from './index';

export type WpConfig = {
    url: string;
    username: string;
};

// =================================================================
// 💎 HIGH-QUALITY SCHEMA.ORG MARKUP GENERATOR
// =================================================================
// This module creates SEO-optimized JSON-LD schema markup.
// It follows Google's latest guidelines to improve search visibility,
// enhance SERP rankings, and increase eligibility for rich snippets.
// =================================================================


// --- PLACEHOLDERS: CUSTOMIZE THESE VALUES ---
// For best results, replace these placeholders with your actual company and author details.
const ORGANIZATION_NAME = "Your Company Name";
const DEFAULT_AUTHOR_NAME = "Expert Author";
// --- END PLACEHOLDERS ---


/**
 * Creates a 'Person' schema object.
 * @param authorName The name of the article's author.
 * @param authorUrl Optional URL to an author page or social profile.
 * @returns A Person schema object.
 */
function createPersonSchema(authorName: string, authorUrl?: string) {
    return {
        "@type": "Person",
        "name": authorName,
        "url": authorUrl || undefined,
    };
}

/**
 * Creates an 'Organization' schema object, used for the publisher property.
 * @param orgName The name of the organization.
 * @param orgUrl The homepage URL of the organization.
 * @param logoUrl A direct URL to the organization's logo image.
 * @returns An Organization schema object.
 */
function createOrganizationSchema(orgName: string, orgUrl: string, logoUrl: string) {
    return {
        "@type": "Organization",
        "name": orgName,
        "url": orgUrl,
        "logo": {
            "@type": "ImageObject",
            "url": logoUrl,
        },
    };
}

/**
 * Creates the core 'Article' schema.
 * @param content The fully generated content object.
 * @param wpConfig The WordPress configuration containing the site URL.
 * @param orgSchema The generated Organization schema for the publisher.
 * @param personSchema The generated Person schema for the author.
 * @returns An Article schema object.
 */
function createArticleSchema(content: GeneratedContent, wpConfig: WpConfig, orgSchema: object, personSchema: object) {
    const today = new Date().toISOString();
    return {
        "@type": "Article",
        "headline": content.title,
        "description": content.metaDescription,
        "image": content.imageDetails.map(img => img.generatedImageSrc).filter(Boolean),
        "datePublished": today,
        "dateModified": today,
        "author": personSchema,
        "publisher": orgSchema,
        "mainEntityOfPage": {
            "@type": "WebPage",
            "@id": `${wpConfig.url.replace(/\/+$/, '')}/${content.slug}`,
        },
    };
}

/**
 * Creates 'FAQPage' schema from structured data.
 * This is the preferred method as it's more reliable than HTML parsing.
 * @param faqData An array of question/answer objects.
 * @returns An FAQPage schema object, or null if no valid FAQs are provided.
 */
function createFaqSchema(faqData: { question: string, answer: string }[]) {
    if (!faqData || faqData.length === 0) {
        return null;
    }
    
    const mainEntity = faqData
        .filter(faq => faq.question && faq.answer)
        .map(faq => ({
            "@type": "Question",
            "name": faq.question,
            "acceptedAnswer": {
                "@type": "Answer",
                "text": faq.answer,
            },
        }));

    if (mainEntity.length === 0) return null;

    return {
        "@type": "FAQPage",
        "mainEntity": mainEntity,
    };
}

/**
 * [Fallback] Creates 'FAQPage' schema by parsing questions and answers from the final HTML content.
 * @param content The fully generated content object.
 * @returns An FAQPage schema object, or null if no valid FAQs are found.
 */
function createFaqSchemaFromHtml(content: GeneratedContent) {
    const mainEntity = [];
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = content.content;
    
    const h3s = tempDiv.querySelectorAll('h3');
    
    for (const h3 of h3s) {
        if (h3.textContent?.trim().endsWith('?')) {
            let nextElement = h3.nextElementSibling;
            let answerText = '';
            
            while (nextElement && !['H2', 'H3', 'H4'].includes(nextElement.tagName)) {
                if (nextElement.tagName === 'P') {
                    answerText += nextElement.textContent + ' ';
                }
                nextElement = nextElement.nextElementSibling;
            }
            
            if (answerText) {
                mainEntity.push({
                    "@type": "Question",
                    "name": h3.textContent.trim(),
                    "acceptedAnswer": {
                        "@type": "Answer",
                        "text": answerText.trim(),
                    },
                });
            }
        }
    }

    if (mainEntity.length === 0) return null;

    return {
        "@type": "FAQPage",
        "mainEntity": mainEntity,
    };
}


/**
 * Creates 'VideoObject' schemas for all embedded YouTube videos in the content.
 * @param content The fully generated content object.
 * @returns An array of VideoObject schemas, or null if no videos are found.
 */
function createVideoObjectSchemas(content: GeneratedContent) {
    const schemas = [];
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = content.content;
    const iframes = tempDiv.querySelectorAll('iframe[src*="youtube.com/embed/"]');
    
    iframes.forEach(iframe => {
        const videoIdMatch = (iframe as HTMLIFrameElement).src.match(/embed\/([^?&]+)/);
        if (videoIdMatch) {
            schemas.push({
                "@type": "VideoObject",
                "name": (iframe as HTMLIFrameElement).title || content.title,
                "description": content.metaDescription,
                "thumbnailUrl": `https://i.ytimg.com/vi/${videoIdMatch[1]}/maxresdefault.jpg`,
                "uploadDate": new Date().toISOString(), // The actual upload date is unknown, so we use the publish date.
                "embedUrl": (iframe as HTMLIFrameElement).src,
            });
        }
    });

    return schemas.length > 0 ? schemas : null;
}

/**
 * The main exported function. It assembles all relevant schema types into a single
 * '@graph' object, which is the recommended way to include multiple schemas on a page.
 * @param content The complete generated content object after all stages.
 * @param wpConfig The WordPress configuration with site URL.
 * @param faqData Optional structured FAQ data for more reliable schema generation.
 * @returns A complete schema.org JSON-LD object.
 */
export function generateFullSchema(
    content: GeneratedContent,
    wpConfig: WpConfig,
    faqData?: { question: string, answer: string }[]
): object {
    const schemas = [];
    
    const orgUrl = wpConfig.url ? wpConfig.url.replace(/\/+$/, '') : 'https://example.com';
    // A reasonable default for the logo URL. Can be customized above.
    const logoUrl = `${orgUrl}/wp-content/themes/your-theme/logo.png`;

    const organizationSchema = createOrganizationSchema(ORGANIZATION_NAME, orgUrl, logoUrl);
    const personSchema = createPersonSchema(DEFAULT_AUTHOR_NAME);

    // 1. Add Article Schema (always present)
    const articleSchema = createArticleSchema(content, wpConfig, organizationSchema, personSchema);
    schemas.push(articleSchema);

    // 2. Add FAQ Schema (if applicable)
    // SOTA Improvement: Use structured data if available, otherwise fall back to HTML parsing.
    const faqSchema = faqData ? createFaqSchema(faqData) : createFaqSchemaFromHtml(content);
    if (faqSchema) schemas.push(faqSchema);

    // 3. Add Video Schemas (if applicable)
    const videoSchemas = createVideoObjectSchemas(content);
    if (videoSchemas) schemas.push(...videoSchemas);

    // Combine all schemas into a single graph for Google
    return {
      "@context": "https://schema.org",
      "@graph": schemas,
    };
}

/**
 * Wraps the generated schema object in a `<script>` tag for embedding in HTML.
 * @param schemaObject The final JSON-LD object from generateFullSchema.
 * @returns A string containing the full schema script tag.
 */
export function generateSchemaMarkup(schemaObject: object): string {
    if (!schemaObject || !Object.prototype.hasOwnProperty.call(schemaObject, '@graph') || (schemaObject as any)['@graph'].length === 0) {
        return '';
    }
    const schemaScript = `<script type="application/ld+json">\n${JSON.stringify(schemaObject, null, 2)}\n</script>`;
    
    // ════════════════════════════════════════════════════════════════════════════════
    // CRITICAL FIX FOR VISIBLE SCHEMA MARKUP
    // ════════════════════════════════════════════════════════════════════════════════
    // The WordPress REST API's 'content' field is aggressively sanitized to prevent
    // security vulnerabilities like Cross-Site Scripting (XSS). This process correctly
    // strips raw <script> tags, which was causing the JSON-LD content to be rendered
    // as visible plain text at the end of the post.
    //
    // The ONLY standard, reliable method to insert a raw script block via the REST API
    // is to wrap it in a Gutenberg "Custom HTML" block. The `<!-- wp:html -->` comments
    // are instructions for the block editor, telling it to preserve the enclosed content
    // exactly as-is, without filtering it. This ensures the <script> tag is correctly
    // embedded in the page's HTML and remains invisible to the reader, as intended.
    //
    // Previous layout distortion issues were caused by other malformed HTML in the main
    // content, which have since been resolved by the `sanitizeHtmlResponse` function.
    // This wrapper is now safe to use and is the definitive solution to this problem.
    // ════════════════════════════════════════════════════════════════════════════════
    return `\n\n<!-- wp:html -->\n${schemaScript}\n<!-- /wp:html -->\n\n`;
}
