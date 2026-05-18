import re
from newspaper import Article, ArticleException

def clean_text(text: str) -> str:
    """
    Remove excessive whitespace, control characters, and normalize spacing.
    """
    # Replace multiple spaces / newlines with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove any non-printable control characters (optional)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text.strip()

def extract_article_text(url: str) -> dict:
    """
    Extract title and clean text from a news article URL.
    Returns:
        dict with keys 'title', 'text', 'error'
        If success, error is None.
    """
    try:
        article = Article(url, language='en')  # language hint helps parsing
        article.download()
        article.parse()
        title = article.title or ''
        text = article.text or ''
        return {
            "title": clean_text(title),
            "text": clean_text(text),
            "error": None
        }
    except ArticleException as e:
        return {
            "title": "",
            "text": "",
            "error": f"Newspaper3k error: {str(e)}"
        }
    except Exception as e:
        return {
            "title": "",
            "text": "",
            "error": f"Unexpected error: {str(e)}"
        }

# Quick local test
if __name__ == "__main__":
    test_url = "https://www.thehindu.com/news/national/supreme-court-to-hear-review-pleas-against-verdict-on-teachers-eligibility-test-on-may-13/article70931580.ece"  # any real article
    result = extract_article_text(test_url)
    if result["error"]:
        print("Error:", result["error"])
    else:
        print("Title:", result["title"])
        print("Text preview:", result["text"][:600])

"""
url = "https://www.thehindu.com/news/national/supreme-court-to-hear-review-pleas-against-verdict-on-teachers-eligibility-test-on-may-13/article70931580.ece"  # a real article
article = Article(url)
article.download()
article.parse()
print(article.title)
print(article.text)"""