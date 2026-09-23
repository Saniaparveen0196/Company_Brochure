import os
import json
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify
from openai import OpenAI

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
LINKS_MODEL = "openai/gpt-oss-20b"          
BROCHURE_MODEL = "openai/gpt-oss-20b" 
MAX_PAGE_CHARS = 1000
MAX_JSON_RETRIES = 2
REQUEST_TIMEOUT = 10


DEFAULT_API_KEY = os.getenv("GROQ_API_KEY", "")

app = Flask(__name__)


def get_client(api_key=None):
    key = (api_key or "").strip() or DEFAULT_API_KEY
    if not key:
        return None
    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")



def fetch_website_links(url):
    resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("mailto:") or href.startswith("#"):
            continue
        links.add(urljoin(url, href))
    return sorted(links)


def fetch_website_contents(url):
    resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "img", "input"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else "No title found"
    text = soup.get_text(separator="\n", strip=True)
    return f"{title}\n\n{text}"


# ----------------------------------------------------------------------
# Link selection (LLM call, with JSON-schema validation + retry)
# ----------------------------------------------------------------------
link_system_prompt = """
You are provided with a list of links found on a webpage.
You are able to decide which of the links would be most relevant to include in a brochure about the company,
such as links to an About page, or a Company page, or Careers/Jobs pages.

You must respond with ONLY valid JSON, matching this exact schema, and nothing else -
no explanation, no markdown fences:

{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}

Rules:
- Every item in "links" MUST be an object with both a "type" key and a "url" key.
- "url" MUST always be a full absolute URL starting with http:// or https:// - never a relative path.
- Do not return a list of plain strings. Do not omit the "type" key.
"""


def get_links_user_prompt(url, links):
    user_prompt = f"""
Here is the list of links on the website {url} -
Please decide which of these are relevant web links for a brochure about the company,
respond with the full https URL in JSON format.
Do not include Terms of Service, Privacy, email links.

Links (some might be relative links):

"""
    user_prompt += "\n".join(links)
    return user_prompt


def _is_valid_links_payload(payload):
    if not isinstance(payload, dict) or "links" not in payload:
        return False
    items = payload["links"]
    if not isinstance(items, list):
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        if "url" not in item or not isinstance(item["url"], str):
            return False
        if not item["url"].startswith("http"):
            return False
    return True


def select_relevant_links(client, url, log):
    log(f"Selecting relevant links for {url}...")
    try:
        links = fetch_website_links(url)
    except Exception as e:
        log(f"Could not fetch links from {url}: {e}")
        return {"links": []}

    messages = [
        {"role": "system", "content": link_system_prompt},
        {"role": "user", "content": get_links_user_prompt(url, links)},
    ]

    for attempt in range(1, MAX_JSON_RETRIES + 2):
        response = client.chat.completions.create(
            model=LINKS_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None

        if parsed is not None and _is_valid_links_payload(parsed):
            log(f"Found {len(parsed['links'])} relevant links")
            return parsed

        log(f"Attempt {attempt}: malformed link JSON, retrying...")
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                "That wasn't valid JSON matching the required schema "
                '{"links": [{"type": "...", "url": "https://..."}]}. '
                "Please respond again with ONLY correctly formatted JSON."
            ),
        })

    log("Giving up after retries - continuing with no relevant links.")
    return {"links": []}


def fetch_page_and_all_relevant_links(client, url, log):
    try:
        contents = fetch_website_contents(url)[:MAX_PAGE_CHARS]
    except Exception as e:
        log(f"Could not fetch landing page {url}: {e}")
        contents = ""

    relevant_links = select_relevant_links(client, url, log)
    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
    for link in relevant_links.get("links", []):
        link_type = link.get("type", "page")
        link_url = link.get("url")
        try:
            page_text = fetch_website_contents(link_url)[:MAX_PAGE_CHARS]
        except Exception as e:
            log(f"Skipping {link_url} ({e})")
            continue
        result += f"\n\n### Link: {link_type}\n"
        result += page_text
    return result


brochure_system_prompt = """
You are an assistant that analyzes the contents of several relevant pages from a company website
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
Include details of company culture, customers and careers/jobs if you have the information.
"""


def get_brochure_user_prompt(client, company_name, url, log):
    user_prompt = f"""
You are looking at a company called: {company_name}
Here are the contents of its landing page and other relevant pages;
use this information to build a short brochure of the company in markdown without code blocks.\n\n
"""
    user_prompt += fetch_page_and_all_relevant_links(client, url, log)
    return user_prompt[:8_000]


def generate_brochure(company_name, url, api_key):
    log_lines = []

    def log(msg):
        log_lines.append(msg)

    client = get_client(api_key)
    if client is None:
        return {
            "ok": False,
            "error": "No Groq API key available. Set GROQ_API_KEY as a Railway variable, or pass your own key.",
            "log": log_lines,
        }

    try:
        user_prompt = get_brochure_user_prompt(client, company_name, url, log)
        response = client.chat.completions.create(
            model=BROCHURE_MODEL,
            messages=[
                {"role": "system", "content": brochure_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        brochure = response.choices[0].message.content
        return {"ok": True, "brochure": brochure, "log": log_lines}
    except Exception as e:
        return {"ok": False, "error": str(e), "log": log_lines}


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True) or {}
    company_name = (data.get("company_name") or "").strip()
    url = (data.get("url") or "").strip()
    api_key = (data.get("api_key") or "").strip()

    if not company_name or not url:
        return jsonify({"ok": False, "error": "Company name and URL are required."}), 400

    result = generate_brochure(company_name, url, api_key)
    status = 200 if result["ok"] else 500
    return jsonify(result), status


@app.route("/healthz")
def healthz():
    return "ok"


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)