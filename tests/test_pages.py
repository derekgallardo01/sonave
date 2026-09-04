"""Public marketing/SEO surfaces: routes, sitemap, guides whitelist."""
import pytest
from fastapi.testclient import TestClient

import conftest


@pytest.fixture
def mod(monkeypatch, tmp_path):
    monkeypatch.setenv("SONAVE_SCORER_URL", "")
    monkeypatch.setenv("SONAVE_API_TOKEN", "t")
    monkeypatch.setenv("SONAVE_PUBLIC_DOMAIN", "usesonave.com")
    return conftest.load_module("rwapp_pages", "railway/app.py")


def test_public_pages_render(mod):
    c = TestClient(mod.app)
    for path, marker in (("/", "FAQPage"), ("/benchmarks", "97.2%"),
                         ("/guides", "Guides"), ("/privacy", "Privacy"),
                         ("/llms.txt", "Sonave"), ("/robots.txt", "Sitemap:")):
        r = c.get(path)
        assert r.status_code == 200 and marker in r.text, path
        assert "__FAVICON__" not in r.text, path


def test_guides_whitelist(mod):
    c = TestClient(mod.app)
    for slug in mod.GUIDE_SLUGS:
        r = c.get(f"/guides/{slug}")
        assert r.status_code == 200 and "usesonave.com/guides/" in r.text, slug
    assert c.get("/guides/../app").status_code in (404, 422)
    assert c.get("/guides/nope").status_code == 404


def test_sitemap_lists_all_public_pages(mod):
    body = TestClient(mod.app).get("/sitemap.xml").text
    for path in ("/benchmarks", "/guides", "/guides/deepfake-detector-accuracy"):
        assert f"https://usesonave.com{path}</loc>" in body, path
