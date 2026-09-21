from web import create_app


def test_index_and_jobs():
    app = create_app()
    with app.test_client() as c:
        assert c.get("/").status_code == 200
        assert c.get("/jobs").status_code == 200
        assert c.get("/audits").status_code == 200
        assert c.get("/settings").status_code == 200


def test_models_masked():
    app = create_app()
    with app.test_client() as c:
        r = c.get("/api/models")
        assert r.status_code == 200
        # api_key should be masked if present
        j = r.get_json()
        if j.get("api_key"):
            assert "***" in j["api_key"] or j.get("api_key_masked")


def test_template_preview_404():
    app = create_app()
    with app.test_client() as c:
        r = c.get("/api/template-preview?template_path=does/not/exist.png")
        assert r.status_code in (400, 404, 422)
