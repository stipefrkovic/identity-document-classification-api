from fastapi.testclient import TestClient

from main import app

client = TestClient(app)
DOC_DIR = "/document/"


class Test_Main:
    def test_main(self):
        response = client.get("/")
        assert response.json() == {"message": "Test"}

    def test_post_pdf_document(self):
        with open("./src/tests/files/id_card_1.pdf", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            assert response.status_code == 200
            assert response.json().get("document_type") == "id_card"

    def test_post_jpg_file(self):
        # Tests that jpg files are rejected
        with open("./src/tests/files/id.jpg", "rb") as f:
            response = client.post(DOC_DIR, files={"document": f})
            assert response.status_code == 400
            assert response.json() == {"detail": "Invalid file type. File must be pdf."}

    def test_post_without_document(self):
        response = client.post(DOC_DIR)
        assert response.status_code == 422
