import unittest

from app.rag_core import (
    build_context,
    build_user_prompt,
    search_result_rows,
    validate_vehicle,
)


SAMPLE_CHUNK = {
    "chunk_id": "chunk-1",
    "chunk_text": "Pare o veículo em segurança.",
    "document_id": "peugeot_208_2023_pt_br",
    "manufacturer": "Peugeot",
    "model": "208",
    "model_year": 2023,
    "section_title": "Temperatura do motor",
    "pdf_page_numbers": [23],
}


class VehicleTests(unittest.TestCase):
    def test_normalizes_vehicle(self):
        self.assertEqual(validate_vehicle(" Peugeot "), "peugeot")

    def test_rejects_unknown_vehicle(self):
        with self.assertRaises(ValueError):
            validate_vehicle("other")


class RetrievalResponseTests(unittest.TestCase):
    def test_converts_manifest_rows(self):
        response = {
            "manifest": {"columns": [{"name": "id"}, {"name": "text"}]},
            "result": {"data_array": [["1", "manual text"]]},
        }
        self.assertEqual(
            search_result_rows(response),
            [{"id": "1", "text": "manual text"}],
        )


class PromptTests(unittest.TestCase):
    def test_context_contains_citation_metadata(self):
        context = build_context([SAMPLE_CHUNK])
        self.assertIn("[Source 1]", context)
        self.assertIn("PDF pages: [23]", context)
        self.assertIn("Peugeot 208 2023", context)

    def test_prompt_names_selected_vehicle_and_question(self):
        prompt = build_user_prompt(
            "O que devo fazer?",
            "peugeot",
            [SAMPLE_CHUNK],
        )
        self.assertIn("Selected vehicle: Peugeot 208 2023", prompt)
        self.assertIn("O que devo fazer?", prompt)


if __name__ == "__main__":
    unittest.main()
