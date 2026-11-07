import unittest

from car_manual_rag.cleaning import (
    classify_content_role,
    clean_extracted_text,
    is_text_retrieval_eligible,
)


class CleanExtractedTextTests(unittest.TestCase):
    def test_none_stays_none(self):
        self.assertIsNone(clean_extracted_text(None))

    def test_empty_whitespace_becomes_none(self):
        self.assertIsNone(clean_extracted_text(" \n\t "))

    def test_decodes_html_entities(self):
        self.assertEqual(
            clean_extracted_text("Vista&#x20;Zoom &amp; sensores"),
            "Vista Zoom & sensores",
        )

    def test_repairs_layout_hyphenation(self):
        self.assertEqual(
            clean_extracted_text("sele-\ncionar o contato"),
            "selecionar o contato",
        )

    def test_preserves_inline_hyphens(self):
        self.assertEqual(
            clean_extracted_text("assistência 24-horas"),
            "assistência 24-horas",
        )

    def test_preserves_portuguese_clitic_hyphen(self):
        self.assertEqual(
            clean_extracted_text("certifique-\nse antes de dirigir"),
            "certifique-se antes de dirigir",
        )
        self.assertEqual(
            clean_extracted_text("enxergá-\nlos durante a noite"),
            "enxergá-los durante a noite",
        )

    def test_repairs_duplicate_hyphen_across_line_break(self):
        self.assertEqual(
            clean_extracted_text("libere-\n-o imediatamente"),
            "libere-o imediatamente",
        )

    def test_repairs_slash_line_break(self):
        self.assertEqual(
            clean_extracted_text("MENU/\nVeículo/\nAjuda"),
            "MENU/Veículo/Ajuda",
        )

    def test_preserves_table_markup(self):
        self.assertEqual(
            clean_extracted_text("<td>A&#x20;B</td>\n<td>C</td>"),
            "<td>A B</td> <td>C</td>",
        )


class ContentRoleTests(unittest.TestCase):
    def test_roles(self):
        self.assertEqual(classify_content_role("page_header"), "page_furniture")
        self.assertEqual(classify_content_role("section_header"), "structure")
        self.assertEqual(classify_content_role("text"), "body")
        self.assertEqual(classify_content_role("figure"), "visual")
        self.assertEqual(classify_content_role("unknown"), "other")

    def test_retrieval_eligibility(self):
        self.assertTrue(is_text_retrieval_eligible("text", "useful text"))
        self.assertTrue(is_text_retrieval_eligible("section_header", "Safety"))
        self.assertFalse(is_text_retrieval_eligible("page_header", "Maintenance"))
        self.assertFalse(is_text_retrieval_eligible("figure", "diagram labels"))
        self.assertFalse(is_text_retrieval_eligible("text", None))


if __name__ == "__main__":
    unittest.main()
