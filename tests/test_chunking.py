import unittest

from car_manual_rag.chunking import build_section_aware_chunks, split_text


class SplitTextTests(unittest.TestCase):
    def test_splits_long_text_with_limit(self):
        text = " ".join(f"word{i}" for i in range(200))
        segments = split_text(text, max_chars=200, overlap_chars=40)
        self.assertGreater(len(segments), 1)
        self.assertTrue(all(len(segment) <= 200 for segment in segments))


class SectionAwareChunkTests(unittest.TestCase):
    def test_preserves_context_and_page_metadata(self):
        elements = [
            {
                "element_id": 1,
                "element_type": "title",
                "cleaned_content": "Owner Manual",
                "pdf_page_numbers": [1],
            },
            {
                "element_id": 2,
                "element_type": "section_header",
                "cleaned_content": "Safety",
                "pdf_page_numbers": [10],
            },
            {
                "element_id": 3,
                "element_type": "text",
                "cleaned_content": "Always wear the seat belt.",
                "pdf_page_numbers": [10],
            },
            {
                "element_id": 4,
                "element_type": "text",
                "cleaned_content": "Check every passenger.",
                "pdf_page_numbers": [11],
            },
        ]

        chunks = build_section_aware_chunks("car", elements, max_chars=300)

        self.assertEqual(len(chunks), 1)
        self.assertIn("Document title: Owner Manual", chunks[0]["chunk_text"])
        self.assertIn("Section: Safety", chunks[0]["chunk_text"])
        self.assertEqual(chunks[0]["pdf_page_numbers"], [10, 11])
        self.assertEqual(chunks[0]["element_ids"], [3, 4])

    def test_section_change_creates_new_chunk(self):
        elements = [
            {
                "element_id": 1,
                "element_type": "section_header",
                "cleaned_content": "Section A",
                "pdf_page_numbers": [1],
            },
            {
                "element_id": 2,
                "element_type": "text",
                "cleaned_content": "Content A",
                "pdf_page_numbers": [1],
            },
            {
                "element_id": 3,
                "element_type": "section_header",
                "cleaned_content": "Section B",
                "pdf_page_numbers": [2],
            },
            {
                "element_id": 4,
                "element_type": "text",
                "cleaned_content": "Content B",
                "pdf_page_numbers": [2],
            },
        ]

        chunks = build_section_aware_chunks("car", elements, max_chars=300)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["section_title"], "Section A")
        self.assertEqual(chunks[1]["section_title"], "Section B")

    def test_table_markup_is_removed_from_chunk_text(self):
        elements = [
            {
                "element_id": 1,
                "element_type": "table",
                "cleaned_content": "<table><tr><td>Oil</td><td>5W-30</td></tr></table>",
                "pdf_page_numbers": [20],
            }
        ]

        chunks = build_section_aware_chunks("car", elements, max_chars=300)

        self.assertEqual(chunks[0]["chunk_text"], "Oil 5W-30")


if __name__ == "__main__":
    unittest.main()
