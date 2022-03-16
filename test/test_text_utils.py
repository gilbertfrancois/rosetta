import unittest
import text_utils as tu


class TestTextUtils(unittest.TestCase):

    def testCyrillic2Latin(self):
        text = "здравствуйте"
        res = tu.cyrillic2latin(text)
        self.assertTrue("zdravstvuyte" == res)

    def testCyrillicAlphabetLowerCase(self):
        text = "абвгдеёжзийклмнопстyыфхцчшщъыьэюя"
        res = tu.cyrillic2latin(text)
        self.assertTrue(len(text) == 33)
        self.assertTrue(res == "abvgdeezhziyklmnopstyyfhtschshschyeyuya")

    def testCyrillicAlphabetUpperCase(self):
        text = "АБВГДЕЁЖЗИЙКЛМНОПСТYЫФХЦЧШЩЪЫЬЭЮЯ"
        res = tu.cyrillic2latin(text)
        self.assertTrue(len(text) == 33)
        self.assertTrue(res == "ABVGDEEZhZIYKLMNOPSTYYFHTsChShSchYEYuYa")

    def testCyrillicTextWithSpecialCharacters(self):
        text = "прима-балерина. &%$#"
        res = tu.cyrillic2latin(text)
        self.assertTrue(res == "prima-balerina. &%$#")

