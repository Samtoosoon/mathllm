import re
from deep_translator import GoogleTranslator
from typing import List, Dict

class MathTranslator:
    def __init__(self, source_lang: str = 'en', target_lang: str = 'hi'):
        self.translator = GoogleTranslator(source=source_lang, target=target_lang)
        # Regex to capture content inside single or double dollar signs to prevent translation
        self.math_regex = re.compile(r'(\$\$[^\$]+\$\$|\$[^\$]+\$)')
    
    def translate_preserve_math(self, text: str) -> str:
        """
        Translates text from source_lang to target_lang while keeping math formulas intact.
        """
        if not text or not isinstance(text, str):
            return ""

        # Extract math blocks
        math_blocks = self.math_regex.findall(text)
        
        # Replace math blocks with placeholders
        placeholder_template = " __MATH_{}__ "
        temp_text = text
        for i, block in enumerate(math_blocks):
            temp_text = temp_text.replace(block, placeholder_template.format(i))
        
        try:
            # Translate the modified text
            translated_text = self.translator.translate(temp_text)
            
            # Put math blocks back
            for i, block in enumerate(math_blocks):
                translated_text = translated_text.replace(placeholder_template.format(i).strip(), f" {block} ")
                
            return translated_text.strip()
        except Exception as e:
            print(f"Translation error: {e}")
            return text # Fallback to original text on failure

    def process_dataset(self, data: List[Dict], limit: int = 0) -> List[Dict]:
        """
        Translates a list of dictionaries containing 'query' and 'response' fields.
        """
        translated_data = []
        count = len(data) if limit == 0 else min(len(data), limit)
        
        print(f"Translating {count} records...")
        for i in range(count):
            item = data[i]
            translated_item = {
                'instruction': "नीचे दिए गए गणित के प्रश्न को हल करें:" if 'instruction' not in item else self.translate_preserve_math(item.get('instruction', '')),
                'input': self.translate_preserve_math(item.get('query', '')),
                'output': self.translate_preserve_math(item.get('response', ''))
            }
            translated_data.append(translated_item)
            
            if (i + 1) % 100 == 0:
                print(f"Translated {i + 1} / {count} records")
                
        return translated_data
