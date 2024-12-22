import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
import nltk

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


class TextPreprocessor:
    def __init__(
        self,
        remove_useless_lines=True,
        remove_unwanted_text=True,
        remove_references=True,
        remove_bullet_points=True,
        remove_specific_patterns=True,
        remove_short_speeches=True,
        remove_links=True,
        remove_special_characters=True,
        tokenize_words=True,
        tokenize_sentences=True,
        remove_stopwords=True,
        apply_stemming=False,
        apply_lemmatization=False,
        word_count_threshold=100,
        apply_pos_tagging=False
    ):
        """
        Initializes the TextPreprocessor with specified preprocessing options.
        """
        self.remove_useless_lines = remove_useless_lines
        self.remove_unwanted_text = remove_unwanted_text
        self.remove_references = remove_references
        self.remove_bullet_points = remove_bullet_points
        self.remove_specific_patterns = remove_specific_patterns
        self.remove_short_speeches = remove_short_speeches
        self.remove_links = remove_links
        self.remove_special_characters = remove_special_characters
        self.tokenize_words = tokenize_words
        self.tokenize_sentences = tokenize_sentences
        self.remove_stopwords = remove_stopwords
        self.apply_stemming = apply_stemming
        self.apply_lemmatization = apply_lemmatization
        self.word_count_threshold = word_count_threshold
        self.apply_pos_tagging = apply_pos_tagging
        
        # Initialize NLTK tools
        self.stop_words = set(stopwords.words('english')) if self.remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer() if self.apply_lemmatization else None
        self.stemmer = PorterStemmer() if self.apply_stemming else None
        
        # Define unwanted text patterns (can be parameterized if needed)
        self.unwanted_text = (
            'Accessible Keys for Video\n'
            '[Space Bar] toggles play/pause;\n'
            '[Right/Left Arrows] seeks the video forwards and back (5 sec );\n'
            '[Up/Down Arrows] increase/decrease volume;\n'
            '[M] toggles mute on/off;\n'
            '[F] toggles fullscreen on/off (Except IE 11);\n'
            'The [Tab] key may be used in combination with the [Enter/Return] key to navigate and activate control buttons, such as caption on/off.'
        )
        
        self.patterns_to_remove = [
            "View speech charts and figures",
            "Accessible Version",
            "Accessible version of figures",
            "Accessible version of charts"
        ]

    def preprocess(self, speech):
        """
        Preprocesses a single speech text.
        
        Parameters:
        - speech (str): The original speech text.
        
        Returns:
        - dict: A dictionary containing the original and processed text and tokens.
        """
        original_speech = speech  # Keep a copy of the original speech
        
        # Step 1: Remove useless lines at the end
        if self.remove_useless_lines:
            idx = speech.find('1. The views expressed here')
            if idx == -1:
                idx = speech.find('1. These views are my own')
            if idx != -1:
                speech = speech[:idx]
        
        # Step 2: Remove unwanted text patterns
        if self.remove_unwanted_text:
            speech = speech.replace(self.unwanted_text, '')
        
        # Step 3: Remove references and footnotes
        if self.remove_references:
            # Remove text from 'References' to 'Footnotes' to 'Return to text'
            pattern = r'(References.*?Footnotes.*?\d+\.\s.+Return to text)'
            speech = re.sub(pattern, '', speech, flags=re.DOTALL)
        
        # Step 4: Remove bullet points
        if self.remove_bullet_points:
            # Remove lines that start with a number followed by a dot
            speech = re.sub(r'\n\s*\d+\.\s+.*', '', speech)
        
        # Step 5: Remove specific patterns
        if self.remove_specific_patterns:
            combined_pattern = '|'.join([re.escape(pattern) for pattern in self.patterns_to_remove])
            speech = re.sub(combined_pattern, '', speech, flags=re.IGNORECASE)
        
        # Step 6: Remove speeches with less than word_count_threshold words
        if self.remove_short_speeches:
            word_count = len(speech.split())
            if word_count < self.word_count_threshold:
                return {
                    'original_speech': original_speech,
                    'processed_speech_text': None,
                    'word_tokens_speech': None,
                    'sent_tokens_speech': None,
                    'word_tokens_speech_wo_stopwords': None,
                    'pos_tags_speech': None
                }
        
        # Step 7: Remove references (another pattern)
        if self.remove_references:
            idx = speech.find('References')
            if idx != -1:
                speech = speech[:idx]
        
        # Step 8: Remove specific reference patterns
        if self.remove_references:
            # Remove patterns like ". 7", ". '7", etc.
            speech = re.sub(r"(?<!\d)\.\s*['\"]?\s*\d+", '.', speech)
            # Remove patterns like "as shown in figure 1"
            speech = re.sub(r'(?i)\bas\s+shown\s+in\s+figure\s+\d+["\']?', '', speech)
        
        # Step 9: Remove footnotes
        if self.remove_references:
            idx = speech.find('Footnotes')
            if idx != -1:
                speech = speech[:idx]
        
        # Step 10: Remove additional specific patterns
        if self.remove_specific_patterns:
            speech = re.sub(r'Accessible Version|Accessible version of figures|Accessible version of charts', '', speech, flags=re.IGNORECASE)
        
        # Step 11: Remove URLs
        if self.remove_links:
            speech = re.sub(r'https?://\S+|www\.\S+', '', speech)
        
        # Step 12: Remove special characters except (- / . ! ?)
        if self.remove_special_characters:
            speech = re.sub(r'[^\w\s\-\/.!?]', '', speech)
        
        # Step 13: Convert text to lowercase
        speech = speech.lower()
        
        # Initialize tokens
        word_tokens = []
        sent_tokens = []
        word_tokens_wo_stopwords = []
        pos_tags = []
        
        # Step 14: Tokenization
        if self.tokenize_words:
            word_tokens = word_tokenize(speech)
        
        if self.tokenize_sentences:
            sent_tokens = sent_tokenize(speech)
        
        # Step 15: POS Tagging using NLTK
        if self.apply_pos_tagging and self.tokenize_words:
            pos_tags_full = pos_tag(word_tokens)  # List of tuples (word, tag)
        
        # Step 16: Remove stopwords
        if self.remove_stopwords and self.tokenize_words and self.apply_pos_tagging:
            word_tokens_wo_stopwords = []
            pos_tags = []
            for word, tag in pos_tags_full:
                if word not in self.stop_words:
                    word_tokens_wo_stopwords.append(word)
                    pos_tags.append(tag)
        elif self.remove_stopwords and self.tokenize_words:
            word_tokens_wo_stopwords = word_tokens.copy()
            pos_tags = []
        
        # Step 17: Stemming
        if self.apply_stemming and self.tokenize_words:
            word_tokens_wo_stopwords = [self.stemmer.stem(word) for word in word_tokens_wo_stopwords]
        
        # Step 18: Lemmatization
        if self.apply_lemmatization and self.tokenize_words:
            word_tokens_wo_stopwords = [self.lemmatizer.lemmatize(word) for word in word_tokens_wo_stopwords]
        
        # Compile the results
        preprocessed_text = speech
        
        return {
            'original_speech': original_speech,
            'processed_speech_text': preprocessed_text,
            'word_tokens_speech': word_tokens if self.tokenize_words else None,
            'sent_tokens_speech': sent_tokens if self.tokenize_sentences else None,
            'word_tokens_speech_wo_stopwords': word_tokens_wo_stopwords if self.tokenize_words else None,
            'pos_tags_speech': pos_tags if self.apply_pos_tagging else None
        }
    

    def preprocess_dataframe(self, df, input_column):
        """
        Applies preprocessing to a specified column in a DataFrame and adds new columns with the results.
        
        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - input_column (str): The name of the column containing text data.
        
        Returns:
        - pd.DataFrame: The DataFrame with processed text and tokens.
        """
        temp = df[input_column].apply(self.preprocess)
        
        processed_df = pd.DataFrame({
            f'{input_column}_processed_text': temp.apply(lambda x: x['processed_speech_text']),
            f'{input_column}_word_tokens': temp.apply(lambda x: x['word_tokens_speech']),
            f'{input_column}_sent_tokens': temp.apply(lambda x: x['sent_tokens_speech']),
            f'{input_column}_word_tokens_wo_stopwords': temp.apply(lambda x: x['word_tokens_speech_wo_stopwords']),
            f'{input_column}_pos_tags': temp.apply(lambda x: x['pos_tags_speech']) if self.apply_pos_tagging else None
        })
        
        return processed_df