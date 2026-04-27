import re

class QueryPreprocessor:
    """Component 3: Query Preprocessor
    Cleans and normalizes the user query before further processing.
    """
    def preprocess(self, query: str) -> str:
        # 1. Lowercase
        query = query.lower()
        # 2. Remove special characters (keep alphanumerics and spaces)
        query = re.sub(r'[^a-zA-Z0-9\s\?]', '', query)
        # 3. Strip extra whitespace
        query = ' '.join(query.split())
        return query
