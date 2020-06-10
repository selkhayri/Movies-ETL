data_dir = "../data/"

re_dollar_amount_1 = r'\$\s*\d+\.?\d*\s*milli?on'
re_dollar_amount_2 = r'\$\s*\d+\.?\d*\s*billi?on'
re_dollar_amount_3 = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'

re_date_form_1 = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
re_date_form_2 = r'\d{4}.[01]\d.[123]\d'
re_date_form_3 = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
re_date_form_4 = r'\d{4}'

