from bow import BagOfWordView

class CapitalizedBoWView(BagOfWordView):
    """
    
    Return:
    -----------
    
    The concatenated string of:

    1. the list of *capitalized* words 
    2. the list of non-capitalized words 

    >>> from data_record import DataRecord
    >>> ext = CapitalizedBoWView()
    >>> dr = DataRecord(['I', 'love', 'NBA'])
    >>> ext.extract(dr)
    ('I NBA', 'love')
    """
    name = "capitalized-bow"

    def extract(self, data_record):
        """
        Param:
        ------
        data_record: DataRecord object        
        """
        
        view1_tokens = [t for t in data_record.tokens if t[0].isalpha() and t[0].upper() == t[0]] # Capitalized words
        view2_tokens = [t for t in data_record.tokens if not t[0].isalpha() or t[0].upper() != t[0]]
        
        view1 = ' '.join(view1_tokens)
        view2 = ' '.join(view2_tokens)

        return (view1, view2)
