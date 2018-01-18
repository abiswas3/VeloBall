import wikipedia as w

from preprocess_for_lstm import parse_document_into_sentences

def get_search_terms(query):

    page = w.page(query)
    return page.links

def get_wiki_page(query):
    try:
        temp = w.page(query)        
        return temp.content    
    except:
        print(query, "had issues")
        return None


if __name__ == '__main__':

    list_query = ['List of structural failures and collapses',
                  'List_of_neurological_conditions_and_disorders',
                  'List_of_British_comedians']

    DOCUMENTS = {}
    for index, lq in enumerate(list_query):

        queries = get_search_terms(lq)
        
        for qNum, query in enumerate(queries):
            print(query)
            data = get_wiki_page(query)
            if data != None:
                # migt want to subsample here or do other things
                # make many documents
                DOCUMENTS[query] = {}
                x = parse_document_into_sentences(data)
                DOCUMENTS[query]['data'] = x
                DOCUMENTS[query]['label'] = index

            print(qNum)

        print()
