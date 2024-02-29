import os
from dotenv import load_dotenv
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.indices.vector_store import VectorStoreIndex

code_string = '''
def fun1():
    a = 1
    b = 2
    result = 0

    if a > b:
        result = a + b
    else:
        if a < b:
            result = b - a

    for i in range(10):
        print(f"i = {i}")

    return result

# Example usage:
result_value = fun1()
print("Result:", result_value)
'''

def main(url:str) -> None:
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])
    index = VectorStoreIndex.from_documents(documents=documents)
    query_engine = index.as_query_engine()
    response = query_engine.query("Give the control flow diagram for the code snippet: " + code_string)
    print(response)


if __name__ == '__main__':
    load_dotenv()
    main(url='https://arxiv.org/pdf/2306.00757.pdf')