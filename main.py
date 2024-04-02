import os
from dotenv import load_dotenv
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.indices.vector_store import VectorStoreIndex

from CFGImageGenerator import CFGImageGenerator

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
    response = query_engine.query("Give me the dot string response of the code snippet. "
                                  "I want to draw the control flow diagram for it. "
                                  "I will directly feed it to the graphviz to generate png of control flow diagram"
                                  "Make sure you don't add any thing other than the connection strings" + code_string)


    print(response)

    cfg = CFGImageGenerator()
    cfg.generate_image(response.__str__())


if __name__ == '__main__':
    load_dotenv()
    main(url='https://arxiv.org/pdf/2306.00757.pdf')