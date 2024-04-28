from tflite_model_maker import searcher
from tflite_support.task import text

def pretty_print(statement):
    print("*******************************************", end='/n/n')
    print(statement, end='/n/n')
    print("*******************************************", end='/n/n')


pretty_print("Executing test against model....")

# Initializes a TextSearcher object.
searcher = text.TextSearcher.create_from_file("searcher.tflite")

# Searches the input query.
results = searcher.search("The Airline Quality Rankings Report looks at the 14 largest U.S. airlines.")
pretty_print(results)