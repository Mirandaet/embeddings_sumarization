import sys
import json
from interface import query_vdb, get_summary, modify

def main():
    if len(sys.argv) < 3:
        print("Usage: python backend.py command [arguments]")
        sys.exit(1)

    command = sys.argv[1]
    arguments = sys.argv[2:]

    if command == 'query':
        question = arguments[0]
        if len(arguments) > 2:
            code_directory = arguments[1]
            create_new_embedding = arguments[2]
            result = query_vdb(question, code_directory, create_new_embedding)
        elif len(arguments) > 1:
            code_directory = arguments[1]
            result = query_vdb(question, code_directory)
        else:
            result = query_vdb(question)
        print(result)
    elif command == 'summary':
        result = get_summary(arguments[0])
        if type(result) == dict:
            for filename, summary in result.items():
                print(f"{filename}: {summary} \n")
        else:
            print (result)
          # assuming the file path is the first argument
    elif command == 'modify':
        result = modify(arguments[0], arguments[1])
    else:
        result = "Unknown command"
        print(result)

    # Output the result

if __name__ == "__main__":
    main()
