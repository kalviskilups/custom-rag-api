import requests

def ask_question():
    query = input("Please enter your query (or type 'q' to quit): ")
    
    if query.lower() == 'q':
        print("Exiting the program...")
        return False

    response = requests.post(
        "http://127.0.0.1:2026/query",
        json={"query": query}
    )
    
    if response.status_code == 200:
        data = response.json()
        print("\nSources:")
        for source in data.get("sources", []):
            print(f"\nSource {source['source']}, Page {source['page']}\n")
            print(f"{source['content']}")

        print(data.get("answer", "No answer found"))
    else:
        print("Error:", response.json().get("error"))
    
    return True 

def main():
    keep_asking = True
    while keep_asking:
        keep_asking = ask_question()

if __name__ == '__main__':
    main()
