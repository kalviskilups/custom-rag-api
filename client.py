import requests

def main():
    query = input("Please enter your query: ")
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

if __name__ == '__main__':
    main()
