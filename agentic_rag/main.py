from graph.graph import app

if __name__ == '__main__':
    print("Advanced RAG")
    print(app.invoke(input={"question": "what is agent memory?"}))