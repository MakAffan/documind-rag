print("TEST GENERATOR STARTED")

from app.rag.generator import generate_answer

def main():
    context = "Python is a programming language created by Guido van Rossum."
    question = "Who created Python?"

    answer = generate_answer(context, question)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
