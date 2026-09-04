import ollama  # this connects to the local ai model running on this machine

print("Type 'exit' to quit.")

# this list is our "memory" — without it, the model forgets everything after each message
# every question and every answer gets added here, and we send the whole thing each time
conversation = []

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # add the user's message to the memory before sending anything
    conversation.append({"role": "user", "content": user_input})

    # send the entire conversation so far, not just this one message
    # this is what lets the model remember earlier questions
    response = ollama.chat(
        model="llama3.1",
        messages=conversation
    )

    reply = response["message"]["content"]
    print("Model:", reply)

    # add the model's own reply to the memory too, so next turn it remembers what it said
    conversation.append({"role": "assistant", "content": reply})
