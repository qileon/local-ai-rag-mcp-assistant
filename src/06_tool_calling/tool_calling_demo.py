import ollama  # this connects to the local ai model

def add_numbers(a, b):
    return a + b

tools = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers together and return the sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number"},
                    "b": {"type": "number", "description": "The second number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]

# this system message is new — it explicitly tells the model that tools are optional,
# not mandatory, so it doesn't assume every question must go through one
system_message = {
    "role": "system",
    "content": (
        "You have access to tools, but they are optional. Only call a tool if the "
        "question itself actually provides the values that tool's parameters need. "
        "Never invent, guess, or make up parameter values that are not stated in the "
        "question. If a tool's required information is not present in the question, "
        "do not call that tool at all, just answer directly using your own knowledge. "
        "Do not refuse to answer just because no tool applies."
    )
}

question = input("Question: ")

response = ollama.chat(
    model="llama3.1",
    messages=[system_message, {"role": "user", "content": question}],
    tools=tools
)

message = response["message"]

if message.get("tool_calls"):
    for call in message["tool_calls"]:
        name = call["function"]["name"]
        args = call["function"]["arguments"]

        print(f"Model wants to call: {name}({args})")

        if name == "add_numbers":
            result = add_numbers(args["a"], args["b"])

        follow_up = ollama.chat(
            model="llama3.1",
            messages=[
                system_message,
                {"role": "user", "content": question},
                message,
                {"role": "tool", "content": str(result)}
            ]
        )

        print("Answer:", follow_up["message"]["content"])
else:
    print("Answer:", message["content"])