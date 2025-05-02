# Dummy-AI-AGENT

## Overview

This project allows you to build an AI agent using **Hugging Face Transformers**. The agent is capable of interacting with tools, processing input, and returning structured outputs (such as "Final Answer") based on a custom system prompt. 

In this setup:
- You provide the agent with **questions** and the agent outputs structured responses, such as tool calls or direct answers.
- The agent can be extended to interact with external APIs like a weather service or other predefined tools (e.g., `get_weather`).

## Features
- **Interactive Q&A**: The agent can answer questions based on a system prompt and generate structured responses.
- **Tool Integration**: The agent can be extended to interact with APIs or predefined actions (e.g., `get_weather`).
- **Transformers Pipeline**: Leverages the power of the Hugging Face Transformers library for seamless text generation.
- **Action-Observation Pattern**: A conversational agent that follows the `Thought-Action-Observation` structure for multi-step problem solving.

## Requirements

- Python 3.7 or higher
- Transformers library
- Hugging Face `pipeline` or `text-generation` models
- Optional: Custom tools (e.g., weather API, etc.)

## Setup

1. **Install dependencies:**

   To get started, clone the repository and install the required Python libraries:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   pip install transformers
   pip install torch
   ```

2. **Set up your Hugging Face API key:**

   If you're using Hugging Face models that require authentication, you'll need to create an account on [Hugging Face](https://huggingface.co/) and obtain an API key. 

   Then, run the following to log in (optional, if needed):
   ```bash
   huggingface-cli login
   ```

3. **Run the agent:**

   To run the agent using the Transformers `pipeline` method, simply use the script below to generate structured outputs based on your prompt.

   ```python
   from transformers import pipeline

   # Define the system prompt
   SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

   get_weather: Get the current weather in a given location

   The way you use the tools is by specifying a json blob.
   Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

   The only values that should be in the "action" field are:
   get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
   example use :
   ```
   {{
     "action": "get_weather",
     "action_input": {"location": "New York"}
   }}
   ```

   ALWAYS use the following format:

   Question: the input question you must answer
   Thought: you should always think about one action to take. Only one action at a time in this format:
   Action:
   ```
   $JSON_BLOB
   ```
   Observation: the result of the action. This Observation is unique, complete, and the source of truth.
   ... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

   You must always end your output with the following format:

   Thought: I now know the final answer
   Final Answer: the final answer to the original input question

   Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer. """

   # User query
   question = "What is the current weather in New York?"

   # Define prompt
   prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
   {SYSTEM_PROMPT}
   <|eot_id|><|start_header_id|>user<|end_header_id|>
   Question: {question}
   <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

   # Create the pipeline
   pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct")

   # Generate output
   output = pipe(prompt, max_new_tokens=200)

   # Get the generated text
   generated_text = output[0]['generated_text']

   # Slice only the assistant's reply (excluding prompt)
   assistant_reply = generated_text[len(prompt):].strip()

   # Check for "Final Answer:" and slice it cleanly
   final_answer_marker = "Final Answer:"
   if final_answer_marker in assistant_reply:
       assistant_reply = assistant_reply.split(final_answer_marker)[-1].strip()
       assistant_reply = f"{final_answer_marker} {assistant_reply}"

   print(assistant_reply)
   ```

### Example Output:

```
Final Answer: The weather in New York is sunny with low temperatures.
```

## How the Agent Works

The agent follows the **Thought-Action-Observation** pattern:
1. **Thought**: It thinks about what action to take based on the system prompt.
2. **Action**: It triggers an action (e.g., calling a weather API) based on the question.
3. **Observation**: It records the result of that action.
4. **Final Answer**: It outputs the final answer after completing the task, with a structured response.

This process can be extended to include additional tools and steps in the decision-making process.

## Customization

- **Tools**: You can customize or extend the list of tools by modifying the **system prompt**. This can include anything from weather tools to more complex integrations with external APIs.
- **Questions**: You can change the `question` variable dynamically to ask different queries.

## Contributing

Feel free to open issues or submit pull requests if you want to improve the code! Contributions are always welcome.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
