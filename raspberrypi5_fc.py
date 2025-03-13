# Function calling code for my raspberry pi 5
# Install ollama
# Install yfinance

# I think model is too big. No enough RAM on raspberry pi 5. Quantization using bitsandbytes requires CUDA, which pi does not have.
# Bitsandbytes currently has a WIP version for multiplatform. But WIP, I won't want to use
# So i'm reverting to ollama. Let's see if it works better
# sudo apt update && sudo apt upgrade
# sudo ufw allow 11434/tcp
# curl -fsSL https://ollama.com/install.sh | sh
# ollama pull llama3.2:3b
#
# For NVIDIA GPU, Modify ollama.service to add env variable to improve token generation speed
# sudo systemctl edit ollama.service
# The above will open an editor. Add the following:
# Environment="OLLAMA_FLASH_ATTENTION=1"
# Reload the system daemon and restart ollama
# sudo systemctl daemon-reload
# sudo systemctl restart ollama

import time
import requests
import re
import json
import yfinance as yf
import subprocess
import socket


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def start_ollama_server():
    if not is_port_in_use(11434):
        try:
            subprocess.Popen(['ollama', 'serve'])
            time.sleep(5)  # Wait for server to start
            print("Ollama server started")
        except Exception as e:
            print(f"Error starting Ollama server: {e}")
    else:
        print("Ollama server is already running")


def query_ollama(prompt, model):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=data)
    return response.json()['response']


def get_stock_price(symbol):
    stock = yf.Ticker(symbol)
    if 'open' in stock.info:
        current_price = stock.info['open']
        s = f"Open price for {symbol} is " + str(current_price)
    else:
        current_price = None
        s = f"Could not retrieve the open price for {symbol}. Key 'open' not found."
    return s


def get_weather(api_key="d2f1bddd18b7df4ded8003d6132f6b80", city="Singapore"):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city}&appid={api_key}&units=metric"
    response = requests.get(complete_url)
    weather_data = response.json()
    if weather_data["cod"] != "404":
        main = weather_data["main"]
        weather = weather_data["weather"][0]
        temperature = main["temp"]
        pressure = main["pressure"]
        humidity = main["humidity"]
        weather_description = weather["description"]
        s = f"The weather in {city} is as follows:\nTemperature: {temperature}°C\nPressure: {pressure} hPa\nHumidity: {humidity}%\nDescription: {weather_description.capitalize()}"
    else:
        s = "City not found. Please check the city name."
    return s


def make_coffee(types_of_coffee='long black', milk='normal', sugar='normal', strength='normal'):
    display = f"""Making a cup of {types_of_coffee} with the following options: Milk = {milk}, Sugar = {sugar} Strength = {strength} """
    return display


def cook_burger(cook="well done"):
    display = f"Cooking a beef burger that is {cook}"
    return display


def cook_fries(type_of_fries="straight"):
    display = f"Cooking {type_of_fries} fries"
    return display


def cook_prawn_noodles(prawn="with prawn", sotong="with sotong"):
    display = f"""Cooking fried prawn noodles with the following options: Prawn = {prawn}, Sotong = {sotong} """
    return display


def extract_function_from_xml(xml_string):
    json_match = re.search(r'{.*}', xml_string)
    if json_match:
        json_str = json_match.group(0)
        data = json.loads(json_str)
        tool_name = data.get('tool_name')
        tool_arguments = data.get('tool_arguments', {})
        arguments = ', '.join([f"{key}='{value}'" for key, value in tool_arguments.items()])
        return f"{tool_name}({arguments})" if arguments else f"{tool_name}()"
    return None


def create_system_prompt(tools_list):
    system_prompt_format = """You are a function calling AI model. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into function. The user may use the terms function calling or tool use interchangeably. Here are the available functions: <tools>{}</tools> For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags in the format: <tool_call>{{"tool_name": "<function-name>", "tool_arguments": <args-dict>}}</tool_call>"""
    tools_str = "\n".join([f"<tool>{tool}</tool>" for tool in tools_list])
    system_prompt = system_prompt_format.format(tools_str)
    return system_prompt


if __name__ == "__main__":
    start_ollama_server()
    content = input("What is your order? ")

    tools_list = [
        {
            "name": "cook_burger",
            "description": "beef burger",
            "parameters": {
                "cook": {
                    "description": "Can be well done, medium or rare",
                    "type": "str",
                    "default": "well done"
                }
            }
        },
        {
            "name": "cook_fries",
            "description": "Potatoes fries",
            "parameters": {
                "type_of_fries": {
                    "description": "Can be straight or curly",
                    "type": "str",
                    "default": "straight"
                }
            }
        },
        {
            "name": "cook_prawn_noodles",
            "description": "Fried prawn noodles",
            "parameters": {
                "prawn": {
                    "description": "Options for prawn. Can be with or without prawn.",
                    "type": "str",
                    "default": "with prawn"
                },
                "sotong": {
                    "description": "Options for sotong. Can be with or without sotong.",
                    "type": "str",
                    "default": "with sotong"
                }
            }
        },
        {
            "name": "make_coffee",
            "description": "Customer orders coffee.",
            "parameters": {
                "types_of_coffee": {
                    "description": "The type of coffee. Examples are latte, americano, cappuccino.",
                    "type": "str",
                    "default": "long black"
                },
                "milk": {
                    "description": "Options for milk with the coffee. Can be 'normal', 'no', 'more', 'less'.",
                    "type": "str",
                    "default": "normal"
                },
                "sugar": {
                    "description": "Options for sugar with the coffee. Can be 'normal', 'no', 'more', 'less'.",
                    "type": "str",
                    "default": "normal"
                },
                "strength": {
                    "description": "Options for coffee strength. Can be 'normal', 'strong', 'weak'.",
                    "type": "str",
                    "default": "normal"
                }
            }
        },
        {
            "name": "get_stock_price",
            "description": "Retrieves the current stock price given a stock symbol.",
            "parameters": {
                "symbol": {
                    "description": "The stock symbol for which the price is what we wanted. Example of stock symbol is HPQ.",
                    "type": "str",
                    "default": ""
                }
            }
        },
        {
            "name": "get_weather",
            "description": "A function that retrieves the current weather for a given city.",
            "parameters": {
                "city": {
                    "description": "The city which we want to know the weather (e.g., 'New York' or 'Singapore').",
                    "type": "str",
                    "default": "Tokyo"
                }
            }
        }
    ]

    system_prompt = create_system_prompt(tools_list)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": {content}}
    ]

    prompt = f"System: {system_prompt}\nUser: {messages[1]['content']}"
    print(prompt)

    model_name = "llama3.2"
    time0 = time.time()
    output_text = query_ollama(prompt, model_name)
    time1 = time.time()
    print(f"Time Taken: {time1 - time0}s\n")
    print(output_text)

    function = extract_function_from_xml(output_text)
    print(f"Function to call is {function}")
    result = "answer = " + function
    exec(result)
    print(answer)


''' ### Code using huggingface. Commented out because too large for raspberry pi 5. Use ollama instead. Code for ollama is above
# Function calling code for my raspberry pi 5
# Install transformers
# Install torch from pytorch.org
# Install yfinance


import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import requests
import re
import json
import yfinance as yf
from huggingface_hub import login


def get_stock_price(symbol):
    # Fetch the stock data
    stock = yf.Ticker(symbol)

    # Check if 'open' exists in the stock info dictionary
    if 'open' in stock.info:
        current_price = stock.info['open']
        s = f"Open price for {symbol} is " + str(current_price)
    else:
        # Handle the case where the key does not exist
        current_price = None
        s = f"Could not retrieve the open price for {symbol}. Key 'open' not found."
    return s
# end of function


def get_weather(api_key="d2f1bddd18b7df4ded8003d6132f6b80", city="Singapore"):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    # Construct the full API URL
    complete_url = f"{base_url}q={city}&appid={api_key}&units=metric"

    # Send a GET request to the API
    response = requests.get(complete_url)

    # Parse the response in JSON format
    weather_data = response.json()

    # Check if the response was successful
    if weather_data["cod"] != "404":
        main = weather_data["main"]
        weather = weather_data["weather"][0]

        # Extract relevant information
        temperature = main["temp"]
        pressure = main["pressure"]
        humidity = main["humidity"]
        weather_description = weather["description"]

        # Display the results
        s = f"The weather in {city} is as follows:\nTemperature: {temperature}°C\nPressure: {pressure} hPa\nHumidity: {humidity}%\nDescription: {weather_description.capitalize()}"
    else:
        s = "City not found. Please check the city name."
    return s
# end of function


def make_coffee(types_of_coffee='long black', milk='normal', sugar='normal', strength='normal'):
    display = f"""Making a cup of {types_of_coffee} with the following options:
Milk = {milk},
Sugar = {sugar}
Strength = {strength}
"""
    return display
# end of function


def cook_burger(cook="well done"):
    display = f"Cooking a beef burger that is {cook}"
    return display
# end of function


def cook_fries(type_of_fries="straight"):
    display = f"Cooking {type_of_fries} fries"
    return display
# end of function


def cook_prawn_noodles(prawn="with prawn", sotong="with sotong"):
    display = f"""Cooking fried prawn noodles with the following options:
Prawn = {prawn},
Sotong = {sotong}
"""
    return display
# end of function


def extract_function_from_xml(xml_string):
    # Extract JSON from the tool_call string
    json_match = re.search(r'{.*}', xml_string)
    if json_match:
        json_str = json_match.group(0)

        # Parse the JSON string
        data = json.loads(json_str)

        # Extract function name and parameters
        tool_name = data.get('tool_name')
        tool_arguments = data.get('tool_arguments', {})

        # Create function call string with actual parameter values
        arguments = ', '.join([f"{key}='{value}'" for key, value in tool_arguments.items()])
        return f"{tool_name}({arguments})" if arguments else f"{tool_name}()"

    return None
# end of function


def create_system_prompt(tools_list):
    system_prompt_format = """You are a function calling AI model. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into function. The user may use the terms function calling or tool use interchangeably.

Here are the available functions:
<tools>{}</tools>

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags in the format:
<tool_call>{{"tool_name": "<function-name>", "tool_arguments": <args-dict>}}</tool_call>"""

    # Convert the tools list to a string representation with proper formatting
    tools_str = "\n".join([f"<tool>{tool}</tool>" for tool in tools_list])

    # Format the system prompt with the tools list
    system_prompt = system_prompt_format.format(tools_str)

    return system_prompt
# end of function


# login(token="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")
model_id = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load the model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config
)
print(f"The model is loaded on:? {model.device}")

tools_list = [
    {
        "name": "cook_burger",      # name of the function
        "description": "beef burger",
        "parameters": {
            "cook": {
                "description": "Can be well done, medium or rare",
                "type": "str",
                "default": "well done"
            }
        }
    },
    {
        "name": "cook_fries",
        "description": "Potatoes fries",
        "parameters": {
           "type_of_fries": {
                "description": "Can be straight or curly",
                "type": "str",
                "default": "straight"
            }
        }
    },
    {
        "name": "cook_prawn_noodles",
        "description": "Fried prawn noodles",
        "parameters":  {
            "prawn": {
                "description": "Options for prawn. Can be with or without prawn.",
                "type": "str",
                "default": "with prawn"
            },
            "sotong": {
                "description": "Options for sotong. Can be with or without sotong.",
                "type": "str",
                "default": "with sotong"
            }
        }
    },
    {
        "name": "make_coffee",
        "description": "Customer orders coffee.",
        "parameters": {
            "types_of_coffee": {
                "description": "The type of coffee. Examples are latte, americano, cappuccino.",
                "type": "str",
                "default": "long black"
            },
            "milk": {
                "description": "Options for milk with the coffee. Can be 'normal', 'no', 'more', 'less'.",
                "type": "str",
                "default": "normal"
            },
            "sugar": {
                "description": "Options for sugar with the coffee. Can be 'normal', 'no', 'more', 'less'.",
                "type": "str",
                "default": "normal"
            },
            "strength": {
                "description": "Options for coffee strength. Can be 'normal', 'strong', 'weak'.",
                "type": "str",
                "default": "normal"
            }
        }
    },
    {
        "name": "get_stock_price",
        "description": "Retrieves the current stock price given a stock symbol.",
        "parameters": {
            "symbol": {
                "description": "The stock symbol for which the price is what we wanted. Example of stock symbol is HPQ.",
                "type": "str",
                "default": ""
            }
        }
    },
    {
        "name": "get_weather",
        "description": "A function that retrieves the current weather for a given city.",
        "parameters": {
            "city": {
                "description": "The city which we want to know the weather  (e.g., 'New York' or 'Singapore').",
                "type": "str",
                "default": "Tokyo"
            }
        }
    }
]

# Create the system prompt with the tools list
system_prompt = create_system_prompt(tools_list)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",
     "content": "Can I have a plate of fried prawn noodles without sotong please?"}
]

prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

print(prompt)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print("Start llm\n")
time0 = time.time()
outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.01, eos_token_id=tokenizer.eos_token_id)
output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
time1 = time.time()
print(f"Time Taken: {time1-time0}s\n")
print(output_text)      # Expected output: <tool_call>{{"tool_name": "<function-name>", "tool_arguments": <args-dict>}}</tool_call>

answer = None   # initialize to Null
# Extract function to call from output_text. Expect function(key=value)
function = extract_function_from_xml(output_text)
print(f"Function to call is {function}")
result = "answer = " + function
exec(result)    # return value from function is in the variable answer
print(answer)
'''