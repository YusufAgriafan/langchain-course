"""
3. Agent Loop dengan Raw ReAct Prompt (Tanpa Native Tool/Function Calling)

File ini menunjukkan cara kerja agent ReAct (Reasoning + Acting) murni lewat teks prompt:
- Tidak mengirim parameter `tools=` ke `ollama.chat()`. Model hanya menerima satu prompt teks.
- Daftar tool dan parameternya disisipkan ke prompt sebagai teks biasa menggunakan `inspect`.
- Model diminta mengeluarkan format terstruktur: `Thought`, `Action`, `Action Input`, atau `Final Answer`.
- Stop sequence (`\\nObservation:`) mencegah model mengarang hasil observasi sendiri.
- Program mem-parsing output LLM dengan regex (`re.search`), menjalankan fungsi Python, lalu
  menyuntikkan hasilnya ke `scratchpad` sebagai `Observation:` untuk iterasi berikutnya.
"""

from langsmith import traceable
from dotenv import load_dotenv
import inspect
import json
import re
import ollama

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:8b"

# -- Daftar Tools (fungsi Python biasa + @traceable untuk LangSmith) --

@traceable(run_type="tool", name="get_product_price")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""

    print(f"    >> Executing get_product_price(product='{product})")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@traceable(run_type="tool", name="apply_discount")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a numeric price and return the final price.
    Only call this AFTER get_product_price has returned a numeric price.

    Args:
        price: Numeric price returned by get_product_price (must be a float, not a tool name or string).
        discount_tier: Discount tier to apply (bronze, silver, or gold).
    """
    print(f"    >> Executing apply_discount(price={price}, discount_tier={discount_tier})")
    price = float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

# Perubahan 3: Tidak lagi menggunakan JSON schema. Deskripsi tool dimasukkan langsung ke dalam prompt sebagai teks biasa.
# Kita mengambil signature dan docstring langsung dari fungsi menggunakan modul `inspect`.

def get_tools_description(tools_dict):
    """Membuat deskripsi teks seluruh tool untuk disisipkan ke prompt ReAct."""
    descriptions = []
    for tool_name, tool_func in tools_dict.items():
        # __wrapped__ dipakai untuk mengambil fungsi asli jika fungsi dibungkus dekorator (@traceable)
        original_function = getattr(tool_func, "__wrapped__", tool_func)
        signature = str(inspect.signature(original_function))
        docstring = inspect.getdoc(original_function) or "No description available."
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions) 

tool_descriptions = get_tools_description(tools)
tool_names = ", ".join(tools.keys())
react_prompt = """
STRICT RULES - you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price - do NOT pass a made-up number or a tool name.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use - do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:{agent_scratchpad}
"""
# Perubahan 4: Hapus parameter `tools=` dari `ollama.chat()`.
# LLM tidak menggunakan native tool calling—seluruh perilaku agent berasal dari prompt ReAct di atas dan parsing regex di bawah.

@traceable(run_type="llm", name="Ollama Chat")
def ollama_chat_traced(messages, model=MODEL, options=None):
    return ollama.chat(
        model=model,
        messages=messages,
        options=options or {"stop": ["Observation:"]})


def parse_react_output(text: str):
    """Parse a ReAct-formatted LLM response.

    Returns (action, action_input, final_answer). Exactly one of (action, final_answer)
    will be non-None on a well-formed response.
    """
    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_match:
        return None, None, final_match.group(1).strip()

    action_match = re.search(r"Action:\s*(.+?)\s*\n", text)
    input_match = re.search(r"Action Input:\s*(.+?)(?:\n(?:Observation|Thought|Action):|\Z)", text, re.DOTALL)
    if action_match and input_match:
        raw_input = input_match.group(1).strip()
        try:
            parsed_input = json.loads(raw_input)
        except json.JSONDecodeError:
            parsed_input = raw_input
        return action_match.group(1).strip(), parsed_input, None

    return None, None, None

# -- Agent Loop (ReAct berbasis teks & scratchpad) --

@traceable(name="Langchain Agent Loop")
def run_agent(question: str):

    print(f"Question: {question}")
    print("=" * 60)

    # Perubahan 5: Satu string prompt menggantikan pemisahan pesan system/user.
    # Riwayat Thought/Action/Observation disimpan di variabel `scratchpad`.
    prompt = react_prompt.format(question=question, tool_descriptions=tool_descriptions, tool_names=tool_names, agent_scratchpad="")
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        full_prompt = prompt + scratchpad

        # Stop token ("\nObservation:") menghentikan LLM sebelum mengarang Observation sendiri;
        # kita akan menjalankan tool sungguhan dan menyuntikkan hasilnya secara manual.
        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation:"], "temperature": 0.0}
        )
        output = response.message.content
        print(f"LLM Output:\n{output}")

        print(f"    [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output, re.DOTALL)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"    [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        # Perubahan 6: Ekstrak Action dan Action Input dari teks mentah menggunakan regex.
        # Cara ini bergantung pada kepatuhan LLM mengikuti format ReAct.
        print(f"    [Parsing] Looking for Action and Action Input in LLM output...")
        action_match = re.search(r"Action:\s*(.+?)\s*\n", output)
        action_input_match = re.search(r"Action Input:\s*(.+?)(?:\n(?:Observation|Thought|Action):|\Z)", output, re.DOTALL)

        if not action_match or not action_input_match:
            print(
                "    [Parsing] ERROR: Could not find Action or Action Input in LLM output. "
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f" [Tool Selected] {tool_name} with args: {tool_input_raw}")

        # Coba parse Action Input sebagai JSON terlebih dahulu (misal {"price": 1299.99, "discount_tier": "gold"})
        try:
            parsed_input = json.loads(tool_input_raw)
        except json.JSONDecodeError:
            parsed_input = None

        print(f" [Tool Executing] {tool_name}({tool_input_raw})...")
        if tool_name not in tools:
            observation = f"ERROR: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
            print(observation)
        elif isinstance(parsed_input, dict):
            # Jika JSON object, kirim sebagai keyword arguments (**parsed_input)
            observation = str(tools[tool_name](**parsed_input))
        elif isinstance(parsed_input, list):
            # Jika JSON array, kirim sebagai positional arguments (*parsed_input)
            observation = str(tools[tool_name](*parsed_input))
        else:
            # Fallback: pisahkan argumen dengan koma, lalu hapus prefix key= dan tanda kutip jika ada
            args = [x.split("=", 1)[-1].strip().strip("'\"") for x in tool_input_raw.split(",")]
            observation = str(tools[tool_name](*args))

        print(f" [Tool Result] {observation}")

        # Perubahan 7: Riwayat disimpan sebagai satu string yang terus bertambah di setiap iterasi (menggantikan messages.append).
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("ERROR: Maximum iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools) !")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")