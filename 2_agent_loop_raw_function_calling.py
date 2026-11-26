"""
2. Agent Loop dengan Raw Function Calling (Ollama SDK langsung tanpa LangChain)

File ini menunjukkan cara kerja agent loop secara manual tanpa abstraksi LangChain:
- Menggunakan `ollama.chat()` langsung dengan parameter `tools=tools_for_llm`.
- Mendefinisikan JSON schema tool secara manual (tanpa `@tool`).
- Menambahkan tracing LangSmith secara manual lewat `@traceable` pada setiap tool dan panggilan LLM.
- Memanggil fungsi Python langsung dengan `tool_to_use(**tool_args)` alih-alih `.invoke()`.
"""

from langsmith import traceable
from dotenv import load_dotenv
import ollama

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:8b"

# -- Daftar Tools (fungsi Python biasa + @traceable untuk LangSmith) --
# Perbedaan 1: Tanpa @tool dari LangChain, kita memakai fungsi Python biasa
# dan menandainya dengan @traceable agar tetap tercatat di LangSmith.

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
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# Perbedaan 2: Tanpa @tool, kita harus mendefinisikan metadata tool secara MANUAL
# dalam format JSON Schema (nama, deskripsi, dan skema parameter) agar LLM tahu cara memanggilnya.
tools_for_llm = [
        {
            "type": "function",
            "function":{
                "name": "get_product_price",
                "description": "Look up the price of a product in the catalog.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product": {
                            "type": "string",
                            "description": "The name of the product to look up."
                        }
                    },
                    "required": ["product"]
                }
            }
        },
        {
            "type": "function",
            "function":{
                "name": "apply_discount",
                "description": "Apply a discount to a numeric price and return the final price.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "price": {
                            "type": "number",
                            "description": "The price to apply the discount to."
                        },
                        "discount_tier": {
                            "type": "string",
                            "description": "The discount tier to apply (bronze, silver, or gold)."
                        }
                    },
                    "required": ["price", "discount_tier"]
                }
            }
        }
    ]

# CATATAN: Ollama sebenarnya juga bisa membuat schema otomatis jika kita langsung
# mengoper fungsi Python ke `tools=[get_product_price, apply_discount]` (mirip @tool di LangChain),
# selama fungsi memiliki type hint dan docstring (bagian Args).
# Di sini schema tetap ditulis manual untuk menunjukkan struktur asli di balik layar.

# --- Helper: panggilan Ollama dengan tracing LangSmith ---
# Perbedaan 3: Tanpa LangChain, kita harus membungkus panggilan LLM dengan @traceable secara manual
# agar setiap request/response ke Ollama tetap terekam di LangSmith.

@traceable(run_type="llm", name="Ollama Chat")
def ollama_chat_traced(messages):
    return ollama.chat(
        model=MODEL,
        tools=tools_for_llm,
        messages=messages)

# -- Agent Loop --

@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }

    print(f"Question: {question}")
    print("=" * 60)

    # Perbedaan 4: Menggunakan dict {"role": ..., "content": ...} alih-alih SystemMessage/HumanMessage
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price - do NOT pass a made-up number "
                "or a tool name.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use - do NOT assume one."
            )
        },
        {"role": "user", "content": question}
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Perbedaan 5: Memanggil ollama.chat() langsung (via helper) alih-alih llm_with_tools.invoke()
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        # Jika tidak ada tool_calls, berarti model sudah memberikan jawaban akhir
        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        # Proses hanya tool call PERTAMA agar eksekusi tetap satu tool per iterasi
        tool_call = tool_calls[0]
        # Perbedaan 6: Akses atribut objek (.function.name) alih-alih akses dict (.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f" [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if(tool_to_use) is None:
            raise ValueError(f"Tool '{tool_name} not found'")

        try:
            # Perbedaan 7: Memanggil fungsi Python langsung dengan **tool_args alih-alih .invoke()
            observation = tool_to_use(**tool_args)
        except Exception as e:
            observation = (
                f"Error executing {tool_name}: {e}\n"
                "Please fix the arguments or call the prerequisite tool "
                "(e.g., get_product_price) first."
            )

        print(f" [Tool Result] {observation}")

        # Simpan pesan AI dan hasil tool (role="tool") ke riwayat percakapan
        messages.append(ai_message)
        messages.append({
            "role": "tool",
            "name": tool_name,
            "content": str(observation)
        })

    print("ERROR: Maximum iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools) !")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")