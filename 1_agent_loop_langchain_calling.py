"""
1. Agent Loop dengan LangChain (.bind_tools)

File ini mendemonstrasikan cara paling ringkas membuat agent loop menggunakan LangChain:
- Mendefinisikan tool dengan dekorator `@tool` (schema JSON dibuat otomatis dari type hints & docstring).
- Menghubungkan tool ke model lewat `llm.bind_tools(tools)`.
- Menggunakan `SystemMessage`, `HumanMessage`, dan `ToolMessage` bawaan LangChain.
- Memproses `ai_message.tool_calls` satu per satu di dalam loop hingga model memberikan jawaban akhir.
"""

from langsmith import traceable
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:8b"

# -- Daftar Tools (menggunakan dekorator @tool dari LangChain) --
# Dekorator @tool otomatis membuat schema fungsi dari type hint dan docstring
# sehingga model tahu nama, deskripsi, dan parameter yang harus dikirim.

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""

    print(f"    >> Executing get_product_price(product='{product})")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
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

# -- Agent Loop --
# @traceable mencatat seluruh eksekusi loop agent ke LangSmith untuk monitoring/debugging.
@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    # Mapping nama tool -> objek tool agar mudah dipanggil saat LLM memilih tool
    tools_dict = {t.name: t for t in tools}

    # Inisialisasi model Ollama via LangChain dan ikatkan (bind) daftar tool ke model
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    # Riwayat percakapan dimulai dari instruksi sistem (SystemMessage) dan pertanyaan user (HumanMessage)
    messages = [
        SystemMessage(
            content=(
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
        ),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Kirim seluruh riwayat pesan ke LLM; model bisa membalas teks langsung atau meminta pemanggilan tool
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        # Jika tidak ada tool_calls, berarti model sudah menghasilkan jawaban akhir (Final Answer)
        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        # Proses hanya tool call PERTAMA agar eksekusi berjalan satu tool per iterasi
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f" [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if(tool_to_use) is None:
            raise ValueError(f"Tool '{tool_name} not found'")

        # Jalankan tool menggunakan .invoke(tool_args) milik LangChain
        try:
            observation = tool_to_use.invoke(tool_args)
        except Exception as e:
            observation = (
                f"Error executing {tool_name}: {e}\n"
                "Please fix the arguments or call the prerequisite tool "
                "(e.g., get_product_price) first."
            )

        print(f" [Tool Result] {observation}")

        # Simpan respons AI dan hasil eksekusi tool (ToolMessage) ke riwayat pesan untuk iterasi berikutnya
        messages.append(ai_message)
        messages.append(ToolMessage(
            name=tool_name,
            tool_call_id=tool_call_id,
            content=str(observation)
        ))

    print("ERROR: Maximum iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools) !")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")