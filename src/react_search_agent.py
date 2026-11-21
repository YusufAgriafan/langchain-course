from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing import List

load_dotenv()

# ==============================================================================
# REFERENSI 1: Custom Tool menggunakan TavilyClient (Kode sebelumnya)
# ==============================================================================
# tavily = TavilyClient()
#
# @tool
# def search_custom(query: str) -> str:
#     """Tool that searches over the internet using TavilyClient."""
#     print(f"\n[Tool Search] Sedang mencari: {query}")
#     return tavily.search(query=query)


# ==============================================================================
# REFERENSI 2: Mock Search Tool statis (Kode awal)
# ==============================================================================
# @tool
# def search_mock(query: str) -> str:
#     """Tool dummy yang mengembalikan string statis."""
#     print(f"Sedang mencari {query}")
#     return "Kualitas udara di beberapa kota di Indonesia sedang mengalami penurunan akibat polusi."


# ==============================================================================
# REFERENSI 3: TavilySearch Bawaan (Default tanpa penyesuaian schema)
# Catatan: Cara ini bisa memicu error "time_range: Input should be 'day'..."
# pada model lokal kecil seperti llama3.2:3b karena model sering mengirim nilai "null".
# ==============================================================================
# default_tavily_search = TavilySearch()


# ==============================================================================
# OPSI UTAMA (AKTIF): TavilySearch dengan Schema yang Disesuaikan
# ==============================================================================
class TavilyQueryInput(BaseModel):
    """Skema input khusus query agar model lokal tidak bingung

    dengan parameter opsional (seperti time_range='null').
    """

    query: str = Field(description="Search query to look up on the web")


# Inisialisasi TavilySearch dengan schema yang aman untuk model lokal
tavily_search = TavilySearch(args_schema=TavilyQueryInput, max_results=5)

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer:str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

# ==============================================================================
# KONFIGURASI MODEL & AGENT
# ==============================================================================
# Model lokal Ollama
llm = ChatOllama(model="llama3.2:3b", temperature=0)

# Pilih tool yang ingin digunakan (aktifkan salah satu):
tools = [tavily_search]  # Aktif: TavilySearch resmi yang aman
# tools = [search_custom]         # Referensi: Custom tool TavilyClient
# tools = [search_mock]           # Referensi: Mock search
# tools = [default_tavily_search] # Referensi: Default TavilySearch

agents = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main() -> None:
    print("Menjalankan agent lokal dengan Ollama & TavilySearch...")
    query = (
        "cari postingan job tentang perusahaan besar yang sedang mencari "
        "software engineer atau ai engineer baik di linkedin atau platform lain"
    )

    # Kirim query ke agent
    result = agents.invoke({"messages": [HumanMessage(content=query)]})

    # Cetak respons terakhir dari agent
    print("\n--- Jawaban Agent ---")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
