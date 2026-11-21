from dotenv import load_dotenv
import dotenv
from langchain_core.prompts.prompt import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

def main():

    information = """
    Hashim Sujono Djojohadikusumo (lahir 5 Juni 1954) adalah pengusaha asal Indonesia. Hashim adalah anak bungsu dari pasangan Soemitro Djojohadikoesoemo dan Dora Marie Sigar,[1][2] serta adik dari Presiden Indonesia kedelapan yaitu Prabowo Subianto. Ia merupakan pemilik perusahaan Arsari Group. Hashim sendiri menjabat sebagai Wakil Ketua Dewan Pembina Partai Gerakan Indonesia Raya.
    Dari sekolah dasar hingga sekolah menengah, ia mengenyam pendidikan di Eropa sebelum melanjutkan pendidikan tinggi di bidang politik dan ekonomi di Pomona College, California, Amerika Serikat. Hashim menikah dengan Anie Harjati, dan dikaruniai 3 orang anak: Aryo, Sara, dan Indra. Hashim adalah penganut agama Kristen.
    Ayah Hashim adalah ekonom Indonesia, Soemitro Djojohadikoesoemo yang juga pernah menjabat sebagai menteri di era presiden Soekarno dan Soeharto. Sedangkan kakeknya, Margono Djojohadikusumo, adalah pendiri Bank BNI 1946 dan mantan ketua Dewan Pertimbangan Agung Sementara (DPAS) 1945 serta Badan Penyelidik Usaha-Usaha Persiapan Kemerdekaan.[4]
    Hashim adalah yang termuda dari empat bersaudara. Ia memiliki dua kakak perempuan, yaitu Biantiningsih dan Maryani Ekowati; serta satu kakak laki-laki yaitu Prabowo Subianto.
    """

    summary_template = """
    Berikan informasi {information} tentang orang yang saya ingin buat:
    1. Ringkasan singkat
    2. Dua hal menarik tentang mereka
    """

    llm = ChatOllama(
        model="gemma3:4b",
        temperature=0,
    )

    summary_promt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    chain = summary_promt_template | llm
    response = chain.invoke(input={"information": information})

    # response = llm.invoke("Halo! Ceritakan singkat tentang dirimu.")
    print(response.content)

if __name__ == "__main__":
    main()