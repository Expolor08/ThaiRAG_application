from google import genai
from google.genai import types
 
from Thaiwiki_ingest import build_index, load_documents
 
NOT_FOUND = "ไม่ทราบ"
 
INSTRUCTIONS = f'''
You are an assistant that answers questions in Thai using only the numbered
context excerpts below, which come from Thai Wikipedia.
 
Give an accurate, concise answer in Thai. If the question is written in another
language, answer in that language. Do not use knowledge from outside the context.
Mention the excerpts you used by number, for example [1] or [2].
If the answer is not found in the context, reply exactly: "{NOT_FOUND}"
'''.strip()
 
USER_PROMPT_TEMPLATE = '''
QUESTION: {question}
 
CONTEXT:
{context}
'''.strip()
 
BOOST = {"title": 1.0, "section": 2.0, "text": 1.0}
 
 
class RAGBase:
 
    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=USER_PROMPT_TEMPLATE,
        model="gemini-3.5-flash",
        boost=None,
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model
        self.boost = boost or BOOST
 
    def search(self, query, num_results=5, title=None):
        """Search the passages. title restricts the search to one Wikipedia page."""
        filter_dict = {"title": title} if title else {}
        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=self.boost,
            filter_dict=filter_dict,
        )
 
    def build_context(self, search_results):
        blocks = []
        for i, doc in enumerate(search_results, 1):
            blocks.append(
                f"[{i}] {doc['title']} > {doc['section']}\n"
                f"{doc['text']}\n"
                f"Source: {doc['url']}"
            )
        return "\n\n".join(blocks)
 
    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)
 
    def llm(self, prompt):
        message_history = [
            types.Content(role="user", parts=[types.Part(text=prompt)])
        ]
        response = self.llm_client.models.generate_content(
            model=self.model,
            contents=message_history,
            config=types.GenerateContentConfig(
                system_instruction=self.instructions
            ),
        )
        # response.text is None when the reply is blocked or empty
        return response.text or NOT_FOUND
 
    def rag(self, query, title=None):
        search_results = self.search(query, title=title)
        if not search_results:
            return NOT_FOUND
        prompt = self.build_prompt(query, search_results)
        return self.llm(prompt)
 
 
if __name__ == "__main__":
    documents = load_documents()        
    index = build_index(documents)
    client = genai.Client()             
 
    rag = RAGBase(index, client)
    print(rag.rag("เมืองหลวงของประเทศไทยคืออะไร"))
 

