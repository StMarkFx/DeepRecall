import ollama
from collections import deque
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

class DeepSeekModel:
    def __init__(self, model_name="deepseek-r1:1.5b", memory_size=5):
        self.model_name = model_name
        self.memory = deque(maxlen=memory_size)  # Store last 'n' messages

        # Initialize LangChain memory for extended conversation
        self.langchain_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # System-wide prompt for accuracy & structure
        self.system_prompt = """You are an AI research assistant for students and researchers.
        Follow these rules:
        - Retrieve the most relevant information from provided documents.
        - Cite sources when using retrieved content.
        - Provide structured responses using bullet points, lists, or concise paragraphs.
        - If the question is unclear, ask a clarifying question first.
        - If requested, summarize documents or explain in simpler terms.
        - Maintain context from previous conversations."""

        # Define a structured prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["chat_history", "context", "question"],
            template="""
            {system_prompt}
            
            Context from retrieved documents:
            {context}
            
            Conversation history:
            {chat_history}
            
            User's new question:
            {question}
            
            Your response:
            """
        )

        # Load DeepSeek model via Ollama
        self.llm = ChatOllama(model=self.model_name, temperature=0.3)

        # Create LangChain LLM chain
        self.chat_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            memory=self.langchain_memory
        )

    def generate_response(self, query, context):
        """Generates a response using DeepSeek AI with memory & retrieval."""
        # Save user query in short-term memory
        self.memory.append({"role": "user", "content": query})

        # Construct conversation history
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.memory])

        # Prepare enhanced prompt
        formatted_prompt = self.prompt_template.format(
            system_prompt=self.system_prompt,
            context=context,
            chat_history=history,
            question=query
        )

        # Get AI response from DeepSeek
        response = self.chat_chain.run(question=query, context=context)

        # Save AI response to memory
        self.memory.append({"role": "assistant", "content": response})

        return response
